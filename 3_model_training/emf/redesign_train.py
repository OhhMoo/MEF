"""Source training and frozen campaign evaluation for calibrated molecular flows."""

import argparse
from contextlib import ExitStack
import csv
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

from .campaigns import CampaignDataset, collate_episodes, neighbourhood
from .contracts import ModelConfig
from .data import collate_graphs
from .models import EMFModel
from .redesign import RedesignModel
from .redesign_contracts import FlowCondition, RedesignConfig, SupportBatch
from .redesign_evaluation import generate_redesign_proposals, evaluate_committed_proposals
from .redesign_objectives import redesign_loss
from .redesign_sampling import frozen_deployment, make_sampling_plan
from .toy_oracle import ToyOracle


ROOT = Path(os.environ.get("EMF_WORKSPACE_ROOT", Path.cwd())).resolve()


def project_path(path):
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def tensor_fingerprint(items):
    result = hashlib.sha256()
    for name, value in sorted(items):
        result.update(name.encode())
        array = value.detach().cpu().contiguous().numpy()
        result.update(str(array.dtype).encode())
        result.update(str(array.shape).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def representation_fingerprint(model):
    return tensor_fingerprint((name, value) for name, value in model.state_dict().items()
                              if name.startswith(("encoder.", "normalizer.")))


def verified_cache_digest(dataset):
    records=dataset.manifest["splits"][dataset.split]
    actual=digest(dataset.data_dir/records["file"])
    if actual!=records["sha256"]:
        raise ValueError("campaign observation cache differs from its manifest")
    return actual


@dataclass
class RedesignRunConfig:
    redesign: RedesignConfig = field(default_factory=RedesignConfig)
    data_dir: str = "2_data_storage/processed/calibrated_redesign_toy"
    initializer: str = "4_experimental_result/result/exploration_001/checkpoints/flow_0004000.pt"
    output_dir: str = "4_experimental_result/result/calibrated_redesign_001/pilot"
    seed: int = 42
    episodes: int = 8
    queries_per_lead: int = 4
    train_steps: int = 20000
    train_seconds: float | None = 1800.
    learning_rate: float = 3e-4
    T: float = .95
    sigma: float = .05
    beta: float = 1.
    gamma: float = 1.
    grad_clip: float = 1.
    bf16: bool = True
    cpu_threads: int = 8
    device: str = "cuda"
    cache_batch_size: int = 512
    log_every: int = 100
    checkpoint_every: int = 2000
    evaluate_after_training: bool = False
    eval_split: str = "validation"
    eval_campaigns: int = 8
    leads_per_campaign: int = 2
    support_sizes: list[int] = field(default_factory=lambda: [16,32,64,128])
    eval_seed: int = 271828
    num_attempts: int = 1000
    nfe: int = 32
    k: int = 16
    sample_batch_size: int = 128
    evaluation_modes: list[str] = field(default_factory=lambda: ["native"])
    adaptation_steps: int = 64
    common_scorer_checkpoint: str | None = None

    def __post_init__(self):
        if min(self.episodes, self.queries_per_lead, self.cpu_threads,
               self.cache_batch_size, self.log_every, self.checkpoint_every,
               self.eval_campaigns, self.leads_per_campaign, self.k) < 1:
            raise ValueError("batch, interval, evaluation and thread counts must be positive")
        if self.train_steps < 0 or self.num_attempts < 1 or self.nfe < 2:
            raise ValueError("invalid training/attempt/NFE counts")
        if not 0 < self.T < 1 or min(self.learning_rate, self.beta, self.gamma, self.grad_clip) <= 0:
            raise ValueError("invalid time or optimization weights")
        if self.train_seconds is not None and self.train_seconds <= 0:
            raise ValueError("train_seconds must be positive")
        if self.eval_split not in {"validation", "test"}:
            raise ValueError("evaluation must use validation or test campaigns")
        if not self.support_sizes or min(self.support_sizes) < 1:
            raise ValueError("support sizes must be positive")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")


def load_redesign_config(path):
    values = yaml.safe_load(project_path(path).read_text())
    architecture = values.pop("redesign", {})
    architecture["model"] = ModelConfig(**architecture.get("model", {}))
    values["redesign"] = RedesignConfig(**architecture)
    return RedesignRunConfig(**values)


def load_initializer(config, device):
    path = project_path(config.initializer)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["config"]["model"] != asdict(config.redesign.model):
        raise ValueError("initializer graph architecture differs")
    base = EMFModel(config.redesign.model)
    base.load_state_dict(checkpoint["model"])
    if not base.normalizer.fitted.item():
        raise ValueError("initializer normalization is not source fitted")
    base.freeze_encoder()
    model = RedesignModel(config.redesign, representation=base).to(device)
    return model, checkpoint


def load_source_checkpoint(path, device):
    payload = torch.load(project_path(path), map_location="cpu", weights_only=False)
    values = dict(payload["config"])
    architecture = dict(values["redesign"])
    architecture["model"] = ModelConfig(**architecture["model"])
    values["redesign"] = RedesignConfig(**architecture)
    config = RedesignRunConfig(**values)
    model = RedesignModel(config.redesign).to(device)
    model.load_state_dict(payload["model"])
    if representation_fingerprint(model) != payload["representation_sha256"]:
        raise ValueError("source checkpoint representation fingerprint differs")
    return model, config, payload


@torch.no_grad()
def cache_graph_features(model, dataset, *, batch_size=512, bf16=True):
    """Frozen learned E features only; no oracle descriptors or outcomes."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    chunks = []
    started = time.perf_counter()
    for first in range(0, len(dataset), batch_size):
        batch = collate_graphs([dataset[i] for i in range(
            first, min(first+batch_size, len(dataset)))]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=bf16 and device.type == "cuda"):
            chunks.append(model.graph_features(batch).cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model.train(was_training)
    features = torch.cat(chunks)
    if not torch.isfinite(features).all():
        raise FloatingPointError("nonfinite frozen graph features")
    return features, time.perf_counter()-started


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key:(json.dumps(value) if isinstance(value,(dict,list)) else value)
                         for key,value in row.items()} for row in rows)


def _rng_payload(generator, episode_rng):
    return {"torch":torch.get_rng_state(),
            "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "noise":generator.get_state(), "episodes":episode_rng.bit_generator.state}


def _save_checkpoint(path, model, optimizer, config, step, training_seconds,
                     generator, episode_rng, data_hash, metrics):
    path = Path(path)
    if path.name == "source_final.pt" and path.exists():
        previous = torch.load(path, map_location="cpu", weights_only=False)
        if (previous["config"] == asdict(config) and previous["step"] == step
                and previous["data_sha256"] == data_hash
                and tensor_fingerprint(previous["model"].items())
                    == tensor_fingerprint(model.state_dict().items())):
            return
        raise FileExistsError("final source checkpoint is immutable; use a fresh output directory")
    payload = {"format_version":1, "config":asdict(config), "model":model.state_dict(),
               "optimizer":optimizer.state_dict(), "step":step,
               "training_seconds":training_seconds, "rng":_rng_payload(generator,episode_rng),
               "data_sha256":data_hash, "representation_sha256":representation_fingerprint(model),
               "metrics":metrics, "created_utc":datetime.now(timezone.utc).isoformat()}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_training(config, resume=None, *, dataset_class=None, objective=None,
                 goal_rule=None, save_step_checkpoints=False, model_factory=None):
    """Train with explicit optional dataset/objective injection for new studies.

    Defaults preserve the calibrated-redesign protocol. An injected objective
    receives the sampled episodes as well as the collated batch, so a separately
    labelled simulator diagnostic can supply privileged source information.
    """
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested without an allocated usable GPU")
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(device)
    output = project_path(config.output_dir)
    if (output/"checkpoints/latest.pt").exists() and resume is None:
        raise FileExistsError("existing source checkpoint; use a fresh path or explicit resume")
    data_hash = digest(project_path(config.data_dir)/"manifest.json")
    previous = None
    if resume:
        previous = torch.load(project_path(resume),map_location="cpu",weights_only=False)
        if previous["config"] != asdict(config) or previous["data_sha256"] != data_hash:
            raise ValueError("resume configuration/data differ")
    final_path = output/"checkpoints/source_final.pt"
    if final_path.exists():
        final = torch.load(final_path,map_location="cpu",weights_only=False)
        if (previous is None or final["config"] != asdict(config)
                or final["data_sha256"] != data_hash or final["step"] != previous["step"]
                or tensor_fingerprint(final["model"].items())
                    != tensor_fingerprint(previous["model"].items())):
            raise FileExistsError("final source checkpoint is immutable; resume it or use a fresh path")
        report = json.loads((output/"source_summary.json").read_text())
        if config.evaluate_after_training:
            run_evaluation(config,final_path)
        return report
    output.mkdir(parents=True, exist_ok=True)
    model, initializer = (model_factory or load_initializer)(config,device)
    initial_representation = representation_fingerprint(model)
    source = (dataset_class or CampaignDataset)(project_path(config.data_dir), "train")
    verified_cache_digest(source)
    (output/"config.yaml").write_text(yaml.safe_dump(asdict(config),sort_keys=False))
    atomic_json(output/"manifest.json", {
        "config":asdict(config), "created_utc":datetime.now(timezone.utc).isoformat(),
        "slurm_job_id":os.environ.get("SLURM_JOB_ID"),
        "initializer_sha256":digest(project_path(config.initializer)),
        "data_sha256":data_hash, "representation_sha256":initial_representation,
        "parameter_count":sum(p.numel() for p in model.parameters()),
        "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
        "source_hashes":{p.name:digest(p) for p in Path(__file__).parent.glob("*.py")},
        "upstream":json.loads((ROOT/"0_external_repo/provenance.json").read_text()),
        "goal_rule":goal_rule or "outcome-independent uniform prior; exact feasibility flow weights",
        "dataset_class":type(source).__module__+"."+type(source).__name__,
        "objective":(getattr(objective, "__name__", "injected") if objective else "redesign_loss"),
        "save_step_checkpoints":save_step_checkpoints,
        "response_queries":"all complete query pairs",
        "count_prior":source.count_prior.tolist(),
        "source_training_observed_compounds":len(source),
        "source_campaigns":len(source.campaign_ids),
    })
    features, cache_seconds = cache_graph_features(model,source.graph_dataset,
                 batch_size=config.cache_batch_size,bf16=config.bf16)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config.learning_rate,weight_decay=0)
    episode_rng = np.random.default_rng(config.seed+1000)
    noise_rng = torch.Generator(device="cpu").manual_seed(config.seed+2000)
    first_step, training_seconds = 0, 0.
    if previous is not None:
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        first_step,training_seconds = previous["step"],previous["training_seconds"]
        torch.set_rng_state(previous["rng"]["torch"])
        if device.type=="cuda":
            torch.cuda.set_rng_state_all(previous["rng"]["cuda"])
        noise_rng.set_state(previous["rng"]["noise"])
        episode_rng.bit_generator.state = previous["rng"]["episodes"]
    history_path = output/"training.csv"
    history = []
    if history_path.exists() and resume:
        with history_path.open() as stream:
            history=list(csv.DictReader(stream))
    model.train()
    metrics = dict(previous.get("metrics",{})) if previous is not None else {}
    step = first_step
    started = time.perf_counter()
    print(json.dumps({"event":"source_training","device":str(device),
          "variant":config.redesign.variant,"cache_seconds":cache_seconds,
          "parameters":sum(p.numel() for p in model.parameters())}),flush=True)
    for step in range(first_step+1,config.train_steps+1):
        if config.train_seconds is not None and training_seconds>=config.train_seconds:
            step -= 1
            break
        tick = time.perf_counter()
        episodes = source.sample_batch(episode_rng,episodes=config.episodes,
                                      queries_per_lead=config.queries_per_lead)
        batch = collate_episodes(episodes,pooled_features=features,device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,
                            enabled=config.bf16 and device.type=="cuda"):
            loss_kwargs = dict(T=config.T,sigma=config.sigma,
                       beta=config.beta,gamma=config.gamma,generator=noise_rng)
            if objective is None:
                loss, values = redesign_loss(model,batch,**loss_kwargs)
            else:
                loss, values = objective(model,batch,episodes=episodes,**loss_kwargs)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite source loss at update {step}")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],config.grad_clip)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"nonfinite source gradient at update {step}")
        optimizer.step()
        if device.type=="cuda":
            torch.cuda.synchronize(device)
        training_seconds += time.perf_counter()-tick
        metrics={key:float(value) for key,value in values.items()}
        if step==1 or step%config.log_every==0 or step==config.train_steps:
            row={"step":step,"training_seconds":training_seconds,
                 "wall_seconds":time.perf_counter()-started,"gradient_norm":float(norm),**metrics}
            history.append(row)
            _write_csv(history_path,history)
            print(json.dumps(row),flush=True)
        if step%config.checkpoint_every==0:
            _save_checkpoint(output/"checkpoints/latest.pt",model,optimizer,config,step,
                             training_seconds,noise_rng,episode_rng,data_hash,metrics)
            if save_step_checkpoints:
                _save_checkpoint(output/"checkpoints"/f"step_{step:07d}.pt",model,optimizer,config,step,
                                 training_seconds,noise_rng,episode_rng,data_hash,metrics)
        if step==1 or step%config.log_every==0:
            atomic_json(output/"status.json",{"phase":"source_training","step":step,
                         "training_seconds":training_seconds,"metrics":metrics})
    if representation_fingerprint(model)!=initial_representation:
        raise AssertionError("source training changed frozen E/normalization")
    if model.decoder.training is False or not any(p.requires_grad for p in model.decoder.parameters()):
        raise AssertionError("source decoder must remain trainable")
    for name in ("latest.pt","source_final.pt"):
        _save_checkpoint(output/"checkpoints"/name,model,optimizer,config,step,
                         training_seconds,noise_rng,episode_rng,data_hash,metrics)
    report={"phase":"source_complete","step":step,"training_seconds":training_seconds,
            "feature_cache_seconds":cache_seconds,"wall_seconds":time.perf_counter()-started+cache_seconds,
            "metrics":metrics,"frozen_representation_unchanged":True,
            "peak_torch_memory_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None}
    atomic_json(output/"source_summary.json",report)
    atomic_json(output/"status.json",report)
    print(json.dumps(report),flush=True)
    if config.evaluate_after_training:
        run_evaluation(config,output/"checkpoints/source_final.pt")
    return report


def _support(records, features, device):
    indices=torch.as_tensor(records["indices"],dtype=torch.long)
    size=len(indices)
    return SupportBatch(features.index_select(0,indices)[None].to(device),
             torch.as_tensor(records["values"],dtype=torch.float32,device=device)[None],
             torch.as_tensor(records["context"],dtype=torch.float32,device=device)[None],
             torch.ones((1,size),dtype=torch.bool,device=device))


def _control_functions():
    directory=Path(os.environ.get("EMF_CONTROL_DIR",ROOT/"4_experimental_result/control"))
    if str(directory) not in sys.path:
        sys.path.insert(0,str(directory))
    from redesign_adaptation import adapt_task_state,adapt_selector
    return adapt_task_state,adapt_selector


def _search_function():
    directory=Path(os.environ.get("EMF_CONTROL_DIR",ROOT/"4_experimental_result/control"))
    if str(directory) not in sys.path:
        sys.path.insert(0,str(directory))
    from redesign_search import generate_predictor_search_proposals
    return generate_predictor_search_proposals


def run_evaluation(config, checkpoint):
    torch.set_num_threads(config.cpu_threads)
    device=torch.device(config.device)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("evaluation CUDA requires allocated GPU")
    if device.type=="cuda":
        torch.set_float32_matmul_precision("high")
    generator,source_config,_=load_source_checkpoint(checkpoint,device)
    if config.common_scorer_checkpoint:
        scorer,_,_=load_source_checkpoint(config.common_scorer_checkpoint,device)
    else:
        scorer=generator
    dataset=CampaignDataset(project_path(config.data_dir),config.eval_split)
    cache_hash=verified_cache_digest(dataset)
    features,cache_seconds=cache_graph_features(scorer,dataset.graph_dataset,
                           batch_size=config.cache_batch_size,bf16=config.bf16)
    # Keep the original molecular proposal machinery for its standard-generator control.
    payload=torch.load(project_path(config.initializer),map_location="cpu",weights_only=False)
    standard_base=EMFModel(config.redesign.model).to(device)
    standard_base.load_state_dict(payload["model"])
    standard_base.freeze_encoder()
    output=project_path(config.output_dir)
    eval_root=output/f"evaluation_{config.eval_split}"
    eval_root.mkdir(parents=True,exist_ok=True)
    eval_rng=np.random.default_rng(config.eval_seed+(0 if config.eval_split=="validation" else 100000))
    n_campaigns=min(config.eval_campaigns,len(dataset.campaign_ids))
    cases=[]
    for campaign in range(n_campaigns):
        leads=eval_rng.choice(dataset.eligible_leads,size=config.leads_per_campaign,replace=False)
        for lead in leads:
            cases.append((campaign,int(lead)))
    protocol={"eval_split":config.eval_split,"support_sizes":config.support_sizes,
        "num_attempts":config.num_attempts,"k":config.k,"nfe":config.nfe,
        "T":config.T,"eval_seed":config.eval_seed,"modes":config.evaluation_modes,
        "cases":[{"campaign":dataset.campaign_ids[c],"lead_index":l,"lead_id":dataset.ids[l]}
                 for c,l in cases],"generator_checkpoint_sha256":digest(project_path(checkpoint)),
        "common_scorer_checkpoint_sha256":digest(project_path(config.common_scorer_checkpoint or checkpoint)),
        "initializer_sha256":digest(project_path(config.initializer)),
        "data_manifest_sha256":digest(project_path(config.data_dir)/"manifest.json"),
        "evaluation_cache_sha256":cache_hash,
        "private_oracle_sha256":dataset.manifest["private_oracle_sha256"],
        "conditional_count_prior":dataset.count_prior.tolist(),
        "sample_batch_size":config.sample_batch_size,"adaptation_steps":config.adaptation_steps,
        "numerical_policy":{"device":config.device,"flow_sampling":"float32",
             "frozen_encoder":"float32","float32_matmul_precision":torch.get_float32_matmul_precision()},
        "feature_cache_seconds":cache_seconds,"blind_protocol_committed_utc":datetime.now(timezone.utc).isoformat()}
    protocol_path=eval_root/"protocol.json"
    if protocol_path.exists():
        old_protocol=json.loads(protocol_path.read_text())
        scientific=lambda value:{key:item for key,item in value.items()
             if key not in {"feature_cache_seconds","blind_protocol_committed_utc"}}
        if scientific(old_protocol)!=scientific(protocol):
            raise FileExistsError("evaluation protocol differs; use a fresh output directory")
        protocol=old_protocol
    else:
        atomic_json(protocol_path,protocol)
    pending=[]
    with ExitStack() as stack:
        for model in {id(m):m for m in (generator,scorer,standard_base)}.values():
            stack.enter_context(frozen_deployment(model))
        model_before=tensor_fingerprint(scorer.state_dict().items())
        for case_number,(campaign,lead_index) in enumerate(cases):
            nested=dataset.nested_support(campaign,lead_index,seed=config.eval_seed+case_number,
                                          sizes=config.support_sizes)
            lead=collate_graphs([nested["lead"]]).to(device)
            baseline=torch.as_tensor(nested["lead_values"],dtype=torch.float32,device=device)[None]
            context=torch.as_tensor(dataset.context,dtype=torch.float32,device=device)[None]
            goal=torch.stack([torch.full_like(baseline[:,0],.25),baseline[:,1]-.1],-1)
            lead_tick=time.perf_counter()
            # Shared frozen E supplies one authoritative lead encoding for this case.
            # Re-encoding identical weights can differ by numerical roundoff on CUDA.
            lead_features=scorer.graph_features(lead)
            if device.type=="cuda":
                torch.cuda.synchronize(device)
            lead_encoding_seconds=time.perf_counter()-lead_tick
            plan=make_sampling_plan(scorer.config,dataset.count_prior,int(lead.counts[0]),
                  config.num_attempts,count_seed=config.eval_seed+case_number*2,
                  noise_seed=config.eval_seed+case_number*2+1)
            for size in config.support_sizes:
                support=_support(nested["records"][size],features,device)
                tick=time.perf_counter()
                with torch.no_grad():
                    native_state=scorer.support_state(support)
                    generator_state=generator.support_state(support)
                    condition=FlowCondition(lead_features,baseline,context,native_state,goal)
                    generator_condition=replace(condition,response_state=generator_state)
                if device.type=="cuda":
                    torch.cuda.synchronize(device)
                inference_seconds=time.perf_counter()-tick
                for mode in config.evaluation_modes:
                    destination=eval_root/f"case_{case_number:03d}"/f"support_{size:03d}"/mode
                    if (destination/"evaluation_summary.json").exists():
                        continue
                    kwargs={}
                    active_scorer,active_condition=scorer,condition
                    adaptation={}
                    if mode=="native":
                        if generator is not scorer:
                            kwargs.update(proposal_model=generator,proposal_condition=generator_condition)
                    elif mode=="swapped":
                        other_campaign=(campaign+1)%len(dataset.campaign_ids)
                        wrong_records=dict(nested["records"][size])
                        wrong_records["values"]=dataset.observations[other_campaign,wrong_records["indices"]]
                        with torch.no_grad():
                            wrong_state=scorer.support_state(_support(wrong_records,features,device))
                        kwargs["flow_state_override"]=wrong_state
                    elif mode=="base_only":
                        kwargs["base_only"]=True
                    elif mode=="standard_proposals":
                        kwargs["proposal_model"]=standard_base
                    elif mode=="task_embedding":
                        fit_state,_=_control_functions()
                        adapted_state,adaptation=fit_state(scorer,support,
                            collate_graphs(nested["records"][size]["graphs"]).to(device),
                            steps=config.adaptation_steps)
                        kwargs["flow_state_override"]=adapted_state
                    elif mode=="selector_adaptation":
                        _,fit_selector=_control_functions()
                        adapted,adapted_state,adaptation=fit_selector(scorer,support,
                            collate_graphs(nested["records"][size]["graphs"]).to(device),
                            steps=config.adaptation_steps,seed=config.eval_seed+case_number)
                        kwargs.update(proposal_model=adapted,
                                      proposal_condition=replace(condition,response_state=adapted_state))
                    elif mode=="predictor_search":
                        pass
                    else:
                        raise ValueError(f"unknown evaluation mode {mode}")
                    metadata={"campaign":dataset.campaign_ids[campaign],"lead_id":dataset.ids[lead_index],
                        "lead_index":lead_index,"case":case_number,"support_size":size,"mode":mode,
                        "source_seed":source_config.seed,"source_variant":source_config.redesign.variant,
                        "eval_split":config.eval_split,
                        "support_ids":list(nested["reserve_ids"][:size]),
                        "lead_support_inference_seconds":inference_seconds,"adaptation":adaptation,
                        "case_shared_lead_encoding_seconds":lead_encoding_seconds,
                        "generator_checkpoint_sha256":protocol["generator_checkpoint_sha256"],
                        "common_scorer_checkpoint_sha256":protocol["common_scorer_checkpoint_sha256"]}
                    if not (destination/"proposal_commit.json").exists():
                        if mode=="predictor_search":
                            _search_function()(active_scorer,active_condition,plan,
                                nested["lead"]["smiles"],nested["reserve_smiles"],destination,
                                num_attempts=config.num_attempts,k=config.k,
                                batch_size=config.sample_batch_size,metadata=metadata)
                        else:
                            generate_redesign_proposals(active_scorer,active_condition,plan,
                                nested["lead"]["smiles"],nested["reserve_smiles"],destination,
                                neighbourhood_fn=neighbourhood,batch_size=config.sample_batch_size,
                                nfe=config.nfe,T=config.T,k=config.k,metadata=metadata,**kwargs)
                    pending.append((destination,campaign,baseline.detach().cpu().numpy()[0],
                                    goal.detach().cpu().numpy()[0],size,metadata))
                    atomic_json(output/"status.json",{"phase":"blind_proposal_generation",
                         "case":case_number,"support_size":size,"mode":mode,
                         "committed_batches":len(pending)})
        if tensor_fingerprint(scorer.state_dict().items())!=model_before:
            raise AssertionError("evaluation changed pretrained scorer weights")
    # Hidden outcomes become reachable only after the generation/ranking loop.
    oracle_path=project_path(config.data_dir)/"private/oracle.json"
    if digest(oracle_path)!=protocol["private_oracle_sha256"]:
        raise ValueError("private oracle differs from committed evaluation protocol")
    oracle=ToyOracle.load(oracle_path)
    rows=[]
    for destination,campaign,baseline,goal,size,metadata in pending:
        summary=evaluate_committed_proposals(destination,
                   lambda smiles,c=campaign:oracle.score_smiles(dataset.campaign_ids[c],smiles),
                   baseline=baseline,goal=goal,support_size=size,k=config.k)
        row={**metadata,**{key:value for key,value in summary.items() if key!="metadata"}}
        rows.append(row)
        _write_csv(eval_root/"metrics.csv",rows)
        print(json.dumps({"event":"external_evaluation","case":metadata["case"],
              "N":size,"mode":metadata["mode"],"success":summary["success"],
              "pre_ranking_yield":summary["pre_ranking_unmeasured_joint_profile_yield_per_attempt"]}),flush=True)
    rows=[]
    for path in sorted(eval_root.glob("case_*/support_*/*/evaluation_summary.json")):
        summary=json.loads(path.read_text())
        rows.append({**summary.get("metadata",{}).get("metadata",{}),
                     **{key:value for key,value in summary.items() if key!="metadata"}})
    _write_csv(eval_root/"metrics.csv",rows)
    aggregate=[]
    for mode in config.evaluation_modes:
        for size in config.support_sizes:
            group=[row for row in rows if row["mode"]==mode and row["support_size"]==size]
            if group:
                aggregate.append({"mode":mode,"support_size":size,"cases":len(group),
                    **{key:float(np.mean([row[key] for row in group])) for key in
                       ("success","efficiency","pre_ranking_unmeasured_joint_profile_yield_per_attempt",
                        "pre_ranking_neighbourhood_joint_profile_yield_per_attempt")},
                    "full_pool_oracle_calls":sum(row["external_outcome_reader_calls"] for row in group)})
    _write_csv(eval_root/"support_curves.csv",aggregate)
    report={"phase":"evaluation_complete","split":config.eval_split,"batches":len(rows),
            "source_weights_unchanged":True,"curves":aggregate}
    atomic_json(eval_root/"summary.json",report)
    atomic_json(output/"status.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--resume")
    parser.add_argument("--evaluate-checkpoint")
    args=parser.parse_args()
    config=load_redesign_config(args.config)
    if args.evaluate_checkpoint:
        run_evaluation(config,args.evaluate_checkpoint)
    else:
        run_training(config,resume=args.resume)


if __name__=="__main__":
    main()
