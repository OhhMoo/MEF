"""Reproducible source training and committed evaluation for masked edit flows."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from rdkit import RDLogger
from .contracts import ModelConfig, GraphBatch, GraphMasks, Latents
from .data import collate_graphs
from .models import EMFModel
from .redesign_contracts import RedesignConfig, CampaignBatch, SupportBatch
from .redesign_train import atomic_json, representation_fingerprint, digest
from .masked_edit_data import MaskedEditDataset, _descriptor, _features
from .masked_edit_model import MaskedEditModel, hybrid_latents, decode_masked_edit
from .masked_edit_objectives import masked_edit_loss
from .masked_edit_sampling import sample_masked_edit_latents, make_paired_gaussian
from .masked_edit_evaluation import ExactMaskedEvaluator

ROOT=Path(os.environ.get('EMF_WORKSPACE_ROOT',Path.cwd())).resolve()
STUDY=ROOT/'4_experimental_result/result/masked_control_001'
INITIALIZER=ROOT/'4_experimental_result/result/exploration_001/checkpoints/flow_0004000.pt'

def now():return datetime.now(timezone.utc).isoformat()
def graph_select(g,i):
    return GraphBatch(g.atoms[i],g.bonds[i],GraphMasks(g.masks.nodes[i],g.masks.pairs[i]))
def mask_select(m,i):return GraphMasks(m.nodes[i],m.pairs[i])
def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        w.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
def setup(seed,device):
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','2')))
    torch.manual_seed(seed);np.random.seed(seed);RDLogger.DisableLog('rdApp.*')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Allocated GPU is required')

def new_model(variant,device,conditioned=False):
    payload=torch.load(INITIALIZER,map_location='cpu',weights_only=False)
    cfg=ModelConfig(**payload['config']['model']);base=EMFModel(cfg)
    base.load_state_dict(payload['model']);base.freeze_encoder()
    model=MaskedEditModel(RedesignConfig(model=cfg,variant='conditional'),base,
                          variant=variant,task_conditioned=conditioned).to(device)
    return model

def load_model(path,device):
    p=torch.load(path,map_location='cpu',weights_only=False)
    model=new_model(p['variant'],device,p['conditioned']);model.load_state_dict(p['model'])
    if representation_fingerprint(model)!=p['representation_sha256']:raise ValueError('Frozen representation changed')
    return model,p

def pair_table(dataset,device):
    rows=[]
    for task in dataset.tasks:
        for i in task['compatible_indices'][dataset.split]:
            rows.append({'task_id':task['task_id'],'index':int(i),'lead_index':task['lead_index']})
    leads=collate_graphs([dataset.graph(r['lead_index']) for r in rows]).to(device)
    queries=collate_graphs([dataset.graph(r['index'],aligned=True,task_id=r['task_id']) for r in rows]).to(device)
    edits=GraphMasks(torch.tensor(np.stack([dataset.masks[r['task_id']].nodes for r in rows]),device=device),
                     torch.tensor(np.stack([dataset.masks[r['task_id']].pairs for r in rows]),device=device))
    return rows,leads,queries,edits

def condition_for(model,dataset,mask_id,batch_size=1,W=None):
    device=next(model.parameters()).device;mask=dataset.masks[mask_id]
    lead=collate_graphs([dataset.graph(mask.lead_index)]).to(device)
    edit=GraphMasks(torch.as_tensor(mask.nodes.copy(),device=device)[None],torch.as_tensor(mask.pairs.copy(),device=device)[None])
    state=torch.zeros(1,model.redesign_config.state_dim,device=device)
    if W is not None:state[0,:6]=torch.as_tensor(W,device=device).flatten()
    cond=model.condition(lead,torch.zeros(1,2,device=device),torch.as_tensor(dataset.context,device=device)[None],state,
                         torch.tensor([[.25,-.1]],device=device),edit)
    return cond if batch_size==1 else cond.expand(batch_size)

def graphs_from_logits(logits):
    atoms=logits.atoms.detach().argmax(-1).cpu().numpy();bonds=logits.bonds.detach().argmax(-1).cpu().numpy()
    return [{'atoms':a,'bonds':b,'count':len(a)} for a,b in zip(atoms,bonds)]

def train(task,device):
    out=Path(task['output']);out.mkdir(parents=True,exist_ok=True)
    final=out/'checkpoint.pt'
    if final.exists():
        p=torch.load(final,map_location='cpu',weights_only=False)
        if p['step']==task['steps']:return final
        raise RuntimeError('Existing checkpoint does not complete requested training')
    seed=task['seed'];setup(seed,device)
    model=(load_model(task['parent'],device)[0] if task.get('parent') else new_model(task['variant'],device))
    model.flow.task_conditioned=bool(task.get('conditioned',False));model.train()
    dataset=MaskedEditDataset(split='train',kind='vector')
    rows,leads,queries,edits=pair_table(dataset,device);n=len(rows);batch_size=task.get('batch_size',64)
    states=torch.zeros(n,64,device=device);deltas=torch.zeros(n,2,device=device)
    source_oracle=None
    if model.flow.task_conditioned:
        source_oracle=ExactMaskedEvaluator()
        scales=source_oracle.scales
        feature=torch.tensor(np.stack([_features(_descriptor(dataset.smiles[r['index']]),_descriptor(dataset.smiles[r['lead_index']]),scales) for r in rows]),device=device,dtype=torch.float32)
        by_task={key:torch.tensor(np.stack([source_oracle.coefficients[cid] for cid in dataset.campaign_ids if dataset.manifest['campaigns'][cid]['mask_id']==key]),device=device,dtype=torch.float32) for key in dataset.masks}
    optimizer=torch.optim.AdamW(list(model.flow.parameters())+list(model.decoder.parameters()),lr=task.get('lr',3e-4),weight_decay=.01)
    rep=representation_fingerprint(model);started=time.perf_counter();metrics={}
    rng=np.random.default_rng(seed);torch_rng=torch.Generator(device=device).manual_seed(seed+11)
    # Uniform pair draws within each mask keep the four geometries equally represented.
    group_indices=[torch.tensor([j for j,r in enumerate(rows) if r['task_id']==key],device=device) for key in dataset.masks]
    with (out/'training.csv').open('w',newline='') as stream:
        writer=None
        for step in range(1,task['steps']+1):
            groups=rng.integers(len(group_indices),size=batch_size)
            ix=torch.stack([group_indices[g][int(rng.integers(len(group_indices[g])))] for g in groups])
            lead=graph_select(leads,ix);query=graph_select(queries,ix);edit=mask_select(edits,ix)
            state=states[ix].clone();delta=deltas[ix].clone();goal=torch.tensor([.25,-.1],device=device).expand(batch_size,-1)
            if model.flow.task_conditioned:
                ws=torch.stack([by_task[rows[int(j)]['task_id']][int(rng.integers(len(by_task[rows[int(j)]['task_id']])))] for j in ix.cpu().tolist()])
                state[:,:6]=ws.reshape(batch_size,6);delta=torch.einsum('bij,bj->bi',ws,feature[ix])
            weights=torch.all(delta>=goal,dim=-1).float()
            support=SupportBatch(torch.zeros(1,1,97,device=device),torch.zeros(1,1,2,device=device),torch.zeros(1,4,device=device),torch.ones(1,1,device=device,dtype=torch.bool))
            batch=CampaignBatch(query,lead,support,torch.zeros(batch_size,device=device,dtype=torch.long),torch.zeros(batch_size,2,device=device),torch.as_tensor(dataset.context,device=device).expand(batch_size,-1),goal,delta,weights)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
                loss,parts=masked_edit_loss(model,batch,edit,response_weight=0.,accept_all=not model.flow.task_conditioned,flow_state_override=state)
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite training at {step}')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(list(model.flow.parameters())+list(model.decoder.parameters()),1.)
            optimizer.step()
            if step==1 or step%100==0 or step==task['steps']:
                metrics={k:float(v) for k,v in parts.items()};metrics.update(step=step,seconds=time.perf_counter()-started,grad_norm=float(grad))
                if writer is None:writer=csv.DictWriter(stream,fieldnames=list(metrics));writer.writeheader()
                writer.writerow(metrics);stream.flush();print(json.dumps(metrics),flush=True)
            if step%5000==0 or step==task['steps']:
                payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'variant':task['variant'],'conditioned':model.flow.task_conditioned,'task':task,'representation_sha256':rep,'training_seconds':time.perf_counter()-started,'created_utc':now(),'rng':torch.get_rng_state(),'numpy_rng':rng.bit_generator.state,'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else [],'noise_rng':torch_rng.get_state()}
                target=final if step==task['steps'] else out/'latest.pt';tmp=target.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(target)
    if rep!=representation_fingerprint(model):raise RuntimeError('Encoder/normalizer changed during source training')
    atomic_json(out/'training_summary.json',dict(task=task,step=task['steps'],metrics=metrics,representation_sha256=rep,finished_utc=now(),source_pairs=n,source_molecules=len(dataset.source_graph_indices),initialization_exposure='Older broad reconstruction/flow initializer; held-out only from this study source updates.'))
    return final

def evaluate_representation(checkpoint,out,split,device,attempts=256,nfe=32):
    model,p=load_model(checkpoint,device);model.eval();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    dataset=MaskedEditDataset(split=split,kind='vector');oracle=ExactMaskedEvaluator();all_rows=[];per_mask={};started=time.perf_counter();physical=0
    with torch.no_grad():
        generated={}
        for mi,mask_id in enumerate(dataset.masks):
            condition=condition_for(model,dataset,mask_id,attempts)
            noise=make_paired_gaussian(model.config,attempts,551100+mi,device=device)
            result=sample_masked_edit_latents(model,condition,noise,nfe=nfe)
            generated[mask_id]=graphs_from_logits(decode_masked_edit(model,result.latents,condition));physical+=attempts*nfe
        for ci,cid in enumerate(dataset.campaign_ids):
            meta=dataset.manifest['campaigns'][cid];mid=meta['mask_id'];support=dataset.nested_support(ci,seed=77600+ci)
            for i,g in enumerate(generated[mid]):
                r=oracle.evaluate_graph(g,dataset.masks[mid],cid,support_smiles=support['reserve_smiles'],source_smiles=[dataset.smiles[j] for j in dataset.source_graph_indices],known_library_smiles=dataset.smiles)
                r.update(variant=p['variant'],seed=p['task']['seed'],sample=i,split=split);all_rows.append(r)
        # Decoder ceiling on held-out aligned/hybrid targets, separate from generated states.
        rows,leads,queries,edits=pair_table(dataset,device)
        lead=model.encode(leads);query=model.encode(queries);target=query if p['variant']=='A' else hybrid_latents(lead,query,edits)
        state=torch.zeros(len(rows),64,device=device)
        cond=model.condition(leads,torch.zeros(len(rows),2,device=device),torch.as_tensor(dataset.context,device=device).expand(len(rows),-1),state,torch.tensor([.25,-.1],device=device).expand(len(rows),-1),edits)
        logits=decode_masked_edit(model,target,cond)
        reconstruction=((logits.atoms.argmax(-1)==queries.atoms).all(-1)&(logits.bonds.argmax(-1)==queries.bonds).all(-1)).float().mean().item()
    for mid in dataset.masks:
        rs=[r for r in all_rows if r['mask_id']==mid];first=[r for r in rs if r['campaign_id']==rs[0]['campaign_id']]
        eligible=[r for r in first if r['chemically_valid'] and r['connected'] and r['protected_copy'] and not r['lead_copy']]
        per_mask[mid]={'attempts':len(first),'valid_fraction':sum(r['chemically_valid'] and r['connected'] for r in first)/len(first),'protected_fraction':sum(r['protected_copy'] for r in first)/len(first),'distinct_eligible':len({r['smiles'] for r in eligible}),'distinct_eligible_yield':len({r['smiles'] for r in eligible})/len(first),'mean_graph_edits':float(np.mean([r['total_edits'] for r in eligible])) if eligible else None}
    metrics=[]
    for cid in dataset.campaign_ids:
        rs=[r for r in all_rows if r['campaign_id']==cid];hits={r['smiles'] for r in rs if r['goal_success'] and not r['support_match']};novel={r['smiles'] for r in rs if r['eligible_novel_success']}
        metrics.append({'campaign_id':cid,'variant':p['variant'],'seed':p['task']['seed'],'attempts':len(rs),'distinct_successes_excluding_support':len(hits),'distinct_successes_excluding_source_support':len(novel),'success_attempts':sum(r['goal_success'] and not r['support_match'] for r in rs)})
    write_csv(out/'attempts.csv',all_rows);write_csv(out/'metrics.csv',metrics)
    summary={'phase':'complete','checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),'split':split,'variant':p['variant'],'seed':p['task']['seed'],'per_mask':per_mask,'mean_distinct_yield':float(np.mean([x['distinct_eligible_yield'] for x in per_mask.values()])),'exact_target_reconstruction':reconstruction,'nfe_per_sample':nfe,'physical_sample_nfe':physical,'logical_attempts':len(all_rows),'physical_attempts':len(dataset.masks)*attempts,'seconds_including_chemistry':time.perf_counter()-started,'completed_utc':now()}
    atomic_json(out/'summary.json',summary);print(json.dumps(summary),flush=True);return summary

def execute(task,device):
    stage=task.get('stage','source')
    if stage in ('source','operation','conditional'):
        checkpoint=train(task,device)
        if stage!='conditional':
            evaluate_representation(checkpoint,Path(task['output'])/'validation','validation',device,attempts=16 if stage=='operation' else 256,nfe=8 if stage=='operation' else 32)
    elif stage=='test':evaluate_representation(task['checkpoint'],task['output'],'test',device)
    else:
        from .masked_edit_experiments import execute_experiment
        execute_experiment(task,device)

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--device',default='cuda',choices=['cpu','cuda']);a=p.parse_args()
    task=json.loads(Path(a.task).read_text())
    if isinstance(task,list):task=task[int(os.environ['SLURM_ARRAY_TASK_ID'])]
    setup(task['seed'],a.device);execute(task,a.device)
if __name__=='__main__':main()
