"""Bounded, validation-gated overnight local molecular edit experiments."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import yaml

ROOT=Path(os.environ.get("EMF_WORKSPACE_ROOT",Path(__file__).resolve().parents[2])).resolve()
STUDY=ROOT/"4_experimental_result/result/local_edit_001"
PYTHON="/hopper/home/miyao/miniconda3/envs/emf/bin/python"

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");temporary.replace(path)

def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare():
    sys.path.insert(0,str(ROOT/"3_model_training"))
    from emf.redesign_train import RedesignRunConfig
    from emf.redesign_contracts import RedesignConfig
    if (STUDY/"study_manifest.json").exists():
        raise FileExistsError("study already prepared")
    data=ROOT/"2_data_storage/processed/evidence_steering_toy"
    alignment=ROOT/"2_data_storage/processed/local_edit_steering/alignment.json"
    config_dir=ROOT/"3_model_training/configs/local_edit_20261013"
    config_dir.mkdir(parents=True,exist_ok=True)
    (STUDY/"logs").mkdir(parents=True,exist_ok=True)
    def task(kind,seed,operation=False):
        name=f"{kind}_seed_{seed}"
        output=STUDY/("operation" if operation else "source")/name
        config=asdict(RedesignRunConfig(
            redesign=RedesignConfig(variant="factorized" if kind=="normalized_known" else "conditional",
                                   rank=8 if kind=="normalized_known" else 32),
            data_dir=str(data),output_dir=str(output),seed=seed,
            train_steps=80 if operation else 20000,train_seconds=120 if operation else 1560,
            cpu_threads=1,support_sizes=[4,16,64],num_attempts=256,nfe=32,k=16,
            checkpoint_every=2000,log_every=100,eval_seed=841203))
        path=config_dir/(("operation_" if operation else "")+name+".yaml")
        path.write_text(yaml.safe_dump(config,sort_keys=False))
        return {"name":name,"kind":kind,"seed":seed,"config":str(path),
                "noise_scale":1.,"context_mode":"known","operation":operation}
    source=[task(kind,seed) for seed in (142,143) for kind in
            ("anchored_known","pooled_known","normalized_known","anchored_prior")]
    stages={
       "operation":{"wall_minutes":10,"concurrency":1,"tasks":[task("anchored_known",142,True)]},
       "source":{"wall_minutes":30,"concurrency":4,"tasks":source},
       "source_completion":{"wall_minutes":3,"concurrency":4,"maximum_tasks":4},
       "control":{"wall_minutes":15,"concurrency":2,"tasks":[t for t in source if t["kind"]=="anchored_prior"]},
       "continuation":{"wall_minutes":20,"concurrency":4,"maximum_tasks":4},
       "calibration":{"wall_minutes":15,"concurrency":2,"maximum_tasks":2},
       "final":{"wall_minutes":4,"concurrency":4,"maximum_tasks":12},
       "control_final":{"wall_minutes":10,"concurrency":2,"maximum_tasks":2},
    }
    snapshot=STUDY/"source_snapshot"
    shutil.copytree(ROOT/"3_model_training/emf",snapshot/"3_model_training/emf",
                    ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    (snapshot/"scripts").mkdir(parents=True)
    for name in ("local_edit_gpu.slurm","local_edit_workflow.py","local_edit_report.py"):
        if (ROOT/"3_model_training/scripts"/name).exists():
            shutil.copy2(ROOT/"3_model_training/scripts"/name,snapshot/"scripts"/name)
    (snapshot/"review").mkdir()
    for path in (ROOT/"5_test").glob("*local_edit*.py"):
        shutil.copy2(path,snapshot/"review"/path.name)
    scorers={str(s):str(ROOT/f"4_experimental_result/result/evidence_steering_001/source/r8_seed_{s}/checkpoints/source_final.pt") for s in (142,143)}
    inputs=[data/"manifest.json",data/"private/oracle.json",alignment,
            ROOT/"4_experimental_result/result/exploration_001/checkpoints/flow_0004000.pt",
            *map(Path,scorers.values()),*data.glob("*.npz")]
    pinned=[p for p in snapshot.rglob("*") if p.is_file()]+list(config_dir.glob("*.yaml"))
    manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"study_root":str(STUDY),
        "snapshot":str(snapshot),"workspace":str(ROOT),"alignment_path":str(alignment),
        "stages":stages,"scorers":scorers,
        "maximum_gpu_hours":8.,"previous_study_gpu_hours":2.7733333333333334,
        "combined_ceiling_gpu_hours":10.773333333333333,
        "maximum_concurrent_gpus":4,
        "planned_cap_gpu_hours":sum(s["wall_minutes"]*len(s.get("tasks",[None]*s.get("maximum_tasks",0)))/60 for s in stages.values()),
        "snapshot_sha256":{str(p):sha(p) for p in pinned},
        "input_sha256":{str(p):sha(p) for p in inputs},
        "gates":{"minimum_local_distinct_yield_per_family_seed":.01,
                 "minimum_valid_slot_fraction_per_family_seed":.25,
                 "positive_local_benefit_each_seed_and_pooled_family":True,
                 "positive_signed_u_reversal_each_family":True,
                 "selection_split":"validation_only"},
        "continuation_rule":"If known-context gates pass, freeze winning generator and fit source-only evidence-to-scalar bridge (or support-only surrogate calibration for controlled prior). Otherwise continue each anchored seed with matched 10000-step runs at categorical weights 0 and .1 on feasibility-weighted predicted clean states; calibrate a repair winner only if validation gates pass. Preserve failed arms.",
        "revision_policy":"Revisability is deferred until this study establishes frozen-generator few-shot local steering.",
        "main_counts":"Distinct local successes per all attempted slots; preserve original Morgan >= .4 rule; exact graph edits secondary.",
        "source_context":"Known scalar is an explicitly privileged simulation diagnostic.",
        "team":{"design_review":"gpt-6-astra","model":"gpt-6-sol","data_control":"gpt-6-sol","reporting":"gpt-6-luna"}}
    if manifest["planned_cap_gpu_hours"]>8:raise RuntimeError("requested caps exceed budget")
    atomic(STUDY/"study_manifest.json",manifest)
    print(json.dumps({"manifest":str(STUDY/"study_manifest.json"),
                      "planned_cap_gpu_hours":manifest["planned_cap_gpu_hours"]}))

def completed_task(task):
    cfg=yaml.safe_load(Path(task["config"]).read_text())
    summary=Path(cfg["output_dir"])/"source_summary.json"
    return summary.exists() and json.loads(summary.read_text()).get("step")==cfg["train_steps"]

def prepare_source_completion(manifest):
    """Complete time-capped source runs without selecting on validation outcomes."""
    decision_path=STUDY/"source_completion_decision.json"
    if decision_path.exists():
        return json.loads(decision_path.read_text())
    tasks=[];incomplete=[]
    for original in manifest["stages"]["source"]["tasks"]:
        cfg=yaml.safe_load(Path(original["config"]).read_text())
        output=Path(cfg["output_dir"])
        summary=output/"source_summary.json";checkpoint=output/"checkpoints/source_final.pt"
        if not summary.exists() or not checkpoint.exists():continue
        status=json.loads(summary.read_text())
        if not 0<int(status.get("step",0))<cfg["train_steps"]:continue
        incomplete.append({"name":original["name"],"step":status["step"],
                           "target":cfg["train_steps"]})
        if len(tasks)>=manifest["stages"]["source_completion"]["maximum_tasks"]:continue
        cfg["output_dir"]=str(STUDY/"source_completion"/original["name"])
        cfg["train_seconds"]=float(status["training_seconds"])+135.
        config_path=ROOT/"3_model_training/configs/local_edit_20261013"/(
            "completion_"+original["name"]+".yaml")
        config_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        tasks.append({**original,"config":str(config_path),
            "resume_checkpoint":str(checkpoint),"resume_checkpoint_sha256":sha(checkpoint),
            "original_source_output":str(output)})
    decision={"created_utc":datetime.now(timezone.utc).isoformat(),
        "rule":"Source update count only; no validation outcome inspected. Preserve full optimizer, RNG, model and cumulative training time. Change output directory and deadline only.",
        "tasks":tasks,"incomplete_originals":incomplete,
        "resolved_config_sha256":{t["config"]:sha(t["config"]) for t in tasks}}
    atomic(decision_path,decision)
    return decision

def resolve_source_completion(manifest,decision):
    replacements={t["name"]:t for t in decision["tasks"] if completed_task(t)}
    decision["effective_source"]=[replacements.get(t["name"],t)
                                  for t in manifest["stages"]["source"]["tasks"]]
    decision["completed_replacements"]=list(replacements)
    atomic(STUDY/"source_completion_decision.json",decision)
    return decision

def effective_source(manifest):
    path=STUDY/"source_completion_decision.json"
    if path.exists():
        return json.loads(path.read_text()).get("effective_source",manifest["stages"]["source"]["tasks"])
    return manifest["stages"]["source"]["tasks"]

def metric_rows(task,suffix="evaluation_validation"):
    cfg=yaml.safe_load(Path(task["config"]).read_text())
    path=Path(cfg["output_dir"])/suffix/"metrics.csv"
    summary=path.parent/"summary.json"
    if not path.exists() or not summary.exists():return []
    if json.loads(summary.read_text()).get("phase")!="evaluation_complete":return []
    rows=list(csv.DictReader(path.open()))
    # Eligibility is computed before oracle access and includes novelty/reserve checks.
    for row in rows:
        campaign_name=row["campaign"]
        pair=int(row["pair_number"]);n=int(row["support_size"])
        # campaign storage index is not in metrics; locate via protocol cases.
        protocol=json.loads((path.parent/"protocol.json").read_text())
        c=protocol["campaigns"].index(campaign_name)
        summary=path.parent/f"pair_{pair:03d}"/f"campaign_{c:03d}"/f"support_{n:03d}"/row["condition"]/"generation_summary.json"
        generated=json.loads(summary.read_text())
        row["eligible_distinct_attempts"]=generated.get("eligible_distinct",0)
    return rows

def assess(tasks,suffix):
    by_seed={str(t["seed"]):metric_rows(t,suffix) for t in tasks}
    if len(by_seed)!=2 or any(not rows for rows in by_seed.values()):
        return {"passed":False,"reason":"two completed seed evaluations required"}
    outcomes=[];score=0.;passed=True
    for seed,rows in by_seed.items():
        hit={c:sum(int(r["local_profile_success_attempts"]) for r in rows if r["condition"]==c) for c in ("correct","opposite")}
        passed &= hit["correct"]>hit["opposite"]
        score+=hit["correct"]
        for family in sorted({r["family"] for r in rows}):
            g=[r for r in rows if r["family"]==family and r["condition"]=="correct"]
            total=sum(int(r["attempts"]) for r in g)
            local=sum(int(r["eligible_distinct_attempts"]) for r in g)/total
            valid=sum(int(r["connected_valid_attempts"]) for r in g)/total
            passed &= local>=.01 and valid>=.25
            outcomes.append({"seed":seed,"family":family,"local_eligible_yield":local,"valid":valid})
    # The campaign suffix a/b declares the positive/negative coefficient pair.
    for family in sorted({r["family"] for rows in by_seed.values() for r in rows}):
        correct=[r for rows in by_seed.values() for r in rows if r["family"]==family and r["condition"]=="correct"]
        opposite=[r for rows in by_seed.values() for r in rows if r["family"]==family and r["condition"]=="opposite"]
        passed &= sum(int(r["local_profile_success_attempts"]) for r in correct)>sum(int(r["local_profile_success_attempts"]) for r in opposite)
        positive=[float(r["mean_generated_u_change"]) for r in correct if r["campaign"].endswith("_a") and r["mean_generated_u_change"]]
        negative=[float(r["mean_generated_u_change"]) for r in correct if r["campaign"].endswith("_b") and r["mean_generated_u_change"]]
        reversal=bool(positive and negative and sum(positive)/len(positive)>sum(negative)/len(negative))
        passed &= reversal
        outcomes.append({"family":family,"signed_u_reversal":reversal})
    return {"passed":bool(passed),"correct_local_successes":score,"checks":outcomes}

def decide(manifest):
    source=effective_source(manifest)
    candidates=[]
    for kind in ("anchored_known","normalized_known","pooled_known"):
        group=[t for t in source if t["kind"]==kind and completed_task(t)]
        result=assess(group,"evaluation_validation")
        candidates.append({"name":kind,"tasks":group,"assessment":result})
    priors=[t for t in source if t["kind"]=="anchored_prior" and completed_task(t)]
    for radius in (.05,.15):
        candidates.append({"name":f"guided_{radius:.2f}","tasks":priors,"radius":radius,
            "assessment":assess(priors,f"control_{radius:.2f}_validation")})
    successful=[c for c in candidates if c["assessment"]["passed"]]
    # Fixed tie preference follows listed order, then higher validation hit count.
    winner=max(successful,key=lambda c:c["assessment"]["correct_local_successes"]) if successful else None
    tasks=[];configdir=ROOT/"3_model_training/configs/local_edit_20261013"
    for seed in (142,143):
        parent=next((t for t in (winner["tasks"] if winner else source)
                     if t["seed"]==seed and (winner or t["kind"]=="anchored_known")),None)
        if parent is None or not completed_task(parent):continue
        cfg=yaml.safe_load(Path(parent["config"]).read_text())
        parent_output=Path(cfg["output_dir"])
        phase="calibration" if winner else "decoded_supervision"
        name=f"{phase}_seed_{seed}"
        cfg["output_dir"]=str(STUDY/"continuation"/name)
        cfg["train_steps"]=10000
        cfg["train_seconds"]=960
        path=configdir/(name+".yaml")
        path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        task={**parent,"name":name,"config":str(path),
              "parent_checkpoint":str(parent_output/"checkpoints/source_final.pt"),
              "continuation_kind":phase,"context_mode":"known"}
        # A scientific continuation starts a new matched optimizer schedule;
        # completing an interrupted source schedule is a different operation.
        task.pop("resume_checkpoint",None)
        task.pop("resume_checkpoint_sha256",None)
        if winner:
            task["checkpoint"]=task["parent_checkpoint"]
            if "radius" in winner:
                task["context_mode"]="calibrated_control"
                task["control_radius"]=winner["radius"]
                task["surrogate_checkpoint"]=str(parent_output/"control_surrogate.pt")
            else:
                task["context_mode"]="bridge"
                task["bridge_checkpoint"]=str(Path(cfg["output_dir"])/"evidence_bridge.pt")
        if winner:
            tasks.append(task)
        else:
            for weight in (0.,.1):
                repair={**task,"terminal_ce_weight":weight,
                        "name":f"decoded_ce_{weight:.1f}_seed_{seed}"}
                repair_cfg=dict(cfg)
                repair_cfg["output_dir"]=str(STUDY/"continuation"/repair["name"])
                repair_path=configdir/(repair["name"]+".yaml")
                repair_path.write_text(yaml.safe_dump(repair_cfg,sort_keys=False))
                repair["config"]=str(repair_path)
                tasks.append(repair)
    decision={"created_utc":datetime.now(timezone.utc).isoformat(),
       "selected_from":"validation only; test metrics not opened for decision",
       "candidates":candidates,"winner":winner["name"] if winner else None,
       "reason":"known-context gates passed" if winner else "no candidate passed all locality/steering gates",
       "continuation":tasks,"final":[t for t in source if completed_task(t)]+tasks,
       "control_final":priors,"calibration":[],
       "resolved_config_sha256":{t["config"]:sha(t["config"]) for t in tasks}}
    atomic(STUDY/"continuation_decision.json",decision)
    return decision

def recovery_decision(manifest,decision):
    """Calibrate a frozen matched-repair winner if it passes validation."""
    if decision["winner"] is not None:
        return decision
    candidates=[]
    for weight in (0.,.1):
        group=[t for t in decision["continuation"] if t.get("terminal_ce_weight")==weight
               and completed_task(t)]
        candidates.append({"weight":weight,"tasks":group,
                           "assessment":assess(group,"evaluation_validation")})
    passing=[c for c in candidates if c["assessment"]["passed"]]
    decision["recovery_candidates"]=candidates
    decision["recovery_winner"]=None
    if passing:
        winner=max(passing,key=lambda c:c["assessment"]["correct_local_successes"])
        decision["recovery_winner"]=winner["weight"]
        for parent in winner["tasks"]:
            cfg=yaml.safe_load(Path(parent["config"]).read_text())
            ckpt=str(Path(cfg["output_dir"])/"checkpoints/source_final.pt")
            name=f"recovery_calibration_seed_{parent['seed']}"
            cfg["output_dir"]=str(STUDY/"continuation"/name)
            path=ROOT/"3_model_training/configs/local_edit_20261013"/(name+".yaml")
            path.write_text(yaml.safe_dump(cfg,sort_keys=False))
            task={**parent,"name":name,"config":str(path),"context_mode":"bridge",
                  "continuation_kind":"calibration","checkpoint":ckpt,
                  "parent_checkpoint":ckpt,
                  "bridge_checkpoint":str(Path(cfg["output_dir"])/"evidence_bridge.pt")}
            decision["calibration"].append(task)
            decision["resolved_config_sha256"][str(path)]=sha(path)
    atomic(STUDY/"continuation_decision.json",decision)
    return decision

def run(resume=False):
    manifest_path=STUDY/"study_manifest.json";manifest=json.loads(manifest_path.read_text())
    state_path=STUDY/"workflow_state.json"
    if state_path.exists() and not resume:raise FileExistsError("workflow state exists; inspect before resuming")
    if resume and not state_path.exists():raise FileNotFoundError("no workflow state to resume")
    history=manifest.get("historical_attempts",{})
    state={"phase":"operation","launched":dict(history.get("launched",{})),
           "accounting_by_stage":dict(history.get("accounting_by_stage",{})),"failures":{},
           "historical_attempts":history.get("records",[]),
           "maximum_gpu_hours":8.,"previous_study_gpu_hours":manifest["previous_study_gpu_hours"]}
    if resume:
        state=json.loads(state_path.read_text())
        state["resumed_utc"]=datetime.now(timezone.utc).isoformat()
        state["manifest_revision"]=manifest.get("revision",1)
        print(json.dumps({"event":"resumed","revision":manifest.get("revision",1),
                          "existing_jobs":state["launched"]}),flush=True)
    atomic(state_path,state)
    def submit(stage,count):
        if not count:return None
        spec=manifest["stages"][stage]
        requested=spec["wall_minutes"]*count/60
        caps=sum(x["allocation_cap_gpu_hours"] for x in state["launched"].values())
        if caps+requested>8+1e-9:raise RuntimeError("submission would exceed study GPU cap")
        cmd=["sbatch","--parsable",f"--job-name=emf-local-{stage}",
             f"--time=00:{spec['wall_minutes']:02d}:00",
             f"--output={STUDY}/logs/{stage}_%A_%a.log",
             f"--array=0-{count-1}%{min(spec['concurrency'],4)}",
             str(Path(manifest["snapshot"])/"scripts/local_edit_gpu.slurm"),
             str(manifest_path),manifest["snapshot"],stage]
        job=subprocess.check_output(cmd,cwd=ROOT,text=True).strip().split(";")[0]
        state["launched"][stage]={"job_id":job,"tasks":count,
            "allocation_cap_gpu_hours":requested,"submitted_utc":datetime.now(timezone.utc).isoformat()}
        atomic(state_path,state);print(json.dumps({"event":"submitted","stage":stage,"job":job}),flush=True)
        return job
    def wait(stage,required=False):
        if stage in state["accounting_by_stage"]:
            if required and stage in state["failures"]:
                raise RuntimeError(f"required completed stage {stage} failed")
            return
        job=state["launched"][stage]["job_id"]
        while True:
            result=subprocess.run(["squeue","-h","-j",job,"-o","%i|%j|%T|%M|%R"],capture_output=True,text=True)
            if result.returncode:time.sleep(30);continue
            state.update(phase=stage,queue=result.stdout.splitlines(),updated_utc=datetime.now(timezone.utc).isoformat())
            atomic(state_path,state)
            if not result.stdout.strip():break
            time.sleep(30)
        expected=state["launched"][stage]["tasks"]
        for attempt in range(6):
            output=subprocess.check_output(["sacct","-n","-P","-j",job,
                "--format=JobID,State,ElapsedRaw,AllocTRES,ExitCode"],text=True)
            rows=[line for line in output.splitlines() if line and "." not in line.split("|")[0]]
            if sum("_" in r.split("|")[0] for r in rows)>=expected:break
            time.sleep(5)
        state["accounting_by_stage"][stage]=rows
        tasks=[r for r in rows if "_" in r.split("|")[0]]
        bad=[r for r in tasks if r.split("|")[1]!="COMPLETED" or r.split("|")[-1]!="0:0"]
        if len(tasks)<expected:bad.append("accounting incomplete")
        if bad:
            state["failures"][stage]=bad
            if required:raise RuntimeError(f"required stage {stage} failed: {bad}")
        atomic(state_path,state)
    def ensure(stage,count,required=False):
        if stage not in state["launched"]:
            submit(stage,count)
        if stage in state["launched"]:
            wait(stage,required=required)
    try:
        ensure("operation",1,required=True)
        ensure("source",8)
        completion=prepare_source_completion(manifest)
        ensure("source_completion",len(completion["tasks"]))
        resolve_source_completion(manifest,completion)
        ensure("control",2)
        state["phase"]="validation_decision";atomic(state_path,state)
        decision=decide(manifest)
        if decision["continuation"]:
            ensure("continuation",len(decision["continuation"]))
        decision=recovery_decision(manifest,decision)
        for stage in ("calibration","final","control_final"):
            count=len(decision[stage])
            if count:
                ensure(stage,count)
        state["phase"]="complete" if not state["failures"] else "complete_with_failures"
        state["completed_utc"]=datetime.now(timezone.utc).isoformat()
    except Exception as error:
        state["phase"]="failed";state["error"]=repr(error)
        raise
    finally:
        state["actual_gpu_hours"]=sum(int(r.split("|")[2]) for rs in state["accounting_by_stage"].values()
                                      for r in rs if "_" in r.split("|")[0])/3600
        atomic(state_path,state)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("prepare","run"))
    p.add_argument("--resume",action="store_true",help="adopt already-submitted jobs after an explicit manifest amendment")
    a=p.parse_args();prepare() if a.command=="prepare" else run(resume=a.resume)
