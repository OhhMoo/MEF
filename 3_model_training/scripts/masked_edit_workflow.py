"""Validation-gated, bounded overnight coordinator. Run persistently in tmux."""
from __future__ import annotations
import argparse,csv,fcntl,json,os,subprocess,sys,time,traceback
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import hashlib
ROOT=Path(os.environ.get('EMF_WORKSPACE_ROOT',Path(__file__).resolve().parents[2])).resolve()
S=ROOT/'4_experimental_result/result/masked_control_001'
PYTHON='/hopper/home/miyao/miniconda3/envs/emf/bin/python'

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def atomic(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def command(args):return subprocess.check_output(args,text=True,cwd=ROOT).strip()
def complete(t):
    o=Path(t['output']);stage=t['stage']
    if stage in ('source','operation'):return (o/'training_summary.json').exists() and (o/'validation/summary.json').exists() and read(o/'training_summary.json')['step']==t['steps']
    if stage=='conditional':return (o/'training_summary.json').exists() and read(o/'training_summary.json')['step']==t['steps']
    if stage=='baselines':return (o/'manifest.json').exists()
    return (o/'summary.json').exists()
def task_status(t,state,job=None):
    atomic(Path(t['output'])/'status.json',dict(name=t['name'],status=state,job_id=job,updated_utc=now(),task=t['name']))

def daily(status):
    marker='<!-- masked-control-001 -->';end='<!-- /masked-control-001 -->'
    lines=[marker,'','## Masked edits and terminal control — overnight study','',f"Updated {now()}. Coordinator state: **{status.get('phase','unknown')}**.",'',
           'We compare lead-relative generation (A), hard-masked graph generation (B), and their combination (C), then test known-response control and calibration from 4/8/16 observations. Outcomes are exact simulated two-property responses. The inherited initializer predates the new molecular split.','',
           '[Live report](../../4_experimental_result/result/masked_control_001/live_report.md) · [Experiment plan](../detail_operation/05_masked_edit_control_experiment_plan.md) · [Run records](../../4_experimental_result/result/masked_control_001/)','']
    sel=S/'representation_selection.json'
    if sel.exists():
        d=read(sel);lines += [f"Validation selected representation **{d['variant']}**. Selection uses distinct valid protected nonlead molecules per all attempts, averaged over three seeds.",'','| Representation | Mean distinct local yield |','|---|---:|']
        lines += [f"| {v} | {x:.4f} |" for v,x in d['scores'].items()];lines+=['']
    rows=[]
    for p in sorted((S/'control_test').glob('*/summary.json')):
        d=read(p)
        for c in d.get('contrasts',[]):rows.append((d['seed'],c))
    if rows:
        lines+=['| Seed | Method | Measurements (0 = known coefficients) | Correct evidence: distinct joint successes | Opposite evidence | Correct top-16 successes / slots |','|---|---|---:|---:|---:|---:|']
        for seed,c in rows:lines.append(f"| {seed} | {c['method']} | {c['support_size']} | {c['correct_distinct_successes']} | {c['opposite_distinct_successes']} | {c['correct_topk_successes']} / {c['topk_budget']} |")
        lines += ['','Distinct successes in this table are summed across campaign cases. See each summary for globally unique successful identities. Unused top-16 slots remain in the denominator. These figures must be read with the forward/backward computation and invalid-attempt counts in the CSV files.','']
    if (S/'revision_decision.json').exists():
        d=read(S/'revision_decision.json');lines += [f"Revision decision: {d['reason']}",'']
    if status.get('errors'):lines += ['Recorded failures: '+ '; '.join(status['errors']), '']
    lines += ['The exact-oracle search is a privileged diagnostic with a declared terminal-query budget. The learned controller uses a source-trained terminal surrogate. Finite-library and direct-mutation baselines use the declared descriptor basis and support-only calibration. No result here establishes drug-discovery performance.', '',end]
    path=ROOT/'1_research_file/result_log/experiment_results_1013.md';lock=Path(str(path)+'.lock')
    with lock.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);old=path.read_text() if path.exists() else '# Experimental results — 1013\n'
        block='\n'.join(lines)+'\n'
        if marker in old:
            before,rest=old.split(marker,1);_,after=rest.split(end,1);new=before+block+after.lstrip('\n')
        else:new=old.rstrip()+'\n\n'+block
        tmp=path.with_suffix('.md.tmp');tmp.write_text(new);tmp.replace(path)

def update(state):
    state['updated_utc']=now();atomic(S/'status.json',state)
    report=ROOT/'3_model_training/scripts/masked_edit_report.py'
    if report.exists():
        p=subprocess.run([PYTHON,str(report),'--study',str(S)],cwd=ROOT,capture_output=True,text=True)
        if p.returncode:print('report error',p.stderr[-1000:],flush=True)
    daily(state)

def wait_job(job,tasks,state,label):
    state['phase']=label;state.setdefault('jobs',[])
    if str(job) not in state['jobs']:state['jobs'].append(str(job))
    while True:
        if all(complete(t) for t in tasks):
            active=''
        else:
            try:
                active=command(['squeue','-h','-j',str(job),'-o','%i|%T'])
            except subprocess.CalledProcessError:
                # Completed jobs can disappear from squeue before this poll.
                records=command(['sacct','-n','-X','-P','-j',str(job),'--format=State'])
                states=[line.split('|')[0].split()[0] for line in records.splitlines() if line.strip()]
                terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
                if states and all(x in terminal for x in states):active=''
                else:raise
        for i,t in enumerate(tasks):
            task_status(t,'completed' if complete(t) else ('running' if active else 'failed'),f'{job}_{i}' if len(tasks)>1 else str(job))
        update(state)
        if not active:break
        time.sleep(30)
    # Allow scheduler completion visibility to settle without assuming success.
    failed=[t['name'] for t in tasks if not complete(t)]
    if failed:state.setdefault('errors',[]).append(f'{label}: incomplete {failed}');update(state)
    return [t for t in tasks if complete(t)]

def accounted_hours(jobs):
    if not jobs:return 0.
    raw=command(['sacct','-n','-X','-P','-j',','.join(jobs),'--format=JobIDRaw,ElapsedRaw,AllocTRES'])
    seconds=0
    for line in raw.splitlines():
        parts=line.split('|')
        if len(parts)<3 or '.' in parts[0]:continue
        gpus=next((int(x.split('=')[1]) for x in parts[2].split(',') if x.startswith('gres/gpu=')),0)
        seconds+=int(parts[1] or 0)*gpus
    return seconds/3600

def submit(tasks,label,minutes,state,concurrency=8):
    if not tasks:return []
    pending=[t for t in tasks if not complete(t)]
    if not pending:return tasks
    # Large ceilings permit the study to finish; this check prevents unbounded continuations.
    used=accounted_hours(state.get('jobs',[]));state['allocated_gpu_hours']=used
    reserved=len(pending)*minutes/60
    if used+reserved>48:raise RuntimeError(f'GPU reservation would exceed operating ceiling: used={used:.2f}, requested maximum={reserved:.2f}, cap=48')
    snap=Path(read(S/'followup_ready.json')['snapshot'])
    for t in pending:
        t['snapshot']=str(snap);atomic(S/'tasks'/f"{t['name']}.json",t)
    array=S/'tasks'/f'{label}_array.json';atomic(array,pending)
    args=['sbatch','--parsable',f'--time={minutes//60:02d}:{minutes%60:02d}:00',f'--array=0-{len(pending)-1}%{min(concurrency,len(pending))}',f'--output={S}/logs/{label}_%A_%a.log',str(snap/'scripts/masked_edit_gpu.slurm'),str(array),str(snap)]
    job=command(args).split(';')[0]
    state.setdefault('submissions',[]).append(dict(stage=label,job_id=job,tasks=[t['name'] for t in pending],submitted_utc=now(),wall_minutes=minutes,task_sha256=sha(array)))
    print(json.dumps(state['submissions'][-1]),flush=True)
    done=wait_job(job,pending,state,label)
    failed=[t for t in pending if not complete(t) and not t.get('retry_attempt')]
    if failed:
        retries=[dict(t,name=t['name']+'_retry1',output=t['output']+'_retry1',retry_attempt=1) for t in failed]
        recovered=submit(retries,label+'_retry1',minutes,state,concurrency)
        replacements={r['name'].removesuffix('_retry1'):r for r in recovered}
        tasks=[replacements.get(t['name'],t) for t in tasks]
        atomic(S/(label+'_recovery.json'),dict(original_failed=[t['name'] for t in failed],recovered=recovered))
    return [t for t in tasks if complete(t)]

def build_task(stage,seed,**kw):
    name=kw.pop('name',f'{stage}_seed{seed}');return dict(stage=stage,seed=seed,name=name,output=str(S/stage/name),**kw)

def choose_representation(tasks):
    scores={v:[] for v in 'ABC'}
    for t in tasks:scores[t['variant']].append(read(Path(t['output'])/'validation/summary.json')['mean_distinct_yield'])
    if any(len(x)!=3 for x in scores.values()):raise RuntimeError('All three seeds per representation are required before selection')
    means={v:sum(x)/len(x) for v,x in scores.items()};winner=max(means,key=lambda v:(means[v],v=='C',v=='B'))
    atomic(S/'representation_selection.json',dict(variant=winner,scores=means,seed_scores=scores,selection_split='validation',selected_utc=now()))
    return [t for t in tasks if t['variant']==winner]

def critic_bad(task):
    data=read(Path(task['output'])/'summary.json')
    audit=data.get('diagnostics',{}).get('audit_generated',{})
    rmse=audit.get('rmse_scaled')
    return not audit.get('records',0) or not rmse or any(not __import__('math').isfinite(x) or x>.2 for x in rmse)


def contrast_score(t):
    d=read(Path(t['output'])/'summary.json');cs=d.get('contrasts',[])
    c=next(x for x in cs if x['method']=='terminal_control' and x['support_size']==0)
    return c['correct_topk_successes'],c['correct_distinct_successes']-c['opposite_distinct_successes']

def main(source_job):
    state=read(S/'status.json') if (S/'status.json').exists() else dict(phase='source',started_utc=now(),jobs=[],errors=[])
    manifest=read(S/'study_manifest.json')
    state['jobs']=list(dict.fromkeys(state.get('jobs',[])+[str(x) for x in manifest.get('operations',{}).values()]))
    sources=wait_job(source_job,manifest['source_tasks'],state,'source')
    if len(sources)!=9:
        while not (S/'followup_ready.json').exists():time.sleep(30)
        failed=[t for t in manifest['source_tasks'] if not complete(t)]
        retries=[dict(t,name=t['name']+'_retry1',output=t['output']+'_retry1',retry_attempt=1) for t in failed]
        sources+=submit(retries,'source_retry1',90,state)
        atomic(S/'source_resolution.json',dict(effective_source=sources))
        if len(sources)!=9:raise RuntimeError('Source grid incomplete after one recorded retry; all seeds required')
    winners=choose_representation(sources)
    while not (S/'followup_ready.json').exists():state['phase']='waiting_for_reviewed_followup_snapshot';update(state);time.sleep(30)
    critics=[build_task('critic',t['seed'],checkpoint=str(Path(t['output'])/'checkpoint.pt'),variant=t['variant'],steps=3000,source_attempts=512) for t in winners]
    conditionals=[build_task('conditional',t['seed'],variant=t['variant'],parent=str(Path(t['output'])/'checkpoint.pt'),conditioned=True,steps=10000,batch_size=64) for t in winners]
    # Critic fitting and conditional adaptation are independent and run together.
    prepared=submit(critics+conditionals,'response_preparation',60,state)
    critics=[t for t in prepared if t['stage']=='critic'];conditionals=[t for t in prepared if t['stage']=='conditional']
    if len(critics)!=3 or len(conditionals)!=3:raise RuntimeError('Missing critic/conditional source preparation')
    repairs=[dict(t,name=t['name']+'_refined',output=t['output']+'_refined',steps=5000,source_attempts=1024) for t in critics if critic_bad(t)]
    if repairs:
        atomic(S/'critic_refinement_decision.json',dict(rule='Source-only audit RMSE exceeds .2, or audit missing; one larger fit, original retained',tasks=repairs))
        refined=submit(repairs,'critic_refinement',45,state)
        for r in refined:
            # Selection compares source-only descriptor diagnostics, never campaign test outcomes.
            for i,old in enumerate(critics):
                if r['seed']==old['seed']:
                    old_report=read(Path(old['output'])/'summary.json');new_report=read(Path(r['output'])/'summary.json')
                    old_metric=old_report['diagnostics']['audit_generated'].get('rmse_scaled') or [float('inf')]
                    new_metric=new_report['diagnostics']['audit_generated'].get('rmse_scaled') or [float('inf')]
                    finite_score=lambda values:sum(values) if all(__import__('math').isfinite(x) for x in values) else float('inf')
                    if finite_score(new_metric)<finite_score(old_metric):critics[i]=r
    critic_by_seed={t['seed']:t for t in critics};cond_by_seed={t['seed']:t for t in conditionals}
    candidates=[]
    for t in winners:
        for label,radii in [('small',[.05,.2]),('medium',[.2,.8]),('large',[.5,2.])]:
            candidates.append(build_task('control_validation',t['seed'],name=f'control_val_{label}_seed{t["seed"]}',variant=t['variant'],checkpoint=str(Path(t['output'])/'checkpoint.pt'),critic=str(Path(critic_by_seed[t['seed']]['output'])/'critic.pt'),conditional_checkpoint=str(Path(cond_by_seed[t['seed']]['output'])/'checkpoint.pt'),radii=radii,support_sizes=[0],attempts=64,nfe=16,iterations=4,split='validation',setting=label))
    oracle_tasks=[build_task('oracle',t['seed'],name=f'oracle_validation_seed{t["seed"]}',variant=t['variant'],checkpoint=str(Path(t['output'])/'checkpoint.pt'),split='validation') for t in winners]
    # Run independent exact-search diagnostics beside the surrogate sweep.
    done=submit(candidates+oracle_tasks,'known_response',90,state)
    candidates=[t for t in done if t['stage']=='control_validation']
    if len(candidates)!=9:raise RuntimeError('Known response sweep incomplete')
    grouped=defaultdict(list)
    for t in candidates:grouped[t['setting']].append(contrast_score(t))
    scores={k:(sum(x[0] for x in v),sum(x[1] for x in v)) for k,v in grouped.items()}
    setting=max(scores,key=lambda k:scores[k]);selected=[t for t in candidates if t['setting']==setting]
    atomic(S/'control_selection.json',dict(setting=setting,scores=scores,selected=selected,selection_split='validation',selected_utc=now()))
    calibration=[dict(t,stage='fewshot_validation',name=f'fewshot_val_seed{t["seed"]}',output=str(S/'fewshot_validation'/f'seed{t["seed"]}'),support_sizes=[16],conditional_checkpoint=None) for t in selected]
    cals=submit(calibration,'fewshot_validation',60,state)
    revision_pass=True;reason=[]
    for t in cals:
        cs=read(Path(t['output'])/'summary.json')['contrasts'];c=next(x for x in cs if x['method']=='terminal_control');u=next(x for x in cs if x['method']=='unguided_rerank')
        ok=c['correct_distinct_successes']>c['opposite_distinct_successes'] and c['correct_distinct_successes']>u['correct_distinct_successes']
        revision_pass &= ok;reason.append(dict(seed=t['seed'],passed=ok,control=c,unguided=u))
    revision_pass &= len(cals)==3
    atomic(S/'revision_decision.json',dict(passed=revision_pass,records=reason,reason='Run nested-evidence revision: validation control improves joint successes over opposite evidence and unguided generation in all three seeds.' if revision_pass else 'Revision deferred: decoded validation steering did not beat both opposite evidence and unguided generation in all three seeds. Retain control/calibration failures for analysis.'))
    finals=[dict(t,stage='control_test',name=f'control_test_seed{t["seed"]}',output=str(S/'control_test'/f'seed{t["seed"]}'),split='test',support_sizes=[0,4,8,16],compute_pool_multiplier=25) for t in selected]
    tests=[build_task('test',t['seed'],name=f'prior_test_{t["variant"]}_seed{t["seed"]}',checkpoint=str(Path(t['output'])/'checkpoint.pt'),variant=t['variant']) for t in sources]
    oracles=[dict(t,name=t['name'].replace('validation','test'),output=t['output'].replace('oracle_validation','oracle_test'),split='test') for t in oracle_tasks]
    baselines=[build_task('baselines',142,name=f'simple_baselines_{split}',split=split,budget=256) for split in ('validation','test')]
    revisions=[dict(t,stage='revision',name=f'revision_seed{t["seed"]}',output=str(S/'revision'/f'seed{t["seed"]}'),split='test',attempts=16,chunk=4,allow_other_seed=True) for t in selected] if revision_pass else []
    submit(finals+oracles+baselines,'heldout_control',90,state)
    submit(tests,'heldout_representation',10,state)
    submit(revisions,'heldout_revision',60,state)
    state['allocated_gpu_hours']=accounted_hours(state['jobs']);state['phase']='completed_with_recorded_failures' if state.get('errors') else 'complete';state['completed_utc']=now();update(state)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-job',required=True);args=parser.parse_args()
    try:main(args.source_job)
    except Exception as exc:
        state=read(S/'status.json') if (S/'status.json').exists() else {};state['phase']='failed';state.setdefault('errors',[]).append(repr(exc));state['traceback']=traceback.format_exc();update(state);raise
