"""Isolated coordinator completion/selection checks; never touch live run state."""
import importlib.util
import json
from pathlib import Path

import pytest


def workflow():
    path = Path(__file__).resolve().parents[1]/'3_model_training/scripts/masked_edit_workflow.py'
    spec = importlib.util.spec_from_file_location('masked_workflow_review', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_completion_requires_requested_update_count_and_representation_evaluation(tmp_path):
    module = workflow()
    task = {'stage':'source', 'output':str(tmp_path/'source'), 'steps':20000}
    output = Path(task['output'])
    write(output/'training_summary.json', {'step':19000})
    write(output/'validation/summary.json', {'phase':'complete'})
    assert not module.complete(task)
    write(output/'training_summary.json', {'step':20000})
    assert module.complete(task)
    (output/'validation/summary.json').unlink()
    assert not module.complete(task)
    assert module.complete({**task,'stage':'conditional'})


def test_representation_selection_uses_all_seeds_and_declared_tie_order(tmp_path):
    module = workflow(); module.S = tmp_path
    tasks = []
    for variant, values in {'A':[.8,.0,.1], 'B':[.4,.4,.4], 'C':[.4,.4,.4]}.items():
        for seed, value in zip((142,143,144),values):
            task = {'variant':variant,'seed':seed,'output':str(tmp_path/f'{variant}{seed}')}
            write(Path(task['output'])/'validation/summary.json', {'mean_distinct_yield':value})
            tasks.append(task)
    selected = module.choose_representation(tasks)
    assert [t['variant'] for t in selected] == ['C']*3
    assert json.loads((tmp_path/'representation_selection.json').read_text())['selection_split']=='validation'
    with pytest.raises(RuntimeError, match='All three seeds'):
        module.choose_representation(tasks[:-1])


def test_critic_missing_generated_audit_triggers_refinement(tmp_path):
    module = workflow()
    task = {'output':str(tmp_path)}
    write(tmp_path/'summary.json', {'diagnostics':{
        'hard':{'records':100,'rmse_scaled':[.01,.01]},
        'train_generated':{'records':100,'rmse_scaled':[.01,.01]},
        'audit_generated':{'records':0,'rmse_scaled':None}}})
    assert module.critic_bad(task)


def test_submission_reserves_requested_gpu_time_before_scheduler_call(tmp_path, monkeypatch):
    module = workflow(); module.S = tmp_path
    monkeypatch.setattr(module, 'accounted_hours', lambda jobs:47.)
    monkeypatch.setattr(module, 'command', lambda args:pytest.fail('must not submit beyond cap'))
    task = {'name':'control','stage':'control_validation','output':str(tmp_path/'control')}
    with pytest.raises(RuntimeError, match='reservation'):
        module.submit([task], 'test', 120, {'jobs':['example']})


def test_failed_followup_retries_once_to_fresh_output_and_returns_replacement(tmp_path, monkeypatch):
    module = workflow(); module.S = tmp_path
    write(tmp_path/'followup_ready.json', {'snapshot':str(tmp_path/'snapshot')})
    monkeypatch.setattr(module, 'accounted_hours', lambda jobs:0.)
    submitted=[]
    monkeypatch.setattr(module, 'command', lambda args:submitted.append(args) or str(100+len(submitted)))
    def wait(job, tasks, state, label):
        for task in tasks:
            if task.get('retry_attempt'):
                write(Path(task['output'])/'summary.json', {'phase':'complete'})
        return [task for task in tasks if module.complete(task)]
    monkeypatch.setattr(module, 'wait_job', wait)
    task={'name':'control','stage':'control_validation','output':str(tmp_path/'control')}
    results=module.submit([task], 'test', 10, {'jobs':[]})
    assert len(submitted)==2 and len(results)==1
    assert results[0]['name']=='control_retry1'
    assert results[0]['output']==str(tmp_path/'control_retry1')
    assert not (tmp_path/'control/summary.json').exists()
    assert (tmp_path/'test_recovery.json').exists()


def test_completed_artifacts_allow_resume_after_scheduler_forgets_job(tmp_path, monkeypatch):
    module = workflow();module.S=tmp_path
    task={'name':'finished','stage':'control_validation','output':str(tmp_path/'finished')}
    write(Path(task['output'])/'summary.json',{'phase':'complete'})
    monkeypatch.setattr(module,'command',lambda args:pytest.fail('completed artifacts require no scheduler query'))
    monkeypatch.setattr(module,'update',lambda state:None)
    assert module.wait_job('expired',[task],{'jobs':[]},'test')==[task]


def test_missing_queue_job_uses_accounting_to_retain_failure(tmp_path, monkeypatch):
    import subprocess
    module=workflow();module.S=tmp_path
    task={'name':'failed','stage':'control_validation','output':str(tmp_path/'failed')}
    def command(args):
        if args[0]=='squeue':raise subprocess.CalledProcessError(1,args)
        return 'FAILED|'
    monkeypatch.setattr(module,'command',command)
    monkeypatch.setattr(module,'update',lambda state:None)
    state={'jobs':[]}
    assert module.wait_job('expired',[task],state,'test')==[]
    assert state['errors']
