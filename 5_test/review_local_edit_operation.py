"""Allocated-GPU smoke review of every conditional overnight continuation route.

Invoked by the coordinator inside the existing operation allocation; this script
never submits work. The tiny fits exercise interfaces, not scientific efficacy.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from emf.contracts import Latents
from emf.data import collate_graphs
from emf.local_edit_calibration import train_bridge
from emf.local_edit_control import train_surrogate
from emf.local_edit_run import architecture, evaluate, load_checkpoint
from emf.models import EMFModel
from emf.redesign_evaluation import load_committed_proposals
from emf.redesign_sampling import frozen_deployment
from emf.redesign_train import (atomic_json, load_redesign_config, project_path,
                                representation_fingerprint, tensor_fingerprint)
from emf.steering_data import SteeringDataset


def audit_commits(directory, attempts):
    summary=json.loads((directory/'summary.json').read_text())
    assert summary['phase']=='evaluation_complete'
    assert summary['source_weights_unchanged'] is True
    commits=list(directory.glob('pair_*/campaign_*/support_*/*/proposal_commit.json'))
    assert len(commits)==summary['batches']
    invalid=total=0
    for path in commits:
        ledger,_,commit=load_committed_proposals(path.parent)
        measured=json.loads((path.parent/'evaluation_summary.json').read_text())
        assert commit['oracle_access_before_commit'] is False
        assert measured['oracle_access_after_proposal_commit'] is True
        assert len(ledger)==attempts==commit['attempts']
        assert [row['attempt_id'] for row in ledger]==list(range(attempts))
        assert measured['attempts_sha256']==commit['attempts_sha256']
        assert measured['ranking_sha256']==commit['ranking_sha256']
        # Every attempted slot remains, including invalid/disconnected outputs.
        invalid+=sum(not (row['chemically_valid'] and row['connected']) for row in ledger)
        total+=len(ledger)
    return {'batches':len(commits),'logical_attempts':total,'invalid_slots_retained':invalid,
            'commit_hashes_verified':True,'oracle_after_commit':True,
            'source_weights_unchanged':True,'aggregate':summary['aggregate']}


def normalized_backward(config, source, device):
    normalized_config=replace(config,redesign=replace(config.redesign,variant='factorized',rank=8))
    payload=torch.load(project_path(config.initializer),map_location='cpu',weights_only=False)
    base=EMFModel(config.redesign.model)
    base.load_state_dict(payload['model'])
    model=architecture(normalized_config,{'kind':'normalized_known'},base).to(device)
    model.train()
    before=representation_fingerprint(model)
    lead=collate_graphs([source.graph(0)]).to(device)
    state=torch.zeros(1,config.redesign.state_dim,device=device);state[:,0]=1.
    condition=model.condition(lead,torch.zeros(1,2,device=device),
        torch.as_tensor(source.context,device=device,dtype=torch.float32)[None],
        state,torch.tensor([[.25,-.1]],device=device))
    z=Latents(torch.randn(1,model.config.max_nodes,model.config.node_dim,device=device),
        torch.randn(1,model.config.max_nodes*(model.config.max_nodes-1)//2,
                    model.config.pair_dim,device=device)).masked(lead.masks)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-5)
    clean=model.predict_clean(z,torch.tensor([.5],device=device),lead.masks,condition)
    loss=clean.nodes.square().mean()+clean.pairs.square().mean()
    loss.backward()
    gradients=[p.grad for p in model.flow.selector.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().max()>0 for g in gradients)
    assert all(p.grad is None and not p.requires_grad for p in model.encoder.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.normalizer.parameters())
    optimizer.step()
    assert representation_fingerprint(model)==before
    result={'finite_loss':float(loss.detach()),'finite_nonzero_selector_gradient':True,
            'encoder_and_normalizer_unchanged_after_optimizer':True}
    del model,optimizer
    return result


def run(manifest_path):
    started=time.perf_counter()
    manifest=json.loads(Path(manifest_path).read_text())
    task=manifest['stages']['operation']['tasks'][0]
    config=load_redesign_config(task['config'])
    device=torch.device(config.device)
    if device.type!='cuda' or not torch.cuda.is_available():
        raise RuntimeError('supplemental operation review requires its allocated CUDA device')
    torch.set_num_threads(config.cpu_threads)
    output=project_path(config.output_dir)
    checkpoint=output/'checkpoints/source_final.pt'
    source=SteeringDataset(config.data_dir,'train')
    coefficients=json.loads((source.data_dir/'private/oracle.json').read_text())['coefficients']
    model,_,_=load_checkpoint(checkpoint,device)
    model.requires_grad_(False).eval()
    before=tensor_fingerprint(model.state_dict().items())
    bridge_path=output/'operation_evidence_bridge.pt'
    _,bridge_summary=train_bridge(model,source,coefficients,str(device),steps=10,
        checkpoint_path=bridge_path,seed=config.seed,episodes_per_size=4,validation_per_size=2)
    assert before==tensor_fingerprint(model.state_dict().items())
    surrogate_path=output/'operation_control_surrogate.pt'
    with frozen_deployment(model),torch.enable_grad():
        _,surrogate_summary=train_surrogate(model,source,coefficients,device,steps=10,
            batch_size=32,checkpoint_path=surrogate_path,seed=config.seed)
    assert before==tensor_fingerprint(model.state_dict().items())
    del model
    routes={
        'known_control':{**task,'checkpoint':str(checkpoint),'control_radius':.05,
                         'surrogate_checkpoint':str(surrogate_path),'context_mode':'known'},
        'calibrated_control':{**task,'checkpoint':str(checkpoint),'control_radius':.05,
                         'surrogate_checkpoint':str(surrogate_path),'context_mode':'calibrated_control'},
        'bridge':{**task,'checkpoint':str(checkpoint),'bridge_checkpoint':str(bridge_path),
                  'context_mode':'bridge'},
    }
    evaluated={}
    for name,probe_task in routes.items():
        suffix='operation_probe_'+name
        evaluate(probe_task,manifest,'validation',output_suffix=suffix,attempts=2)
        evaluated[name]=audit_commits(output/suffix,2)
        if 'control' in name:
            costs=json.loads((output/suffix/'control_cost.json').read_text())
            assert costs['flow_calls']>0 and costs['candidate_corrections']>0
            assert costs['surrogate_forward_calls']==2*costs['flow_calls']
            assert costs['surrogate_backward_calls']==2*costs['flow_calls']
            evaluated[name]['control_cost']=costs
    evaluated['original_operation']=audit_commits(output/'evaluation_validation',16)
    report={'status':'passed','reviewed_utc':datetime.now(timezone.utc).isoformat(),
        'device':str(device),'cuda_name':torch.cuda.get_device_name(),
        'manifest':str(Path(manifest_path).resolve()),'operation_checkpoint':str(checkpoint),
        'bridge':bridge_summary,'surrogate':surrogate_summary,
        'frozen_generator_sha256':before,'evaluations':evaluated,
        'normalized_backward':normalized_backward(config,source,device),
        'elapsed_seconds':time.perf_counter()-started,
        'scope':'Interface and frozen-weight smoke checks; tiny fits do not establish efficacy.'}
    report_path=Path(manifest['workspace'])/'5_test/results/local_edit_gpu_operation.json'
    atomic_json(report_path,report)
    print(json.dumps({'status':'passed','report':str(report_path),
                      'elapsed_seconds':report['elapsed_seconds']}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    run(parser.parse_args().manifest)
