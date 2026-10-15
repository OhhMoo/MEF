"""Read-only, float32 evidence-route diagnostics for original redesign checkpoints.

The Euler sampler is inherited from the documented ELF graph adaptation. No
training or private oracle information enters generation. Jacobians use the
exact chain rule B @ da/db, retaining only active coordinates.
"""
from dataclasses import replace
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .campaigns import CampaignDataset, neighbourhood
from .chem import decode_graph, smiles_to_graph
from .contracts import GraphMasks, Latents
from .data import collate_graphs
from .redesign_contracts import FlowCondition
from .redesign_train import (load_source_checkpoint, project_path, cache_graph_features,
                             _support, digest, atomic_json)
from .toy_oracle import ToyOracle


def active_flat(value, masks):
    return torch.cat((value.nodes[masks.nodes].flatten(), value.pairs[masks.pairs].flatten()))


def active_basis(parts, masks):
    return torch.cat((parts.node_directions[masks.nodes].flatten(0, 1),
                      parts.pair_directions[masks.pairs].flatten(0, 1)), 0)


def exact_composed_jacobian(model, z, t, masks, condition):
    """Single-state exact float32 Jacobian; no endpoint-time velocity factor."""
    if len(z.nodes) != 1:
        raise ValueError('Jacobian requires one trajectory')
    with torch.enable_grad(), torch.autocast(device_type=z.nodes.device.type, enabled=False):
        state = condition.response_state.detach().float().requires_grad_(True)
        parts = model.flow.decompose(z, t, masks, replace(condition, response_state=state))
        rows = [torch.autograd.grad(parts.strengths[0, i], state, retain_graph=True)[0][0]
                for i in range(model.flow.rank)]
        ja = torch.stack(rows)
        basis = active_basis(parts, masks).detach()
        jacobian = basis @ ja
    singular = torch.linalg.svdvals(jacobian.detach())
    tolerance = max(jacobian.shape) * torch.finfo(torch.float32).eps * singular.max().clamp_min(1e-30)
    return jacobian.detach(), {'singular_values': singular.cpu().tolist(),
        'numerical_rank': int((singular > tolerance).sum()), 'rank_bound': model.flow.rank,
        'rank_tolerance': float(tolerance), 'frobenius_norm': float(jacobian.norm()),
        'active_coordinates': jacobian.shape[0], 'evidence_coordinates': jacobian.shape[1]}


@torch.no_grad()
def reference_trajectory(model, noise, counts, condition, *, nfe=32, T=.95, probes=(.1,.5,.9)):
    """Probe exact times through dense Euler output without changing the grid."""
    device = noise.nodes.device
    masks = GraphMasks.from_counts(counts.to(device), model.config.max_nodes)
    z = noise.masked(masks)
    grid = torch.linspace(0, T, nfe, device=device)
    snapshots = {}
    condition = condition.expand(len(counts))
    for step in range(nfe - 1):
        t = grid[step]
        clean = model.predict_clean(z, t.expand(len(counts)), masks, condition)
        velocity = Latents((clean.nodes-z.nodes)/(1-t), (clean.pairs-z.pairs)/(1-t))
        for probe in probes:
            if float(t) <= probe <= float(grid[step+1]) and probe not in snapshots:
                snapshots[probe] = Latents(z.nodes+(probe-t)*velocity.nodes,
                                           z.pairs+(probe-t)*velocity.pairs).masked(masks)
        dt = grid[step+1]-t
        z = Latents(z.nodes+dt*(clean.nodes-z.nodes)/(1-t),
                    z.pairs+dt*(clean.pairs-z.pairs)/(1-t)).masked(masks)
    final = model.predict_clean(z, grid[-1].expand(len(counts)), masks, condition).masked(masks)
    return final, masks, snapshots


@torch.no_grad()
def stage_comparison(model, z, t, masks, reference, alternate):
    a = model.flow.decompose(z, t, masks, reference)
    b = model.flow.decompose(z, t, masks, alternate)
    correction = Latents(torch.einsum('bndr,br->bnd', a.node_directions, b.strengths-a.strengths),
                         torch.einsum('bpdr,br->bpd', a.pair_directions, b.strengths-a.strengths))
    delta = active_flat(correction, masks)
    logits_a, logits_b = model.decode(a.compose(), masks), model.decode(b.compose(), masks)
    la = torch.cat((logits_a.atoms[masks.nodes].flatten(),logits_a.bonds[masks.pairs].flatten()))
    lb = torch.cat((logits_b.atoms[masks.nodes].flatten(),logits_b.bonds[masks.pairs].flatten()))
    margins = []
    for value, mask in ((logits_a.atoms,masks.nodes),(logits_a.bonds,masks.pairs)):
        top = value[mask].topk(2,dim=-1).values
        margins.extend((top[:,0]-top[:,1]).cpu().tolist())
    return {'evidence_l2': float((alternate.response_state-reference.response_state).norm()),
        'correction_l2':float(delta.norm()),'correction_rms':float(delta.square().mean().sqrt()),
        'velocity_delta_l2':float(delta.norm()/(1-t[0])), 'logit_delta_l2':float((lb-la).norm()),
        'reference_margin_mean':float(np.mean(margins)), 'reference_margin_min':float(min(margins)),
        'changed_atom_categories':int(((logits_a.atoms.argmax(-1)!=logits_b.atoms.argmax(-1))&masks.nodes).sum()),
        'changed_bond_categories':int(((logits_a.bonds.argmax(-1)!=logits_b.bonds.argmax(-1))&masks.pairs).sum()),
        'base_delta_l2':float((active_flat(a.base,masks)-active_flat(b.base,masks)).norm()),
        'basis_delta_l2':float((active_basis(a,masks)-active_basis(b,masks)).norm())}


def run(args):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    output=Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists():
        raise FileExistsError('Trace destination already has a manifest; use a distinct directory')
    started=time.perf_counter()
    manifest={'checkpoints':[], 'arguments':vars(args), 'precision':'float32; TF32 disabled',
        'probe_rule':'dense output along N16 reference Euler trajectory',
        'jacobian_rule':'exact B @ da/db over active coordinates', 'completed_cases':0}
    atomic_json(output/'manifest.json',manifest)
    global_case=0
    for checkpoint in args.checkpoints:
        model,config,payload=load_source_checkpoint(checkpoint,torch.device(args.device))
        model.float().freeze_deployment()
        if not hasattr(model.flow,'decompose'):
            raise ValueError('Trace requires original factorized checkpoints')
        dataset=CampaignDataset(project_path(config.data_dir),args.split)
        features,_=cache_graph_features(model,dataset.graph_dataset,bf16=False)
        protocol_path=project_path(config.output_dir)/f'evaluation_{args.split}'/'protocol.json'
        if protocol_path.exists():
            protocol=json.loads(protocol_path.read_text())
            cases=[(dataset.campaign_ids.index(row['campaign']),row['lead_index']) for row in protocol['cases']]
        else:
            rng=np.random.default_rng(config.eval_seed+(0 if args.split=='validation' else 100000))
            cases=[(campaign,int(lead)) for campaign in range(min(config.eval_campaigns,len(dataset.campaign_ids)))
                   for lead in rng.choice(dataset.eligible_leads,size=config.leads_per_campaign,replace=False)]
        manifest['checkpoints'].append({'path':str(checkpoint),'sha256':digest(project_path(checkpoint)),
            'step':payload['step'],'original_protocol':str(protocol_path) if protocol_path.exists() else None})
        for local_case,(campaign,lead_index) in enumerate(cases[:args.cases_per_checkpoint]):
            if time.perf_counter()-started>=args.max_seconds:
                manifest['stopped_by_time_cap']=True
                break
            destination=output/f'case_{global_case:03d}'
            destination.mkdir()
            nested=dataset.nested_support(campaign,lead_index,seed=config.eval_seed+local_case,sizes=(16,128))
            lead=collate_graphs([nested['lead']]).to(args.device)
            baseline=torch.as_tensor(nested['lead_values'],device=args.device,dtype=torch.float32)[None]
            context=torch.as_tensor(dataset.context,device=args.device,dtype=torch.float32)[None]
            goal=torch.stack((torch.full_like(baseline[:,0],.25),baseline[:,1]-.1),-1)
            conditions={}
            with torch.no_grad():
                for size in (16,128):
                    support=_support(nested['records'][size],features,args.device)
                    conditions[f'n{size}']=model.condition(lead,baseline,context,model.support_state(support),goal)
                perm=np.r_[0,np.random.default_rng(config.eval_seed+local_case+771).permutation(np.arange(1,128))]
                shuffled=replace(support,values=support.values[:,perm].clone())
                conditions['permuted128']=replace(conditions['n128'],response_state=model.support_state(shuffled))
            # Replay the original count/noise law, with identical draws in all arms.
            from .redesign_sampling import make_sampling_plan
            prior=dataset.count_prior
            plan=make_sampling_plan(model.config,prior,int(lead.counts[0]),args.trajectories,
                count_seed=config.eval_seed+local_case*2,noise_seed=config.eval_seed+local_case*2+1)
            noise=Latents(plan.noise.nodes.to(args.device),plan.noise.pairs.to(args.device))
            counts=plan.counts.to(args.device)
            final,masks,snapshots=reference_trajectory(model,noise,counts,conditions['n16'],nfe=args.nfe)
            stages=[]
            jacobians=[]
            for probe,z in snapshots.items():
                for trajectory in range(args.trajectories):
                    index=torch.tensor([trajectory],device=args.device)
                    one_z=Latents(z.nodes[index],z.pairs[index])
                    one_masks=GraphMasks(masks.nodes[index],masks.pairs[index])
                    t=torch.tensor([probe],device=args.device)
                    for arm in ('n128','permuted128'):
                        stages.append({'trajectory':trajectory,'time':probe,'comparison':f'{arm}-n16',
                            **stage_comparison(model,one_z,t,one_masks,conditions['n16'],conditions[arm])})
                    stages.append({'trajectory':trajectory,'time':probe,'comparison':'permuted128-n128',
                        **stage_comparison(model,one_z,t,one_masks,conditions['n128'],conditions['permuted128'])})
                    if global_case<args.jacobian_cases and trajectory<args.jacobian_trajectories:
                        _,metrics=exact_composed_jacobian(model,one_z,t,one_masks,conditions['n16'])
                        jacobians.append({'trajectory':trajectory,'time':probe,**metrics})
            finals={'n16':final}
            for arm in ('n128','permuted128'):
                finals[arm],_,_=reference_trajectory(model,noise,counts,conditions[arm],nfe=args.nfe,probes=())
            proposals=[]
            with torch.no_grad():
                for arm,latent in finals.items():
                    logits=model.decode(latent,masks)
                    atoms,bonds=logits.atoms.argmax(-1).cpu(),logits.bonds.argmax(-1).cpu()
                    for i,count in enumerate(counts.tolist()):
                        outcome=decode_graph(atoms[i],bonds[i],count,max_nodes=model.config.max_nodes)
                        valid=outcome.chemically_valid and outcome.connected
                        supported=False
                        if valid:
                            try:
                                smiles_to_graph(outcome.smiles,model.config.max_nodes)
                                supported=True
                            except ValueError:
                                pass
                        proposals.append({'arm':arm,'trajectory':i,'smiles':outcome.smiles,
                            'valid':valid,'supported':supported,'failure_reason':outcome.failure_reason,
                            'local':bool(valid and neighbourhood(outcome.smiles,nested['lead']['smiles'])['eligible']),
                            'support_match':bool(outcome.smiles in nested['reserve_smiles']),
                            'atoms':atoms[i].tolist(),'bonds':bonds[i].tolist()})
            # Commit exact decoded proposals before loading private truth.
            atomic_json(destination/'proposals.json',proposals)
            oracle=ToyOracle.load(project_path(config.data_dir)/'private'/'oracle.json')
            calls=0
            truth={}
            for row in proposals:
                if row['valid'] and row['supported'] and not row['support_match']:
                    if row['smiles'] not in truth:
                        truth[row['smiles']]=oracle.score_smiles(dataset.campaign_ids[campaign],row['smiles'])
                        calls+=1
                    y=truth[row['smiles']]
                    row['profile_success']=bool(y[0]-float(baseline[0,0])>=.25 and y[1]>=float(goal[0,1]))
                else:
                    row['profile_success']=False
                row['local_profile_success']=row['profile_success'] and row['local']
            withheld=np.setdiff1d(np.arange(len(dataset)),nested['reserve_indices'])
            response=[]
            with torch.no_grad():
                for arm,condition in conditions.items():
                    state=condition.expand(len(withheld))
                    distribution=model.predict_response_features(state.lead_features,
                        features[withheld].to(args.device),state.baseline,state.context,state.response_state)
                    labels=torch.as_tensor(dataset.observations[campaign,withheld],device=args.device)-baseline
                    response.append({'arm':arm,'withheld_count':len(withheld),
                        'mean_log_likelihood':float(distribution.log_prob(labels).mean()),
                        'predicted_mean':distribution.mean.cpu().tolist(),
                        'response_mse':float((distribution.mean-labels).square().mean()),
                        'endpoint1_sign_accuracy':float(((distribution.mean[:,0]>=0)==(labels[:,0]>=0)).float().mean()),
                        'query_indices':withheld.tolist()})
            case_summary={}
            for arm in conditions:
                selected=[row for row in proposals if row['arm']==arm]
                reference=[row for row in proposals if row['arm']=='n16']
                seen=set()
                distinct_hits=0
                for row in selected:
                    if row['local_profile_success'] and row['smiles'] not in seen:
                        distinct_hits+=1
                    if row['valid']:
                        seen.add(row['smiles'])
                case_summary[arm]={'attempts':len(selected),
                    'valid_per_attempt':sum(row['valid'] for row in selected)/len(selected),
                    'local_profile_slots_per_attempt':sum(row['local_profile_success'] for row in selected)/len(selected),
                    'distinct_local_profile_per_attempt':distinct_hits/len(selected),
                    'paired_changed_vs_n16':sum(a['smiles']!=b['smiles'] for a,b in zip(selected,reference))/len(selected)}
            atomic_json(destination/'summary.json',case_summary)
            atomic_json(destination/'stages.json',stages)
            atomic_json(destination/'jacobians.json',jacobians)
            atomic_json(destination/'outcomes.json',proposals)
            atomic_json(destination/'response.json',response)
            atomic_json(destination/'case.json',{'checkpoint':str(checkpoint),'campaign':dataset.campaign_ids[campaign],
                'lead_index':lead_index,'lead_smiles':nested['lead']['smiles'], 'support_indices':nested['reserve_indices'].tolist(),
                'permutation':perm.tolist(),'oracle_calls':calls,'noise_seed':plan.noise_seed,'nfe':args.nfe,
                'elapsed_seconds':time.perf_counter()-started})
            global_case+=1
            manifest['completed_cases']=global_case
            manifest['elapsed_seconds']=time.perf_counter()-started
            atomic_json(output/'manifest.json',manifest)
        if manifest.get('stopped_by_time_cap'):
            break
    manifest['elapsed_seconds']=time.perf_counter()-started
    manifest['complete']=not manifest.get('stopped_by_time_cap',False)
    atomic_json(output/'manifest.json',manifest)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoints',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--split',default='test',choices=('validation','test'))
    parser.add_argument('--cases-per-checkpoint',type=int,default=8)
    parser.add_argument('--trajectories',type=int,default=16)
    parser.add_argument('--jacobian-cases',type=int,default=8)
    parser.add_argument('--jacobian-trajectories',type=int,default=2)
    parser.add_argument('--nfe',type=int,default=32)
    parser.add_argument('--max-seconds',type=float,default=1800)
    run(parser.parse_args())


if __name__=='__main__':
    main()
