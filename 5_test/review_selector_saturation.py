"""Read-only selector activation and local-gradient audit of trained checkpoints."""
from pathlib import Path
from dataclasses import replace
from datetime import datetime,timezone
import json
import numpy as np
import torch
from emf.redesign_train import load_source_checkpoint,load_initializer,cache_graph_features,_support,digest
from emf.steering_data import SteeringDataset
from emf.data import collate_graphs
from emf.contracts import Latents
from emf.redesign_objectives import weighted_velocity_matching_loss
from emf.steering_trace import exact_composed_jacobian,reference_trajectory,active_flat

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'5_test/results/selector_saturation_review.json'
STUDY=ROOT/'4_experimental_result/result/evidence_steering_001'
torch.set_num_threads(1)
torch.set_float32_matmul_precision('highest')
dataset=SteeringDataset(ROOT/'2_data_storage/processed/evidence_steering_toy','train')
private=json.loads((dataset.data_dir/'private/oracle.json').read_text())


def normalize_pool(module,args):
    values=args[0]
    h=128
    left,right=values[:,:h],values[:,h:2*h]
    normalized=[v/torch.sqrt(v.square().mean(-1,keepdim=True)+1e-6) for v in (left,right)]
    return (torch.cat((*normalized,values[:,2*h:]),-1),)


def probe(model,z,target,t,masks,condition,alternate,normalization=False):
    captures=[]
    normalization_hook=model.flow.selector.register_forward_pre_hook(normalize_pool) if normalization else None
    hook=model.flow.selector.register_forward_pre_hook(lambda module,args:captures.append(args[0].detach()))
    prediction=model.flow.decompose(z,t,masks,condition)
    alt=model.flow.decompose(z,t,masks,alternate)
    hook.remove()
    values=captures[0]
    first=model.flow.selector[0](values)
    second=model.flow.selector[2](model.flow.selector[1](first))
    loss=weighted_velocity_matching_loss(prediction.compose(),target,t,masks,torch.ones_like(t))
    parameters=[(name,p) for name,p in model.flow.named_parameters() if p.requires_grad]
    gradients=torch.autograd.grad(loss,[p for name,p in parameters],allow_unused=True)
    groups={name:0. for name in ('base','basis','selector_head_weight','selector_head_bias','selector_hidden')}
    for (name,p),grad in zip(parameters,gradients):
        if grad is None:
            continue
        group=('selector_head_weight' if name=='selector.4.weight' else 'selector_head_bias' if name=='selector.4.bias' else 'selector_hidden' if name.startswith('selector.')
               else 'basis' if 'basis' in name else 'base')
        groups[group]+=float(grad.square().sum())
    groups={key:value**.5 for key,value in groups.items()}
    correction=active_flat(prediction.compose(),masks)-active_flat(prediction.base,masks)
    # Form the intervention directly so subtraction of F cannot hide small changes.
    da=alt.strengths-prediction.strengths
    change=Latents(torch.einsum('bndr,br->bnd',prediction.node_directions,da),
                   torch.einsum('bpdr,br->bpd',prediction.pair_directions,da))
    _,jacobian=exact_composed_jacobian(model,z,t,masks,condition)
    result={'normalized_pool_counterfactual':normalization,
       'pool_node_rms':float(values[:,:128].square().mean().sqrt()),
       'pool_pair_rms':float(values[:,128:256].square().mean().sqrt()),
       'other_input_rms':float(values[:,256:].square().mean().sqrt()),
       'second_preactivation_min':float(second.min()),'second_preactivation_max':float(second.max()),
       'fraction_second_below_minus20':float((second < -20).float().mean()),
       'fraction_second_below_minus10':float((second < -10).float().mean()),
       'strength_l2':float(prediction.strengths.norm()),'strength_minus_bias_l2':float((prediction.strengths-model.flow.selector[-1].bias).norm()),
       'composed_correction_l2':float(correction.norm()),'intervention_delta_l2':float(active_flat(change,masks).norm()),
       'jacobian_frobenius':jacobian['frobenius_norm'],'flow_loss':float(loss),
       'flow_only_gradient_norms':groups}
    if normalization_hook:
        normalization_hook.remove()
    return result


rows=[]
checkpoints={name:STUDY/name/'checkpoints/source_final.pt' for name in
             ('source/r8_seed_142','source/r8_seed_143','privileged/seed_142')}
for name,path in checkpoints.items():
    trained,config,payload=load_source_checkpoint(path,'cpu')
    models=[(name,trained)]
    if name=='source/r8_seed_142':
        torch.manual_seed(config.seed)
        initialized,_=load_initializer(config,'cpu')
        models.insert(0,('initializer_r8_seed_142',initialized))
    for model_name,model in models:
        model.eval()
        features,_=cache_graph_features(model,dataset.graph_dataset,bf16=False)
        for family,lead_index in enumerate(dataset.family_leads):
            campaign=next(i for i,campaign in enumerate(dataset.campaign_ids) if dataset.campaign_family[campaign]==family)
            campaign_id=dataset.campaign_ids[campaign]
            opposite=dataset.campaign_ids.index(dataset.campaign_pairs[campaign_id])
            nested=dataset.nested_support(campaign,lead_index,seed=9100+family,sizes=(64,))
            support=_support(nested['records'][64],features,'cpu')
            lead=collate_graphs([nested['lead']])
            condition=model.condition(lead,torch.zeros(1,2),torch.tensor(dataset.context)[None],
                                      model.support_state(support),torch.tensor([[.25,-.1]]))
            support_flip=replace(support,values=torch.tensor(dataset.observations[opposite,nested['reserve_indices']])[None])
            alternate=replace(condition,response_state=model.support_state(support_flip))
            if model_name.startswith('privileged'):
                state=torch.zeros(1,64);state[:,0]=private['coefficients'][campaign_id]
                condition=replace(condition,response_state=state)
                alternate=replace(condition,response_state=-state)
            condition=replace(condition,response_state=condition.response_state.detach())
            alternate=replace(alternate,response_state=alternate.response_state.detach())
            # First feasible source analogue is fixed by index, never selected by model outputs.
            members=dataset.family_indices[family]
            eligible=members[(dataset.observations[campaign,members,0]>=.25)&(dataset.observations[campaign,members,1]>=-.1)]
            query=collate_graphs([dataset.graph(int(eligible[0]))])
            with torch.no_grad():
                target=model.encode(query)
            rng=torch.Generator().manual_seed(9182+family)
            noise=Latents(torch.randn(target.nodes.shape,generator=rng),torch.randn(target.pairs.shape,generator=rng))
            _,_,trajectory=reference_trajectory(model,noise,lead.counts,condition,nfe=32)
            for time in (.1,.5,.9):
                interpolated=Latents(time*target.nodes+(1-time)*noise.nodes,time*target.pairs+(1-time)*noise.pairs)
                for state_kind,z in (('source_interpolation',interpolated),('reference_trajectory',trajectory[time])):
                    for normalized in (False,True):
                        record=probe(model,z,target,torch.tensor([time]),lead.masks,condition,alternate,normalized)
                        rows.append({'checkpoint':model_name,'family':family,'time':time,'state_kind':state_kind,**record})
        print(model_name,'complete',flush=True)
summary=[]
for name in sorted({row['checkpoint'] for row in rows}):
    for normalized in (False,True):
        group=[row for row in rows if row['checkpoint']==name and row['normalized_pool_counterfactual']==normalized]
        summary.append({'checkpoint':name,'normalized_pool_counterfactual':normalized,'probes':len(group),
            'all_second_units_below_minus10_probes':sum(row['second_preactivation_max'] < -10 for row in group),
            **{key:float(np.median([row[key] for row in group])) for key in
                ('pool_node_rms','pool_pair_rms','fraction_second_below_minus20','fraction_second_below_minus10',
                 'second_preactivation_max','strength_minus_bias_l2','intervention_delta_l2','jacobian_frobenius')},
            'median_flow_gradients':{key:float(np.median([row['flow_only_gradient_norms'][key] for row in group]))
                                    for key in ('base','basis','selector_head_weight','selector_head_bias','selector_hidden')}})
report={'reviewed_utc':datetime.now(timezone.utc).isoformat(),'device':'cpu','precision':'float32',
        'protocol':'Two source families; first positive source campaign per family; first feasible query by index; source-interpolation and actual32NFE trajectories at t=.1,.5,.9. Same states for native and flipped outcomes. No optimizer updates.',
        'counterfactual_scope':'Normalizing pooled inputs of already trained weights diagnoses sensitivity; it is not a trained intervention or a generation-quality result.',
        'checkpoint_sha256':{name:digest(path) for name,path in checkpoints.items()},'summary':summary,'probes':rows}
DEST.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(summary,indent=2))
