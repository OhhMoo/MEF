"""Independent active-coordinate sensitivity and paired trajectory review."""
from dataclasses import replace
import torch
from emf.contracts import ModelConfig, GraphMasks, Latents
from emf.redesign import RedesignModel
from emf.redesign_contracts import RedesignConfig, FlowCondition
from emf.redesign_sampling import sample_redesign_latents
from emf.steering_trace import exact_composed_jacobian, active_flat, reference_trajectory, stage_comparison


def fixture():
    torch.manual_seed(777)
    torch.set_num_threads(1)
    config=ModelConfig(max_nodes=4,node_dim=4,pair_dim=2,hidden_dim=16,depth=1,num_heads=2)
    model=RedesignModel(RedesignConfig(model=config,rank=2,state_dim=5,context_dim=2))
    torch.nn.init.normal_(model.flow.selector[-1].weight,std=.2)
    model.float().freeze_deployment()
    counts=torch.tensor([3])
    masks=GraphMasks.from_counts(counts,4)
    noise=Latents(torch.randn(1,4,4),torch.randn(1,6,2)).masked(masks)
    condition=FlowCondition(torch.randn(1,7),torch.zeros(1,2),torch.zeros(1,2),
                            torch.randn(1,5),torch.tensor([[.25,-.1]]))
    return model,counts,masks,noise,condition


def test_composed_jacobian_matches_direct_autograd_and_rank():
    model,counts,masks,z,condition=fixture()
    t=torch.tensor([.5])
    jac,metrics=exact_composed_jacobian(model,z,t,masks,condition)
    def full(state):
        return active_flat(model.predict_clean(z,t,masks,replace(condition,response_state=state)),masks)
    direct=torch.autograd.functional.jacobian(full,condition.response_state).squeeze(1)
    torch.testing.assert_close(jac,direct,atol=2e-7,rtol=1e-5)
    assert metrics['numerical_rank']<=2
    assert jac.shape==(3*4+3*2,5)
    assert metrics['frobenius_norm']>0


def test_reference_sampler_preserves_original_path_and_nfe():
    model,counts,masks,z,condition=fixture()
    traced,trace_masks,probes=reference_trajectory(model,z,counts,condition,nfe=8)
    sampled=sample_redesign_latents(model,counts,condition,nfe=8,initial_noise=z)
    torch.testing.assert_close(traced.nodes,sampled.latents.nodes,atol=0,rtol=0)
    torch.testing.assert_close(traced.pairs,sampled.latents.pairs,atol=0,rtol=0)
    assert set(probes)=={.1,.5,.9}
    assert sampled.nfe==8
    for value in probes.values():
        assert not value.nodes[~masks.nodes].any()
        assert not value.pairs[~masks.pairs].any()


def test_evidence_intervention_cannot_change_chemical_base_or_basis():
    model,counts,masks,z,condition=fixture()
    alternate=replace(condition,response_state=condition.response_state+1)
    metrics=stage_comparison(model,z,torch.tensor([.9]),masks,condition,alternate)
    assert metrics['base_delta_l2']==0
    assert metrics['basis_delta_l2']==0
    assert metrics['correction_l2']>0
    assert abs(metrics['velocity_delta_l2']/metrics['correction_l2']-10)<1e-4


def test_committed_steering_dataset_has_paired_inputs_and_separate_queries():
    import json
    from pathlib import Path
    import numpy as np
    from emf.steering_data import SteeringDataset, SteeringOracle, GOAL
    root=Path(__file__).resolve().parents[1]
    path=root/'2_data_storage/processed/evidence_steering_toy'
    dataset=SteeringDataset(path,'test')
    oracle=SteeringOracle.load(path/'private/oracle.json')
    feasibility=json.loads((path/'feasibility.json').read_text())['cases']
    by_id={row['campaign_id']:row for row in feasibility}
    assert all(len(family)>=128 for family in dataset.family_indices)
    for i,name in enumerate(dataset.campaign_ids):
        opposite=dataset.campaign_ids.index(dataset.campaign_pairs[name])
        lead=dataset.campaign_lead_indices[name]
        a=dataset.sample_episode(np.random.default_rng(33),support_size=64,campaign_index=i)
        b=dataset.sample_episode(np.random.default_rng(33),support_size=64,campaign_index=opposite)
        np.testing.assert_array_equal(a.support_indices,b.support_indices)
        np.testing.assert_array_equal(a.query_indices,b.query_indices)
        np.testing.assert_array_equal(a.lead_values,np.zeros(2))
        np.testing.assert_array_equal(a.context,b.context)
        np.testing.assert_array_equal(a.goals,np.tile(GOAL,(len(a.query_indices),1)))
        assert not set(a.query_indices)&set(a.support_indices)
        np.testing.assert_allclose(a.support_values[:,0],-b.support_values[:,0])
        np.testing.assert_allclose(a.support_values[:,1],b.support_values[:,1])
        nested=dataset.nested_support(i,lead,seed=dataset.paired_support_seed(name))
        np.testing.assert_array_equal(nested['reserve_indices'],by_id[name]['reserve_indices'])
        assert by_id[name]['feasible_outside_max_support']>0
        assert oracle.coefficients[name]==-oracle.coefficients[dataset.campaign_pairs[name]]
    private=json.loads((path/'private/oracle.json').read_text())
    train=private['coefficient_magnitudes']['train']
    held=private['coefficient_magnitudes']['validation']+private['coefficient_magnitudes']['test']
    assert min(abs(a-b) for a in train for b in held)>.04


def test_privileged_override_changes_only_flow_condition(monkeypatch):
    from emf.chem import smiles_to_graph
    from emf.data import collate_graphs
    from emf.redesign_contracts import CampaignBatch, SupportBatch
    from emf.redesign_objectives import redesign_loss
    model,_,_,_,_=fixture()
    graphs=collate_graphs([dict(smiles_to_graph(smiles,4),id=smiles) for smiles in ('CCO','CCN')])
    features=model.graph_features(graphs)
    support=SupportBatch(features[None],torch.tensor([[[0.,0.],[.5,0.]]]),
                         torch.zeros(1,2),torch.ones(1,2,dtype=torch.bool))
    batch=CampaignBatch(graphs,graphs,support,torch.zeros(2,dtype=torch.long),
        torch.zeros(2,2),torch.zeros(2,2),torch.tensor([[.25,-.1],[.25,-.1]]),
        torch.tensor([[.5,0.],[.5,0.]]),torch.ones(2))
    original_flow=model.predict_clean
    original_response=model.predict_response_features
    observed={}
    def flow(z,t,masks,condition):
        observed['flow']=condition.response_state.detach().clone()
        return original_flow(z,t,masks,condition)
    def response(lead,candidate,baseline,context,state):
        observed['response']=state.detach().clone()
        return original_response(lead,candidate,baseline,context,state)
    monkeypatch.setattr(model,'predict_clean',flow)
    monkeypatch.setattr(model,'predict_response_features',response)
    _,normal=redesign_loss(model,batch,generator=torch.Generator().manual_seed(12))
    inferred=observed['response'].clone()
    privileged=torch.zeros_like(inferred)
    privileged[:,0]=1.3
    _,diagnostic=redesign_loss(model,batch,generator=torch.Generator().manual_seed(12),
                              flow_state_override=privileged)
    torch.testing.assert_close(observed['flow'],privileged,rtol=0,atol=0)
    torch.testing.assert_close(observed['response'],inferred,rtol=0,atol=0)
    torch.testing.assert_close(normal['response_loss'],diagnostic['response_loss'],rtol=0,atol=0)
    torch.testing.assert_close(normal['decode_loss'],diagnostic['decode_loss'],rtol=0,atol=0)


def test_workflow_selects_largest_available_common_update(tmp_path,monkeypatch):
    import importlib.util
    import json
    from pathlib import Path
    import yaml
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('review_workflow',root/'3_model_training/scripts/steering_workflow.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module,'STUDY',tmp_path)
    tasks=[]
    for name,final in [('r8',19013),('r32',18215),('conditional',20000)]:
        output=tmp_path/name
        checkpoints=output/'checkpoints'
        checkpoints.mkdir(parents=True)
        for step in (16000,18000):
            (checkpoints/f'step_{step:07d}.pt').write_bytes(str(step).encode())
        (checkpoints/'source_final.pt').write_bytes(str(final).encode())
        (output/'source_summary.json').write_text(json.dumps({'step':final}))
        config=output/'config.yaml'
        config.write_text(yaml.safe_dump({'output_dir':str(output)}))
        tasks.append({'name':name,'config':str(config)})
    selected=module.choose_checkpoints({'stages':{'source':{'tasks':tasks}}})
    assert selected['step']==18000
    assert all(path.endswith('step_0018000.pt') for path in selected['checkpoints'].values())
