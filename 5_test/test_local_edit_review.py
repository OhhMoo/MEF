"""Independent local-edit algebra, alignment, and conditioning checks."""
import itertools
import numpy as np
import torch
from emf.contracts import ModelConfig,Latents
from emf.models import EMFModel
from emf.redesign_contracts import RedesignConfig
from emf.redesign import FactorizedFlow
from emf.local_edit_factorized import NormalizedFactorizedFlow
from emf.local_edit_data import _minimum_edit_permutation
from emf.local_edit_model import LocalEditModel
from emf.local_edit_sampling import anchored_initial_noise
from emf.data import collate_graphs
from emf.chem import smiles_to_graph
from emf.redesign_sampling import sample_redesign_latents


def test_exact_alignment_matches_exhaustive_permutations():
    rng=np.random.default_rng(53)
    n=4
    left,right=np.triu_indices(n,1)
    for _ in range(12):
        graphs=[{'count':n,'atoms':rng.integers(0,4,n,dtype=np.uint8),
                 'bonds':rng.integers(0,4,len(left),dtype=np.uint8)} for _ in range(2)]
        lead,candidate=graphs
        matrices=[]
        for graph in graphs:
            matrix=np.zeros((n,n),dtype=np.uint8)
            matrix[left,right]=graph['bonds'];matrix+=matrix.T
            matrices.append(matrix)
        def cost(permutation):
            atom=np.count_nonzero(candidate['atoms']!=lead['atoms'][list(permutation)])
            bond=sum(matrices[1][i,j]!=matrices[0][permutation[i],permutation[j]] for i,j in zip(left,right))
            return int(atom+bond)
        expected=min((cost(p),p) for p in itertools.permutations(range(n)))
        permutation,value=_minimum_edit_permutation(lead,candidate)
        assert (value,tuple(permutation))==expected


def test_normalization_only_variant_preserves_every_initial_parameter():
    torch.set_num_threads(1)
    config=RedesignConfig(model=ModelConfig(max_nodes=4,node_dim=8,pair_dim=4,hidden_dim=16,depth=1,num_heads=2),rank=2,state_dim=3,context_dim=2)
    base=EMFModel(config.model).flow
    torch.manual_seed(19);ordinary=FactorizedFlow(config,base)
    torch.manual_seed(19);normalized=NormalizedFactorizedFlow(config,base)
    new=normalized.state_dict()
    for name,value in ordinary.state_dict().items():
        corresponding='selector.selector.'+name[len('selector.'):] if name.startswith('selector.') else name
        torch.testing.assert_close(value,new[corresponding],rtol=0,atol=0)
    assert sum(p.numel() for p in ordinary.parameters())==sum(p.numel() for p in normalized.parameters())
    # First layer of wrapped selector sees RMS-one trunk pools; task stays exact.
    observed=[]
    hook=normalized.selector.selector[0].register_forward_pre_hook(lambda module,args:observed.append(args[0]))
    x=torch.randn(3,2*16+config.feature_dim+config.context_dim+config.state_dim+6)
    x[:,:32]*=100
    normalized.selector(x)
    hook.remove()
    pools=observed[0][:,:32].reshape(3,2,16)
    torch.testing.assert_close(pools.square().mean(-1),torch.ones(3,2),rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(observed[0][:,32:],x[:,32:],rtol=0,atol=0)


def test_zero_residual_initialization_returns_full_lead_through_sampler():
    torch.manual_seed(41)
    config=RedesignConfig(model=ModelConfig(max_nodes=4,node_dim=8,pair_dim=4,hidden_dim=16,depth=1,num_heads=2),rank=2,state_dim=3,context_dim=2)
    model=LocalEditModel(config,EMFModel(config.model))
    lead=collate_graphs([dict(smiles_to_graph('CCO',4),id='lead')])
    condition=model.condition(lead,torch.zeros(1,2),torch.zeros(1,2),torch.tensor([[1.,0.,0.]]),torch.tensor([[.25,-.1]]))
    eps=Latents(torch.randn_like(condition.lead.nodes),torch.randn_like(condition.lead.pairs))
    start=anchored_initial_noise(eps,lead.masks,condition,sigma=.4)
    torch.testing.assert_close(start.nodes,(condition.lead.nodes+.4*eps.nodes)*lead.masks.nodes[...,None])
    model.freeze_deployment()
    result=sample_redesign_latents(model,lead.counts,condition,nfe=8,initial_noise=start)
    torch.testing.assert_close(result.latents.nodes,condition.lead.nodes,atol=0,rtol=0)
    torch.testing.assert_close(result.latents.pairs,condition.lead.pairs,atol=0,rtol=0)
    assert result.nfe==8
    assert not start.nodes[~lead.masks.nodes].any()
    assert not start.pairs[~lead.masks.pairs].any()


def test_two_constraint_projection_handles_satisfied_endpoint_and_infeasibility():
    from emf.local_edit_control import _minimum_norm_two_constraints
    jacobian=torch.tensor([[[1.,0.],[1.,1.]],[[1.,0.],[-1.,1.]],[[1.,0.],[-1.,0.]]])
    gaps=torch.tensor([[1.,-.5],[1.,-.2],[1.,0.]])
    delta,feasible=_minimum_norm_two_constraints(jacobian,gaps,1e-4)
    torch.testing.assert_close(delta[:2],torch.tensor([[1.,0.],[1.,.8]]),rtol=1e-6,atol=1e-6)
    assert feasible.tolist()==[True,True,False]
    assert torch.isfinite(delta).all()


def test_controller_masks_padding_and_clips_by_all_active_coordinates():
    from emf.contracts import GraphMasks
    from emf.local_edit_control import correct_clean
    class LinearHead(torch.nn.Module):
        def forward(self,lead,candidate):
            return candidate[:,:2]-lead[:,:2]
    masks=GraphMasks.from_counts(torch.tensor([2]),3)
    nodes=torch.zeros(1,3,2);nodes[:,2]=999
    pairs=torch.zeros(1,3,1);pairs[:,1:]=999
    clean=Latents(nodes,pairs)
    with torch.no_grad():
        corrected,stats=correct_clean(clean,masks,torch.zeros(1,4),torch.zeros(1,2),
            torch.tensor([[1.,.5]]),1.,LinearHead(),radius=.1)
    assert not corrected.nodes[~masks.nodes].any()
    assert not corrected.pairs[~masks.pairs].any()
    delta=torch.cat((corrected.nodes[masks.nodes].flatten(),corrected.pairs[masks.pairs].flatten()))
    torch.testing.assert_close(delta.square().mean().sqrt(),torch.tensor(.1),rtol=1e-6,atol=1e-6)
    assert stats['radius_clipped']==[True]
    assert stats['linearized_feasible']==[True]  # Before clipping only.
    assert all(v>0 for v in stats['remaining_deficit_after'][0])
    assert stats['surrogate_backward_calls']==2
    assert not corrected.nodes.requires_grad and not corrected.pairs.requires_grad


def test_projection_is_consistent_under_small_jacobian_scale():
    from emf.local_edit_control import _minimum_norm_two_constraints
    jacobian=torch.eye(2)[None]*1e-6
    delta,feasible=_minimum_norm_two_constraints(jacobian,torch.ones(1,2),1e-4)
    assert feasible.tolist()==[True]
    torch.testing.assert_close(delta,torch.full((1,2),1e6),rtol=1e-5,atol=1.)


def test_bridge_uses_source_support_only_and_preserves_generator(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pytest
    import emf.local_edit_calibration as bridge
    class Source:
        split='train'
        campaign_ids=('source_plus','source_minus')
        support_sizes=(4,16,64)
        graph_dataset=object()
        manifest={'splits':{'train':{'campaign_ids':list(campaign_ids)},
                            'validation':{'campaign_ids':['val']},
                            'test':{'campaign_ids':['test']}}}
        def sample_episode(self,rng,support_size,queries_per_lead,campaign_index):
            coefficient=(1.,-1.)[campaign_index]
            # No query field exists: accidental query use must fail here.
            return SimpleNamespace(support_indices=np.arange(support_size)%4,
                support_values=np.tile([coefficient,0.],(support_size,1)).astype('float32'),
                context=np.zeros(2,dtype='float32'),
                campaign_id=self.campaign_ids[campaign_index])
    class Frozen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor=torch.nn.Parameter(torch.ones(1))
            self.redesign_config=SimpleNamespace(feature_dim=4,state_dim=64,context_dim=2)
        def support_state(self,support):
            assert not self.training and not self.anchor.requires_grad
            state=torch.zeros(len(support.values),64)
            state[:,0]=(support.values[:,:,0]*support.mask).sum(1)/support.mask.sum(1)
            return state*self.anchor
    monkeypatch.setattr(bridge,'cache_graph_features',lambda *a,**k:(torch.zeros(4,4),0.))
    model=Frozen();before={k:v.clone() for k,v in model.state_dict().items()}
    dataset=Source()
    head,summary=bridge.train_bridge(model,dataset,{'source_plus':1.,'source_minus':-1.},
        'cpu',checkpoint_path=tmp_path/'bridge.pt',steps=20,episodes_per_size=4,
        validation_per_size=2)
    assert summary['training_episodes']==12
    assert summary['source_campaign_ids']==['source_plus','source_minus']
    assert summary['generator_sha256_before']==summary['generator_sha256_after']
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value,before[name],rtol=0,atol=0)
    assert model.training and model.anchor.requires_grad  # deployment flags restored.
    mapped=bridge.bridge_state(head,torch.randn(3,64))
    assert mapped.shape==(3,64) and not mapped[:,1:].any()
    assert not any(p.requires_grad for p in head.parameters())
    dataset.split='test'
    with pytest.raises(ValueError,match='source train split'):
        bridge.train_bridge(model,dataset,{},'cpu',checkpoint_path=tmp_path/'forbidden.pt',steps=1)
    assert not (tmp_path/'forbidden.pt').exists()


def test_evaluation_reuses_exact_full_lead_and_initial_noise(monkeypatch,tmp_path):
    """Inject encoder rounding drift; an evidence swap must not re-encode X0."""
    from dataclasses import replace
    import emf.local_edit_run as runner
    import emf.steering_evaluation as evaluator
    from emf.redesign_contracts import FlowCondition
    from emf.redesign_sampling import make_sampling_plan
    from emf.redesign_train import RedesignRunConfig
    config=RedesignConfig(model=ModelConfig(max_nodes=4,node_dim=8,pair_dim=4,
        hidden_dim=16,depth=1,num_heads=2),rank=2,state_dim=3,context_dim=2)
    model=LocalEditModel(config,EMFModel(config.model))
    lead=collate_graphs([dict(smiles_to_graph('CCO',4),id='lead')])
    calls=[];original=model.condition
    def drifting_condition(*args,**kwargs):
        calls.append(1)
        condition=original(*args,**kwargs)
        return replace(condition,lead=Latents(condition.lead.nodes+len(calls)*1e-4,
                                             condition.lead.pairs+len(calls)*1e-4))
    monkeypatch.setattr(model,'condition',drifting_condition)
    cfg=RedesignRunConfig(redesign=config,output_dir=str(tmp_path),device='cpu',seed=142)
    monkeypatch.setattr(runner,'load_redesign_config',lambda path:cfg)
    def fake_evaluate(*args,**kwargs):
        prepare=kwargs['proposal_setup']
        condition=FlowCondition(torch.zeros(1,config.feature_dim),torch.zeros(1,2),
                                torch.zeros(1,2),torch.zeros(1,3),torch.tensor([[.25,-.1]]))
        prior=torch.zeros(4,4);prior[2,2]=1
        plan=make_sampling_plan(config.model,prior,3,2,count_seed=1,noise_seed=2)
        left,left_plan=prepare(model,condition,plan,lead,'positive',None,None,None)
        other=replace(condition,response_state=torch.ones(1,3))
        right,right_plan=prepare(model,other,plan,lead,'negative',None,None,None)
        assert len(calls)==1
        for field in ('nodes','pairs'):
            assert torch.equal(getattr(left.lead,field),getattr(right.lead,field))
            assert torch.equal(getattr(left_plan.noise,field),getattr(right_plan.noise,field))
        assert not torch.equal(left.response_state,right.response_state)
        return {'paired_anchor_and_noise_exact':True}
    monkeypatch.setattr(evaluator,'run_evaluation',fake_evaluate)
    result=runner.evaluate({'config':'unused','kind':'anchored_known','operation':True},
                           {'scorers':{'142':'unused'}},'validation')
    assert result['paired_anchor_and_noise_exact']
