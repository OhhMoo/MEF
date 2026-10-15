"""Frozen-generator control experiments; exact truth is read after commitment."""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from .contracts import GraphMasks, GraphLogits, Latents
from .masked_edit_run import (load_model,condition_for,graphs_from_logits,write_csv,now)
from .masked_edit_data import MaskedEditDataset,fit_support_response,_descriptor,_features
from .masked_edit_model import decode_masked_edit
from .masked_edit_sampling import sample_masked_edit_latents,make_paired_gaussian
from .masked_edit_control import SoftGraphDescriptorCritic,optimize_terminal_control,descriptor_features
from .masked_edit_evaluation import ExactMaskedEvaluator
from .redesign_train import atomic_json,tensor_fingerprint


def load_critic(path,model,device):
    p=torch.load(path,map_location='cpu',weights_only=False)
    critic=SoftGraphDescriptorCritic(model.config,hidden_dim=p.get('hidden_dim',64),depth=p.get('depth',2)).to(device)
    critic.load_state_dict(p.get('state_dict',p.get('model',p.get('critic'))));critic.eval().requires_grad_(False)
    return critic,p

def score_prediction(model,critic,result,condition,W,lead_raw,scales):
    with torch.no_grad():
        logits=decode_masked_edit(model,result.latents,condition)
        pred=critic(logits,result.masks)
        features=descriptor_features(pred,lead_raw/scales)
        return logits,torch.einsum('bij,bj->bi',W,features)

def truth_rows(graphs,pred,oracle,dataset,cid,support,metadata):
    mask=dataset.masks[dataset.manifest['campaigns'][cid]['mask_id']];rows=[]
    for i,graph in enumerate(graphs):
        r=oracle.evaluate_graph(graph,mask,cid,support_smiles=support['reserve_smiles'],source_smiles=[dataset.smiles[j] for j in dataset.source_graph_indices],known_library_smiles=dataset.smiles)
        r.update(metadata,sample=i,predicted_delta=pred[i].tolist())
        r['eligible']=bool(r['chemically_valid'] and r['connected'] and r['protected_copy'] and not r['lead_copy'] and not r['support_match'])
        # Ranking uses the same terminal surrogate and the supplied evidence.
        deficit=np.maximum(np.asarray([.25,-.1])-pred[i],0)
        r['rank_score']=float(-(deficit@deficit)+1e-5*pred[i,0]);rows.append(r)
    return rows

def metrics_for(rows,k=16):
    ranked=sorted((r for r in rows if r['eligible']),key=lambda r:r['rank_score'],reverse=True)
    unique={}
    for r in ranked:
        if r['smiles'] not in unique:unique[r['smiles']]=r
    chosen=list(unique.values())[:k];hits={r['smiles'] for r in rows if r['eligible'] and r['goal_success']}
    return {'attempts':len(rows),'valid':sum(r['chemically_valid'] and r['connected'] for r in rows),'protected':sum(r['protected_copy'] for r in rows),'eligible_distinct':len(unique),'distinct_successes':len(hits),'success_attempts':sum(r['eligible'] and r['goal_success'] for r in rows),'topk':k,'topk_filled':len(chosen),'topk_successes':sum(r['goal_success'] for r in chosen),'topk_success_fraction':sum(r['goal_success'] for r in chosen)/k,'novel_source_distinct_successes':len({r['smiles'] for r in rows if r['eligible_novel_success']}),'first_seen_identity_successes':sum(r['global_unique'] and r['eligible'] and r['goal_success'] for r in rows)}

def run_control(task,device):
    started=time.perf_counter();out=Path(task['output']);out.mkdir(parents=True,exist_ok=True)
    model,p=load_model(task['checkpoint'],device);model.eval().requires_grad_(False)
    critic,cp=load_critic(task['critic'],model,device);frozen=tensor_fingerprint(model.state_dict().items())
    dataset=MaskedEditDataset(split=task.get('split','validation'),kind='vector');truth=ExactMaskedEvaluator();scales=torch.tensor(truth.scales,device=device,dtype=torch.float32)
    conditional=None
    if task.get('conditional_checkpoint'):conditional=load_model(task['conditional_checkpoint'],device)[0].eval().requires_grad_(False)
    all_rows=[];metrics=[];audits=[];calibration=[];nfef=nfeb=physical=0
    attempts=task.get('attempts',64);nfe=task.get('nfe',16);chunk=task.get('chunk',8);supports=task.get('support_sizes',[0]);ledgers={}
    for ci,cid in enumerate(dataset.campaign_ids[:task.get('campaign_limit',len(dataset.campaign_ids))]):
        meta=dataset.manifest['campaigns'][cid];mid=meta['mask_id'];mask=dataset.masks[mid]
        support=dataset.nested_support(ci,seed=77600+ci);lead_raw=torch.tensor(_descriptor(mask.lead_smiles),device=device,dtype=torch.float32)[None]
        base=condition_for(model,dataset,mid)
        # All evidence arms use one cached lead encoding and identical Gaussian noise.
        gaussian=make_paired_gaussian(model.config,attempts,91200+ci,device=device)
        for N in supports:
            if N==0:coeff=truth.coefficients[cid];diag={'basis_rank':3,'condition_number':None}
            else:
                rec=support['records'][N];diag=fit_support_response([g['smiles'] for g in rec['graphs']],rec['values'],mask.lead_smiles,truth.scales);coeff=diag['coefficients']
            calibration.append({'campaign_id':cid,'support_size':N,'basis_rank':diag['basis_rank'],'condition_number':diag['condition_number'],'coefficient_rmse':float(np.sqrt(np.mean((coeff-truth.coefficients[cid])**2)))})
            methods=['unguided_rerank','terminal_control']
            if conditional is not None:methods.append('conditional_flow')
            if task.get('compute_pool_multiplier',1)>1:methods.append('unguided_compute_pool')
            for method in methods:
                for evidence in ('correct','opposite'):
                    armW=np.array(coeff,copy=True)
                    if evidence=='opposite':armW[0]*=-1
                    method_attempts=attempts*int(task.get('compute_pool_multiplier',1)) if method=='unguided_compute_pool' else attempts
                    method_noise=gaussian
                    if method_attempts>attempts:
                        extra=make_paired_gaussian(model.config,method_attempts-attempts,191200+ci,device=device)
                        method_noise=Latents(torch.cat((gaussian.nodes,extra.nodes)),torch.cat((gaussian.pairs,extra.pairs)))
                    key=(method,N,evidence);oracle=ledgers.setdefault(key,ExactMaskedEvaluator());case=[];casef=caseb=0;casestart=time.perf_counter()
                    for first in range(0,method_attempts,chunk):
                        count=min(chunk,method_attempts-first);cond=base.expand(count);noise=Latents(method_noise.nodes[first:first+count],method_noise.pairs[first:first+count]);W=torch.tensor(armW,device=device,dtype=torch.float32)[None].expand(count,-1,-1)
                        m=model
                        if method=='conditional_flow':
                            m=conditional;state=cond.response_state.clone();state[:,:6]=W.reshape(count,6);cond=replace(cond,response_state=state)
                        if method=='terminal_control':
                            control=optimize_terminal_control(m,cond,noise,critic,W,lead_raw.expand(count,-1),scales,cond.goal,nfe=nfe,iterations=task.get('iterations',4),radii=tuple(task.get('radii',[.1,.3,1.])))
                            result=control.sample;casef+=control.forward_nfe*count;caseb+=control.backward_nfe*count
                            audits.append({'campaign_id':cid,'method':method,'support_size':N,'evidence':evidence,'first':first,'predicted_before':control.predicted_before.cpu().tolist(),'predicted_after':control.predicted_after.cpu().tolist(),'control_energy':control.control_energy.cpu().tolist(),'accepted_steps':control.accepted_steps.cpu().tolist(),'gradient_norms':[x.cpu().tolist() for x in control.gradient_norms],'forward_sample_nfe':control.forward_nfe*count,'backward_sample_nfe':control.backward_nfe*count})
                        else:
                            with torch.no_grad():result=sample_masked_edit_latents(m,cond,noise,nfe=nfe)
                            casef+=nfe*count
                        logits,pred=score_prediction(m,critic,result,cond,W,lead_raw.expand(count,-1),scales)
                        rows=truth_rows(graphs_from_logits(logits),pred.cpu().numpy(),oracle,dataset,cid,support,dict(method=method,support_size=N,evidence=evidence,seed=task['seed'],split=dataset.split))
                        for i,r in enumerate(rows):r['sample']=first+i
                        case.extend(rows);physical+=count
                    nfef+=casef;nfeb+=caseb;all_rows.extend(case)
                    metrics.append(dict(campaign_id=cid,mask_id=mid,method=method,support_size=N,evidence=evidence,seed=task['seed'],forward_sample_nfe=casef,backward_sample_nfe=caseb,seconds_including_chemistry=time.perf_counter()-casestart,**metrics_for(case)))
                    write_csv(out/'metrics.csv',metrics);print(json.dumps(metrics[-1]),flush=True)
    write_csv(out/'attempts.csv',all_rows);write_csv(out/'calibration.csv',calibration)
    atomic_json(out/'control_audits.json',{'records':audits})
    if tensor_fingerprint(model.state_dict().items())!=frozen:raise RuntimeError('Frozen generator changed')
    contrasts=[]
    for N in supports:
        for method in sorted({r['method'] for r in metrics}):
            correct=[r for r in metrics if r['method']==method and r['support_size']==N and r['evidence']=='correct'];opp=[r for r in metrics if r['method']==method and r['support_size']==N and r['evidence']=='opposite']
            contrasts.append(dict(method=method,support_size=N,correct_distinct_successes=sum(r['distinct_successes'] for r in correct),opposite_distinct_successes=sum(r['distinct_successes'] for r in opp),correct_topk_successes=sum(r['topk_successes'] for r in correct),opposite_topk_successes=sum(r['topk_successes'] for r in opp),topk_budget=sum(r['topk'] for r in correct),correct_global_unique_successes=len({r['smiles'] for r in all_rows if r['method']==method and r['support_size']==N and r['evidence']=='correct' and r['eligible'] and r['goal_success']}),opposite_global_unique_successes=len({r['smiles'] for r in all_rows if r['method']==method and r['support_size']==N and r['evidence']=='opposite' and r['eligible'] and r['goal_success']})))
    summary={'phase':'complete','stage':task['stage'],'seed':task['seed'],'variant':p['variant'],'split':dataset.split,'contrasts':contrasts,'physical_terminal_proposals':physical,'forward_sample_nfe':nfef,'backward_sample_nfe':nfeb,'seconds_including_chemistry':time.perf_counter()-started,'completed_utc':now(),'task':task,'claim_scope':'Known response basis; support-only ridge calibration; terminal surrogate control; exact simulated truth after commit. Extra controller computation recorded, not assumed equal to unguided proposals.'}
    atomic_json(out/'summary.json',summary);return summary

def execute_experiment(task,device):
    stage=task['stage']
    if stage=='critic':
        from .masked_edit_critic_training import train_critic
        train_critic(task['checkpoint'],task['output'],device=device,seed=task['seed'],steps=task.get('steps',3000),source_attempts=task.get('source_attempts',512))
    elif stage in ('control_validation','control_test','fewshot_validation'):run_control(task,device)
    elif stage=='oracle':
        from .masked_edit_oracle_experiment import run_oracle_experiment
        run_oracle_experiment(task,device)
    elif stage=='baselines':
        from .masked_edit_baselines import evaluate_simple_baselines
        evaluate_simple_baselines(task.get('split','validation'),task['output'],seed=task['seed'],budget=task.get('budget',256))
    elif stage=='revision':
        from .masked_edit_revision import run_revision
        run_revision(task,device)
    else:raise ValueError(stage)
