"""Independent CUDA smoke review; invoked within coordinator-owned allocation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from emf.contracts import GraphMasks, Latents, ModelConfig
from emf.data import collate_graphs
from emf.masked_edit_data import MaskedEditDataset
from emf.masked_edit_control import SoftGraphDescriptorCritic, optimize_terminal_control
from emf.masked_edit_model import MaskedEditModel, decode_masked_edit
from emf.masked_edit_objectives import masked_edit_loss
from emf.masked_edit_sampling import make_paired_gaussian, sample_masked_edit_latents
from emf.models import EMFModel
from emf.redesign_contracts import CampaignBatch, RedesignConfig, SupportBatch


def run(output):
    if not torch.cuda.is_available():
        raise RuntimeError('Run this check inside an allocated CUDA job')
    started = perf_counter()
    torch.set_num_threads(2)
    device = torch.device('cuda')
    source = MaskedEditDataset(split='train')
    rows = source.sample_pairs(np.random.default_rng(171), 2)
    lead = collate_graphs([row['lead_graph'] for row in rows]).to(device)
    query = collate_graphs([row['query_graph'] for row in rows]).to(device)
    edit = GraphMasks(torch.as_tensor(np.stack([r['atom_editable'] for r in rows]), device=device),
                      torch.as_tensor(np.stack([r['bond_editable'] for r in rows]), device=device))
    config = RedesignConfig(ModelConfig(max_nodes=9, node_dim=8, pair_dim=4,
        hidden_dim=32, depth=1, num_heads=4), state_dim=6, context_dim=4)
    support = SupportBatch(torch.randn(1, 4, config.feature_dim, device=device),
        torch.zeros(1, 4, 2, device=device), torch.zeros(1, 4, device=device),
        torch.ones(1, 4, dtype=torch.bool, device=device))
    zeros = torch.zeros(2, 2, device=device)
    batch = CampaignBatch(query, lead, support, torch.zeros(2, dtype=torch.long, device=device),
        zeros, torch.zeros(2, 4, device=device), zeros, zeros, torch.ones(2, device=device))
    reports = {}
    for variant in ('A', 'B', 'C'):
        torch.manual_seed(171)
        model = MaskedEditModel(config, EMFModel(config.model), variant=variant,
                                task_conditioned=False).to(device)
        frozen = {name: tensor.detach().clone() for name, tensor in model.encoder.state_dict().items()}
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
        loss, _ = masked_edit_loss(model, batch, edit, accept_all=True,
            generator=torch.Generator(device=device).manual_seed(11))
        loss.backward()
        assert torch.isfinite(loss)
        assert model.decoder.node_head[-1].weight.grad is not None
        assert torch.isfinite(model.decoder.node_head[-1].weight.grad).all()
        optimizer.step()
        assert all(torch.equal(tensor, model.encoder.state_dict()[name]) for name, tensor in frozen.items())
        model.freeze_deployment()
        before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        condition = model.condition(lead, zeros, batch.context, torch.zeros(2, 6, device=device), zeros, edit)
        gaussian = make_paired_gaussian(config.model, 2, 19, device=device)
        gain = torch.tensor(.1, device=device, requires_grad=True)
        def control(state, time, supplied):
            return Latents(torch.ones_like(state.nodes)*gain, torch.ones_like(state.pairs)*gain)
        result = sample_masked_edit_latents(model, condition, gaussian, nfe=4,
            velocity_controller=control, return_trace=True)
        value = result.latents.nodes.square().mean()+result.latents.pairs.square().mean()
        derivative = torch.autograd.grad(value, gain)[0]
        assert derivative.isfinite() and derivative.abs() > 0
        logits = decode_masked_edit(model, result.latents, condition)
        if variant != 'A':
            for state in (*result.states, result.latents):
                assert torch.equal(state.nodes[~edit.nodes], condition.lead.nodes[~edit.nodes])
                assert torch.equal(state.pairs[~edit.pairs], condition.lead.pairs[~edit.pairs])
            assert torch.equal(logits.atoms.argmax(-1)[~edit.nodes], lead.atoms[~edit.nodes])
            assert torch.equal(logits.bonds.argmax(-1)[~edit.pairs], lead.bonds[~edit.pairs])
        assert all(torch.equal(tensor, model.state_dict()[name]) for name, tensor in before.items())
        critic = SoftGraphDescriptorCritic(config.model, hidden_dim=16, depth=1).to(device)
        optimized = optimize_terminal_control(model, condition, gaussian, critic,
            torch.tensor([[[1., 0., 0.], [0., 1., 0.]]], device=device).expand(2, -1, -1),
            torch.zeros(2, 2, device=device), torch.ones(2, device=device),
            torch.full((2, 2), 1., device=device), nfe=3, iterations=1, radii=(.02,))
        assert optimized.forward_nfe == 6 and optimized.backward_nfe == 6
        assert torch.isfinite(optimized.predicted_after).all()
        assert all(torch.isfinite(g).all() for g in optimized.gradient_norms)
        assert all(torch.equal(tensor, model.state_dict()[name]) for name, tensor in before.items())
        reports[variant] = {'finite_training_loss': float(loss.detach()),
            'terminal_control_gradient': float(derivative.detach()), 'nfe': result.nfe,
            'frozen_encoder_and_deployment_state': True, 'protected_checks': variant != 'A',
            'optimizer_forward_nfe':optimized.forward_nfe,
            'optimizer_backward_nfe':optimized.backward_nfe,
            'optimizer_gradient_norms':[g.cpu().tolist() for g in optimized.gradient_norms]}
    torch.cuda.synchronize()
    report = {'status':'passed', 'reviewed_utc':datetime.now(timezone.utc).isoformat(),
        'device':torch.cuda.get_device_name(), 'variants':reports,
        'elapsed_seconds':perf_counter()-started,
        'scope':'Allocated CUDA integration smoke; random representation and one update establish interfaces only.'}
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status':'passed','report':str(path)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='5_test/results/masked_edit_gpu_operation.json')
    run(parser.parse_args().output)
