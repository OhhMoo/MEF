"""Support-only task-state and selector adaptation controls.

All labels come from already observed compounds. Pair construction increases
training examples, not independent observations. Source parameters are never
updated: task adaptation optimizes only b, and selector adaptation returns an
independent copy with only its selector updated before deployment is frozen.
"""

import copy
import hashlib
import math
from time import perf_counter

import numpy as np
import torch

from emf.campaigns import molecular_fingerprint, neighbour_pairs
from emf.chem import decode_graph
from emf.contracts import GraphBatch, GraphMasks, Latents
from emf.objectives import interpolate_latents
from emf.redesign_contracts import FlowCondition, SupportBatch
from emf.redesign_objectives import goal_satisfied, weighted_velocity_matching_loss, _draw, _noise
from emf.sampling import model_device


def _digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(str((name, str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _take(graphs, indices):
    index = torch.as_tensor(indices, device=graphs.atoms.device, dtype=torch.long)
    names = index.cpu().tolist()
    return GraphBatch(graphs.atoms[index], graphs.bonds[index],
                      GraphMasks(graphs.masks.nodes[index], graphs.masks.pairs[index]),
                      tuple(graphs.smiles[i] for i in names) if graphs.smiles else (),
                      tuple(graphs.ids[i] for i in names) if graphs.ids else ())


def _prepare(model, support: SupportBatch, graphs: GraphBatch, max_pairs):
    """Align cached evidence and graphs; select pairs without endpoint filtering."""
    if support.features.shape[0] != 1 or max_pairs < 1:
        raise ValueError('adaptation requires one support campaign and max_pairs>=1')
    device = model_device(model)
    observed = support.mask[0].nonzero().flatten()
    slots, count = support.mask.shape[1], len(observed)
    if len(graphs.atoms) == slots:
        graphs = _take(graphs, observed)
    elif len(graphs.atoms) != count:
        raise ValueError('support graph rows must match support slots or observed records')
    if graphs.smiles:
        smiles = list(graphs.smiles)
    else:
        smiles = []
        for atoms, bonds, n in zip(graphs.atoms.cpu(), graphs.bonds.cpu(), graphs.counts.cpu()):
            result = decode_graph(atoms.numpy(), bonds.numpy(), int(n), model.config.max_nodes)
            if not result.chemically_valid or not result.connected:
                raise ValueError('adaptation requires valid observed support graphs')
            smiles.append(result.smiles)
    if len(set(smiles)) != count:
        raise ValueError('support compounds must have distinct standardized identities')
    order = sorted(range(count), key=lambda i: smiles[i])
    graphs = _take(graphs, order).to(device)
    features = support.features[0, observed][order].detach().to(device)
    values = support.values[0, observed][order].detach().float().to(device)
    smiles = [smiles[i] for i in order]
    fingerprints = [molecular_fingerprint(smiles_i) for smiles_i in smiles]
    counts = graphs.counts.cpu().numpy()
    neighbours, similarities = neighbour_pairs(fingerprints, counts, max_neighbours=4)
    strict_count = int((neighbours >= 0).sum())
    pairs = [(lead, int(neighbours[lead, rank]), float(similarities[lead, rank]))
             for rank in range(4) for lead in range(count) if neighbours[lead, rank] >= 0]
    fallback = False
    if not pairs and count >= 2:
        closest, scores = neighbour_pairs(fingerprints, counts, min_similarity=0., max_neighbours=1)
        candidates = [(lead, int(closest[lead, 0]), float(scores[lead, 0]))
                      for lead in range(count) if closest[lead, 0] >= 0]
        if candidates:
            lead, analogue, similarity = min(candidates, key=lambda pair: (-pair[2], pair[0], pair[1]))
            pairs = [(lead, analogue, similarity), (analogue, lead, similarity)]
            fallback = True
    pairs = pairs[:max_pairs]
    indices = torch.tensor([[lead, analogue] for lead, analogue, _ in pairs], dtype=torch.long,
                           device=device).reshape(-1, 2)
    pair_records = [{"lead": smiles[lead], "analogue": smiles[analogue], "similarity": similarity,
                     "count_difference": abs(int(counts[lead]) - int(counts[analogue]))}
                    for lead, analogue, similarity in pairs]
    pair_metadata = {"unique_observations": count, "known_endpoint_measurements": 2 * count,
                     "additional_oracle_calls": 0, "cached_support_features": True,
                     "strict_candidate_pairs": strict_count, "pairs_selected": len(pairs),
                     "skipped": not bool(pairs),
                     "skip_reason": 'no count-compatible observed pairs' if not pairs else None,
                     "pair_leads": len(set(indices[:, 0].cpu().tolist())),
                     "max_pairs": int(max_pairs), "max_neighbours_per_lead": 4,
                     "minimum_similarity": .4, "max_count_difference": 2,
                     "similarity_fallback": fallback,
                     "fallback_rule": 'global closest count-compatible pair, both directions' if fallback else None,
                     "pair_similarity_min": min((p[2] for p in pairs), default=None),
                     "pair_similarity_mean": float(np.mean([p[2] for p in pairs])) if pairs else None,
                     "pair_similarity_max": max((p[2] for p in pairs), default=None),
                     "pairs": pair_records, "observed_smiles": smiles}
    return features, values, graphs, indices, support.context.detach().float().to(device), pair_metadata


def _settings(steps, learning_rate):
    if steps < 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('steps must be nonnegative and learning_rate finite and positive')


def adapt_task_state(model, support: SupportBatch, support_graphs: GraphBatch,
                     steps=64, learning_rate=.02, max_pairs=128):
    """Fit b with frozen p and pair changes; requested goals never enter this fit."""
    _settings(steps, learning_rate)
    device = model_device(model)
    _synchronize(device)
    started, before = perf_counter(), _digest(model)
    features, values, _, pairs, context, pair_metadata = _prepare(model, support, support_graphs, max_pairs)
    with torch.no_grad():
        initial = model.support_state(support.to(device)).detach().clone()
    state = initial.clone().requires_grad_()
    optimizer = torch.optim.Adam([state], lr=learning_rate)
    trajectory = []
    if len(pairs):
        lead, analogue = pairs[:, 0], pairs[:, 1]
        baseline, delta = values[lead], values[analogue] - values[lead]

        def objective():
            distribution = model.predict_response_features(features[lead], features[analogue], baseline,
                                                            context.expand(len(pairs), -1), state.expand(len(pairs), -1))
            nll = -distribution.log_prob(delta).mean()
            regularization = .01 * (state - initial).square().mean()
            return nll + regularization, nll, regularization

        with torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
            initial_loss, initial_nll, _ = objective()
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                loss, nll, penalty = objective()
                state.grad, = torch.autograd.grad(loss, state)
                if not torch.isfinite(loss) or not torch.isfinite(state.grad).all():
                    raise FloatingPointError('nonfinite task-state adaptation')
                optimizer.step()
                trajectory.append({"step": step + 1, "loss": float(loss.detach()),
                                   "nll": float(nll.detach()), "regularization": float(penalty.detach())})
            final_loss, final_nll, _ = objective()
    else:
        initial_loss = initial_nll = final_loss = final_nll = None
    _synchronize(device)
    after = _digest(model)
    metadata = {"method": 'task_state', **pair_metadata, "requested_steps": int(steps),
                "steps": len(trajectory), "learning_rate": float(learning_rate),
                "trainable_parameters": state.numel(), "trained_source_parameters": 0,
                "state_regularization": .01, "goal_free": True, "precision": 'FP32',
                "covariance_floor": model.redesign_config.covariance_floor,
                "initial_loss": float(initial_loss.detach()) if initial_loss is not None else None,
                "final_loss": float(final_loss.detach()) if final_loss is not None else None,
                "initial_nll": float(initial_nll.detach()) if initial_nll is not None else None,
                "final_nll": float(final_nll.detach()) if final_nll is not None else None,
                "state_drift_l2": float((state.detach() - initial).norm()), "trajectory": trajectory,
                "source_state_sha256_before": before, "source_state_sha256_after": after,
                "source_weights_unchanged": before == after, "seconds": perf_counter() - started}
    return state.detach(), metadata


def adapt_selector(model, support: SupportBatch, support_graphs: GraphBatch,
                   steps=64, learning_rate=3e-4, max_pairs=128, seed=1729):
    """Adapt only a copied factorized selector using independent support-pair goals."""
    _settings(steps, learning_rate)
    if not hasattr(model.flow, 'selector'):
        raise ValueError('selector adaptation requires the factorized source model')
    device = model_device(model)
    _synchronize(device)
    started, before = perf_counter(), _digest(model)
    features, values, graphs, pairs, context, pair_metadata = _prepare(model, support, support_graphs, max_pairs)
    adapted = copy.deepcopy(model)
    adapted.freeze_deployment()
    adapted._deployment_frozen = False
    adapted.flow.selector.requires_grad_(True)
    adapted.flow.selector.train()
    with torch.no_grad():
        state = adapted.support_state(support.to(device)).detach()
        clean_support = adapted.encode(graphs).detach()
    parameters = list(adapted.flow.selector.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    goal_uniform_digest = hashlib.sha256()
    trajectory = []
    if len(pairs):
        lead, analogue = pairs[:, 0], pairs[:, 1]
        masks = GraphMasks(graphs.masks.nodes[analogue], graphs.masks.pairs[analogue])
        clean = Latents(clean_support.nodes[analogue], clean_support.pairs[analogue])
        baseline, delta = values[lead], values[analogue] - values[lead]
        with torch.enable_grad():
            for step in range(steps):
                uniform = _draw((len(pairs), 2), device, generator, normal=False)
                goal_uniform_digest.update(uniform.detach().cpu().numpy().tobytes())
                goal = torch.stack([2 * uniform[:, 0] - 1, baseline[:, 1] - uniform[:, 1]], -1)
                weights = goal_satisfied(delta, baseline, goal)
                time = .95 * _draw((len(pairs),), device, generator, normal=False)
                noisy = interpolate_latents(clean, _noise(clean, masks, generator), time, masks)
                condition = FlowCondition(features[lead], baseline, context.expand(len(pairs), -1),
                                          state.expand(len(pairs), -1), goal)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    prediction = adapted.predict_clean(noisy, time, masks, condition)
                    loss = weighted_velocity_matching_loss(prediction, clean, time, masks, weights)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
                if not torch.isfinite(loss) or not torch.isfinite(norm):
                    raise FloatingPointError('nonfinite selector adaptation')
                optimizer.step()
                trajectory.append({"step": step + 1, "flow_loss": float(loss.detach()),
                                   "accepted_fraction": float(weights.float().mean()), "gradient_norm": float(norm)})
    names = [name for name, parameter in adapted.named_parameters() if parameter.requires_grad]
    adapted.freeze_deployment()
    _synchronize(device)
    after = _digest(model)
    metadata = {"method": 'selector', **pair_metadata, "requested_steps": int(steps),
                "steps": len(trajectory), "learning_rate": float(learning_rate), "seed": int(seed),
                "trainable_parameters": sum(parameter.numel() for parameter in parameters),
                "trainable_names": names, "trajectory": trajectory,
                "goal_prior": 'independent kappa Uniform(-1,1), tau Uniform(y02-1,y02)',
                "goal_uniform_sha256": goal_uniform_digest.hexdigest(),
                "flow_denominator": 'fixed selected-pair batch size', "T": .95,
                "precision": 'CUDA bf16 trunk / FP32 heads and loss' if device.type == 'cuda' else 'CPU FP32',
                "covariance_floor": model.redesign_config.covariance_floor,
                "accepted_fraction_mean": float(np.mean([row['accepted_fraction'] for row in trajectory])) if trajectory else None,
                "source_state_sha256_before": before, "source_state_sha256_after": after,
                "source_weights_unchanged": before == after,
                "adapted_state_sha256": _digest(adapted), "seconds": perf_counter() - started}
    return adapted, state, metadata
