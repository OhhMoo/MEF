"""Check set invariance and normalized joint-response mathematics."""

import inspect

import torch

from emf.response import EvidenceEncoder, ResponseModel


def test_support_order_and_invalid_records_cannot_change_evidence():
    torch.manual_seed(5)
    encoder = EvidenceEncoder(7, 3, state_dim=4, hidden_dim=8)
    features, values = torch.randn(2, 4, 7), torch.randn(2, 4, 2)
    context = torch.randn(2, 3)
    mask = torch.tensor([[True, True, True, False], [True, False, False, False]])
    expected = encoder(features, values, context, mask)
    permutation = torch.tensor([2, 0, 3, 1])
    actual = encoder(features[:, permutation], values[:, permutation], context, mask[:, permutation])
    torch.testing.assert_close(actual, expected)
    dirty_features = torch.where(mask[..., None], features, torch.nan)
    dirty_values = torch.where(mask[..., None], values, torch.nan)
    torch.testing.assert_close(encoder(dirty_features, dirty_values, context, mask), expected)
    assert 'goal' not in inspect.signature(encoder.forward).parameters


def test_joint_response_matches_normalized_gaussian_density_and_positive_floor():
    torch.manual_seed(8)
    response = ResponseModel(7, 3, state_dim=4, hidden_dim=8)
    arguments = (torch.randn(2, 7), torch.randn(2, 2), torch.randn(2, 7),
                 torch.randn(2, 3), torch.randn(2, 4))
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        distribution = response(*arguments)
    assert distribution.mean.dtype == distribution.covariance_matrix.dtype == torch.float32
    eigenvalues = torch.linalg.eigvalsh(distribution.covariance_matrix)
    assert eigenvalues.min() >= 1e-4
    delta = torch.randn(2, 2)
    centered = delta - distribution.mean
    quadratic = (centered[..., None, :] @ torch.linalg.solve(
        distribution.covariance_matrix, centered[..., None])).flatten()
    _, logdet = torch.linalg.slogdet(distribution.covariance_matrix)
    expected = -.5 * (2 * torch.log(torch.tensor(2 * torch.pi)) + logdet + quadratic)
    torch.testing.assert_close(distribution.log_prob(delta), expected)
    assert 'goal' not in inspect.signature(response.forward).parameters
    (-distribution.log_prob(delta).mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in response.parameters())


def test_empty_evidence_is_a_finite_global_prior():
    encoder = EvidenceEncoder(7, 3, state_dim=4, hidden_dim=8)
    actual = encoder(torch.empty(2, 0, 7), torch.empty(2, 0, 2), torch.randn(2, 3),
                     torch.empty(2, 0, dtype=torch.bool))
    assert actual.shape == (2, 4) and torch.isfinite(actual).all()
