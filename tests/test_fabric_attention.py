from __future__ import annotations

import torch

from trustheart.fabric.backbone import ConvFabric
from trustheart.salience.attention import PrivacyAwareAttention
from trustheart.salience.saliency import saliency_map, upsample_to


def test_fabric_embedding_shape() -> None:
    fabric = ConvFabric()
    x = torch.randn(4, 1, 256)
    assert fabric(x).shape == (4, 128)
    assert fabric.feature_map(x).shape[1] == 128


def test_attention_rows_sum_to_one() -> None:
    attn = PrivacyAwareAttention(dim=128, heads=4)
    feature_map = torch.randn(3, 128, 16)
    pooled, weights = attn(feature_map)
    assert pooled.shape == (3, 128)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_perturbation_changes_weights() -> None:
    torch.manual_seed(0)
    attn = PrivacyAwareAttention(dim=128, heads=4, lambda_att=0.5)
    feature_map = torch.randn(2, 128, 16)
    _, clean = attn(feature_map, perturb=False)
    _, noisy = attn(feature_map, perturb=True)
    assert not torch.allclose(clean, noisy)


def test_saliency_normalised_range() -> None:
    attention = torch.softmax(torch.randn(2, 16, 16), dim=-1)
    relevance = torch.rand(2, 16)
    pooled = saliency_map(attention, relevance)
    mapped = upsample_to(pooled, 256)
    assert mapped.shape == (2, 256)
    assert float(mapped.min()) >= 0.0
    assert float(mapped.max()) <= 1.0 + 1e-6
