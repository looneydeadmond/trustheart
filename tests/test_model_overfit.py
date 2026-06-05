from __future__ import annotations

import torch
from torch import nn

from trustheart.assembly.config import ModelConfig
from trustheart.assembly.model import TrustHeart


def test_forward_outputs_shapes() -> None:
    model = TrustHeart(ModelConfig(n_classes=5))
    out = model(torch.randn(4, 1, 256))
    assert out.logits.shape == (4, 5)
    assert out.embedding.shape == (4, 128)
    assert out.attention is not None
    assert out.attention.shape == (4, 16, 16)


def test_gradients_reach_all_parameters() -> None:
    model = TrustHeart(ModelConfig(n_classes=3))
    logits = model.logits(torch.randn(8, 1, 256))
    loss = nn.CrossEntropyLoss()(logits, torch.randint(0, 3, (8,)))
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name


def test_single_batch_overfit() -> None:
    torch.manual_seed(0)
    model = TrustHeart(ModelConfig(n_classes=4))
    x = torch.randn(16, 1, 256)
    y = torch.randint(0, 4, (16,))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    first = float(loss_fn(model.logits(x), y).item())
    for _ in range(120):
        optimizer.zero_grad()
        loss = loss_fn(model.logits(x), y)
        loss.backward()
        optimizer.step()
    final = float(loss_fn(model.logits(x), y).item())
    assert final < 0.25 * first
