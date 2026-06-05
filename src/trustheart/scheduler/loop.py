from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..assembly.config import ExperimentConfig
from ..assembly.model import TrustHeart
from ..governor.allocation import allocate_budget, noise_scales
from ..governor.engine import PrivateStep, per_sample_gradients
from ..governor.sensitivity import clip_thresholds, gradient_sensitivity
from ..instrument.logs import get_logger
from ..instrument.seeding import set_seed
from ..instrument.types import TrainState
from .optim import build_optimizer, build_schedule

logger = get_logger("trustheart.scheduler")
TensorPair = tuple[torch.Tensor, torch.Tensor]


def _build_private_step(
    model: TrustHeart,
    xb: torch.Tensor,
    yb: torch.Tensor,
    loss_fn: nn.Module,
    config: ExperimentConfig,
    total_steps: int,
) -> PrivateStep:
    grads = per_sample_gradients(model, loss_fn, model.logits, xb, yb)
    sensitivities = gradient_sensitivity(grads)
    clip = clip_thresholds(grads, config.privacy.clip_percentile)
    budgets = allocate_budget(sensitivities, config.privacy.total_epsilon, config.privacy.alpha)
    noise = noise_scales(
        budgets, clip, total_steps, config.privacy.delta, config.privacy.coefficient
    )
    return PrivateStep(clip, noise)


def _private_update(
    model: TrustHeart,
    xb: torch.Tensor,
    yb: torch.Tensor,
    loss_fn: nn.Module,
    step: PrivateStep,
    optimizer: torch.optim.Optimizer,
) -> None:
    grads = per_sample_gradients(model, loss_fn, model.logits, xb, yb)
    private = step.privatize(grads)
    for name, param in model.named_parameters():
        param.grad = private[name]
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def train(
    model: TrustHeart,
    train_ds: Dataset[TensorPair],
    config: ExperimentConfig,
    device: torch.device,
) -> TrainState:
    set_seed(config.train.seed)
    model.to(device)
    loader = DataLoader(train_ds, batch_size=config.train.batch_size, shuffle=True, drop_last=False)
    optimizer = build_optimizer(model, config.train)
    schedule = build_schedule(optimizer, config.train.epochs)
    loss_fn = nn.CrossEntropyLoss()

    batches = max(1, len(loader))
    total_steps = (
        config.train.max_steps
        if config.train.max_steps is not None
        else config.train.epochs * batches
    )
    private = config.privacy.enabled and model.config.use_apba

    step_engine: PrivateStep | None = None
    state = TrainState()
    done = False
    for epoch in range(config.train.epochs):
        state.epoch = epoch
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if private and step_engine is None:
                step_engine = _build_private_step(model, xb, yb, loss_fn, config, total_steps)
            if private and step_engine is not None:
                _private_update(model, xb, yb, loss_fn, step_engine, optimizer)
            else:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model.logits(xb), yb)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                batch_loss = float(loss_fn(model.logits(xb), yb).item())
            state.history.append(batch_loss)
            state.step += 1
            if config.train.max_steps is not None and state.step >= config.train.max_steps:
                done = True
                break
        schedule.step()
        if done:
            break
    logger.info(
        "training finished step=%d final_loss=%.6f",
        state.step,
        state.history[-1] if state.history else float("nan"),
    )
    return state
