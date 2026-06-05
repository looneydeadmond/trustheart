from __future__ import annotations

from torch import nn


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def quantized_size_kb(model: nn.Module, bits: int = 8) -> float:
    bytes_per = bits / 8.0
    return parameter_count(model) * bytes_per / 1024.0


def footprint_report(model: nn.Module, ladl_overhead_kb: float = 2.5) -> dict[str, float]:
    params = float(parameter_count(model))
    return {
        "parameters": params,
        "int8_size_kb": quantized_size_kb(model, 8),
        "float32_size_kb": quantized_size_kb(model, 32),
        "ladl_overhead_kb": ladl_overhead_kb,
    }
