from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    in_channels: int = 1
    channels: tuple[int, ...] = (32, 64, 128, 128)
    kernel: int = 7
    n_classes: int = 5
    attention_heads: int = 4
    lambda_att: float = 0.0
    ladl_components: int = 2
    ladl_beta: float = 0.5
    use_apba: bool = True
    use_paam: bool = True
    use_ladl: bool = True


@dataclass
class DataConfig:
    source: str = "synthetic"
    root: str = "data"
    n_classes: int = 5
    per_class: int = 64
    sampling_rate: float = 360.0
    train_fraction: float = 0.7
    seed: int = 7


@dataclass
class PrivacyConfig:
    enabled: bool = True
    total_epsilon: float = 4.2
    delta: float = 1e-5
    alpha: float = 0.5
    warmup_epochs: int = 5
    clip_percentile: float = 90.0
    coefficient: float = 1.0


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_steps: int | None = None
    seed: int = 7
    device: str = "auto"
    out_dir: str = "runs/main"


@dataclass
class ExperimentConfig:
    name: str = "main"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
