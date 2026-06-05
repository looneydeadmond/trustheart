# Repository Plan

## Directory tree

```
trustheart/
├── README.md                         consumer-electronics owner's manual
├── LICENSE                           MIT
├── pyproject.toml                    build + tooling (ruff/black/isort/mypy/pytest)
├── requirements.txt
├── environment.yml
├── Dockerfile                        CUDA 11.8 / torch 2.0 base
├── Makefile                          install/lint/type/test/smoke/docker
├── .gitignore
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml          ruff + mypy + pytest on smoke
├── configs/
│   ├── model/{trustheart,encoder_small}.yaml
│   ├── data/{mitbih,cpsc2018,synthetic}.yaml
│   ├── train/{main,dp_uniform}.yaml
│   └── experiment/
│       ├── main.yaml                 main_cpsc.yaml
│       ├── ablation_components_*.yaml
│       ├── ablation_alpha_*.yaml
│       ├── ablation_ladl_*.yaml
│       ├── supplementary_*.yaml
│       └── _smoke.yaml
├── src/trustheart/
│   ├── afe/            conditioning.py rpeak.py segment.py        (SPM)
│   ├── acquisition/    sources.py synthetic.py beats.py splits.py (datasets)
│   ├── fabric/         backbone.py blocks.py                      (encoder)
│   ├── governor/       sensitivity.py allocation.py accountant.py engine.py (APBA/DP)
│   ├── salience/       attention.py perturbation.py saliency.py   (PAAM)
│   ├── watchdog/       mixture.py detector.py                     (LADL)
│   ├── decoder/        head.py                                    (classifier)
│   ├── assembly/       model.py config.py                         (full TrustHeart)
│   ├── scheduler/      loop.py optim.py checkpoint.py             (training)
│   ├── bench/          classification.py robustness.py detection.py
│   │                   interpretability.py footprint.py attacks.py
│   │                   intervals.py significance.py               (evaluation)
│   ├── instrument/     seeding.py logs.py io.py device.py types.py
│   └── console/        run.py                                     (CLI)
├── scripts/            launch_train.sh launch_eval.sh prepare_data.sh
├── tests/
└── docs/               project-context.md implementation-map.md
                        repo-plan.md deviations.md
```

## Module responsibilities

- `afe` — deterministic signal conditioning and beat framing (no learnable params).
- `acquisition` — `EcgSource` Protocol; `WfdbSource` (real MIT-BIH/CPSC via `wfdb`)
  and `SyntheticEcg` (parametric PQRST generator); `BeatDataset`; De Chazal and
  k-fold splits.
- `fabric` — Conv1d encoder producing a 128-d embedding.
- `governor` — gradient-sensitivity estimation, inverse-sensitivity budget split,
  per-layer noise calibration, the private SGD step, and an RDP accountant.
- `salience` — multi-head attention with Laplacian DP perturbation and saliency.
- `watchdog` — per-class diagonal GMM and the Mahalanobis anomaly score with a
  percentile threshold.
- `decoder` — two-layer classification head.
- `assembly` — frozen dataclass config + the composed `TrustHeart` module with
  `use_apba/use_paam/use_ladl` toggles.
- `scheduler` — seedable training loop, Adam + cosine schedule, atomic checkpoints.
- `bench` — every reported metric, attack, statistic, and the deployment footprint
  estimator.
- `instrument` — seeding, logging, atomic IO, device selection, shared typed records.
- `console` — a `jsonargparse` subcommand CLI (train/evaluate/detect/explain/export).

## Dependencies (pinned)

torch==2.0.1, numpy>=1.24,<2.0, scipy>=1.10, scikit-learn>=1.3, wfdb>=4.1,
jsonargparse>=4.27, PyYAML>=6.0, onnx>=1.15 (export extra). Dev: ruff, black, isort,
mypy, pytest, hypothesis.

## Test coverage targets

shape, physics-invariant (passband response, attention rows sum to 1, GMM densities
non-negative), single-batch overfit, metric-vs-sklearn, gradient-flow, DP numerics
(noise monotone in sensitivity, RDP→ε≈4.2), numerical regression, AST+tokenize
no-comment/no-docstring guard, forbidden-phrase/emoji guard, end-to-end 2-step smoke.
