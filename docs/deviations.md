# Deviations

Intentional departures from a literal transcription of the manuscript. Each item
cites the paper location it relates to.

## D1 — Synthetic ECG for offline tests
§IV.A, Table I. The reported results use MIT-BIH and CPSC2018, which require a
PhysioNet / ICBEB download. `acquisition/sources.py` provides a faithful `wfdb`
reader for the real recordings; `acquisition/synthetic.py` provides a parametric
PQRST generator with class-conditioned morphology, so the test suite, the smoke
config, and a no-network demo run without external data. Reported numbers require
the real datasets.

## D2 — Hand-written per-layer DP-SGD instead of Opacus
§III.D, Eq 7–9, Algorithm 1. The manuscript reports Opacus 1.4, but APBA assigns a
distinct per-layer noise scale σ^(k) from inverse gradient sensitivity, which the
standard Opacus `PrivacyEngine` does not express. The private step, per-layer noise,
and the Rényi-DP accountant are implemented directly on torch tensors (R1: implement
manually when a library cannot express a paper detail natively).

## D3 — Microcontroller deployment reproduced as ONNX + documented quantization
§IV.B, §IV.G, Table XIII. STM32CubeAI and TensorFlow Lite Micro on-device execution
cannot run in this environment. `console/run.py export` writes an ONNX graph;
`bench/footprint.py` estimates parameter count, int8 size, and the reported 295 KB /
35 ms / 2.3 mW envelope. On-device latency and power are quoted from the paper, not
measured here.

## D4 — Training budget in CI
§IV.B, Table XVI. The paper trains 100 epochs at batch 256 on RTX 3090. The smoke
configuration (`configs/experiment/_smoke.yaml`) runs 2 optimizer steps on synthetic
data for unit testing only; it is labelled accordingly and must not be used for
reporting.

## D5 — SAP smoothing approximation
§IV.B, Table XVII. Smooth Adversarial Perturbation is implemented as PGD with a
Gaussian-smoothed gradient (σ=3) matching the description; the exact reference
implementation details beyond the stated σ, ε, step, and iteration count are not
specified in the manuscript.
