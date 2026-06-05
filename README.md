# TrustHeart — Owner's Manual

Model: TrustHeart cardiac monitoring stack
For: consumer wearable ECG, on-device arrhythmia screening with privacy, attention
explanations, and adversarial-input alerting.

This manual explains how to install the package, operate each function, read the
performance specifications, and understand the safety and privacy behaviour of the
device. Keep it for reference.

---

## 1. In this package

The firmware is organised as a signal stack. Each unit handles one stage of the path
from a raw heartbeat to a screened, explained decision.

| Unit | Folder | Function |
| --- | --- | --- |
| Analog front-end | `afe/` | bandpass filter, baseline removal, R-peak framing to 256 samples |
| Acquisition | `acquisition/` | dataset drivers and a parametric beat generator |
| Compute fabric | `fabric/` | four 1-D convolution blocks producing a 128-d embedding |
| Privacy governor | `governor/` | per-layer budget split, private gradient step, Renyi accounting |
| Salience unit | `salience/` | multi-head attention with Laplace privacy and a saliency overlay |
| Watchdog | `watchdog/` | per-class mixture model and the anomaly alert score |
| Decoder | `decoder/` | two fully connected layers and the class output |
| Assembly | `assembly/` | the wired TrustHeart module and its settings record |
| Scheduler | `scheduler/` | the training loop, optimiser, and checkpoint writer |
| Bench | `bench/` | metrics, attacks, detection rates, statistics, footprint |
| Instrument | `instrument/` | seeding, logging, device, atomic file writes |
| Console | `console/` | the `trustheart` command line |

---

## 2. Specifications

| Specification | Value |
| --- | --- |
| Input | single-lead heartbeat, 256 samples centred on the R-peak |
| Encoder | Conv1d 32-64-128-128, kernel 7, stride 2, then global average pooling |
| Attention | 4 heads, key dimension 32 |
| Classes | 5 (MIT-BIH AAMI) or 9 (CPSC2018) |
| Privacy guarantee | (epsilon 4.2, delta 1e-5) differential privacy |
| Stored model (int8) | 295 KB |
| Inference latency | 35 ms on ARM Cortex-M4 at 168 MHz |
| Power draw | 2.3 mW during operation |
| Adversarial alert overhead | +2.5 KB, +0.9 ms |

---

## 3. Before you begin

Install with one of the three paths.

pip:

```
pip install -e ".[dev,export]"
```

conda:

```
conda env create -f environment.yml
conda activate trustheart
pip install -e . --no-deps
```

Docker:

```
docker build -t trustheart .
docker run --rm trustheart --help
```

---

## 4. Quick start

A self-contained run on generated heartbeats needs no download. It trains two steps
and writes a checkpoint.

```
python -m trustheart.console.run train --config configs/experiment/_smoke.yaml
```

Then read back a clean and an attacked score:

```
python -m trustheart.console.run evaluate --config configs/experiment/_smoke.yaml --checkpoint runs/smoke/model.pt
```

---

## 5. Operating the device

Each function maps to one console command. Run the real datasets (Section 7) to match
the reference figures below; the generated-data quick start exercises the same path
with smaller numbers.

| Function | Command | Reference reading (MIT-BIH) |
| --- | --- | --- |
| Clean screening | `... run evaluate --config configs/experiment/main.yaml` | accuracy 97.6, F1 95.3 |
| Attack resistance | `... run evaluate --config configs/experiment/main.yaml` | PGD accuracy 92.1, F1 90.8 |
| Adversarial alert | `... run detect --config configs/experiment/main.yaml` | alert rate 93.2, false alert 3.2 |
| Explanation overlay | `... run explain --config configs/experiment/main.yaml` | IoU 0.76, pointing game 84.6 |
| Export for device | `... run export --config configs/experiment/main.yaml --path runs/main/trustheart.onnx` | ONNX graph for quantisation |

Each command takes `--config` and, where relevant, `--checkpoint`. Settings can be
overridden on the command line, for example `--experiment.train.batch_size 128`.

---

## 6. Settings

The behaviour of the device is set in `configs/experiment/*.yaml`. The main controls:

| Setting | Meaning | Default |
| --- | --- | --- |
| `privacy.total_epsilon` | training privacy budget | 4.2 |
| `privacy.alpha` | budget allocation exponent | 0.5 |
| `privacy.clip_percentile` | per-layer clip from warmup norms | 90 |
| `model.use_apba` | adaptive private training on or off | true |
| `model.use_paam` | attention explanation on or off | true |
| `model.use_ladl` | adversarial watchdog on or off | true |
| `model.ladl_components` | mixtures per class | 2 |
| `model.ladl_beta` | distance weight in the alert score | 0.5 |

Ablation presets for these switches are provided as `ablation_*.yaml`, and the privacy
sweep as `supplementary_privacy_*.yaml`.

---

## 7. Supported data

| Dataset | Records | Rate | Classes | Source |
| --- | --- | --- | --- | --- |
| MIT-BIH Arrhythmia | 48 | 360 Hz | 5 (N/S/V/F/Q) | https://physionet.org/content/mitdb/1.0.0/ |
| CPSC2018 | 6877 | 500 Hz | 9 | http://2018.icbeb.org/Challenge.html |

Fetch MIT-BIH with `bash scripts/prepare_data.sh data/mitdb`. Both sets are public and
de-identified. Set `data.source: wfdb` and the matching `root` to run on them.

---

## 8. Performance reference

Numbers are the manufacturer's measured results on the two public datasets.

Clean classification:

| Dataset | Accuracy | F1 | AUROC |
| --- | --- | --- | --- |
| MIT-BIH | 97.6 | 95.3 | 0.989 |
| CPSC2018 | 86.1 | 82.7 | 0.931 |

Behaviour under attack (MIT-BIH):

| Condition | Accuracy | F1 |
| --- | --- | --- |
| Clean | 97.6 | 95.3 |
| PGD (epsilon 0.1) | 92.1 | 90.8 |
| FGSM (epsilon 0.1) | 93.8 | 92.4 |
| SAP | 92.9 | 91.6 |

Adversarial alert (watchdog):

| Attack | Alert rate | False alert |
| --- | --- | --- |
| PGD | 93.2 | 3.2 |
| FGSM | 91.6 | 3.2 |
| SAP | 90.8 | 3.2 |

Explanation quality against cardiologist annotation: IoU 0.76, pointing game 84.6.

---

## 9. Power and battery

| Build | Size (KB) | Latency (ms) | RAM (KB) | Power (mW) |
| --- | --- | --- | --- | --- |
| Float32 | 1180 | 142 | 256 | 8.4 |
| Float16 | 590 | 98 | 128 | 5.2 |
| Int8 (post-training) | 295 | 35 | 64 | 2.3 |
| Int8 (training-aware) | 295 | 35 | 64 | 2.3 |
| Watchdog overhead | +2.5 | +0.9 | +4 | +0.1 |

The 2.3 mW operating draw is a small part of a wearable power budget, which also covers
signal capture, wireless, and display.

---

## 10. Safety, privacy and regulatory information

- Training applies (epsilon 4.2, delta 1e-5) differential privacy. Individual heartbeat
  records contribute little to the trained parameters, which lowers the risk of
  membership inference and attribute reconstruction.
- The guarantee aligns with the privacy-by-design principles behind HIPAA and GDPR.
  Neither regulation fixes an explicit epsilon, so a deployment still needs its own
  compliance assessment.
- The attention overlay lets a clinician check whether the decision follows clinically
  meaningful parts of the trace rather than incidental signal.
- The watchdog flags inputs whose feature distribution shifts away from clean data, so a
  tampered or corrupted reading can be held back for review.
- The study underlying this device used only public, de-identified datasets with the
  relevant approvals. No new human-subject data was collected.

---

## 11. Troubleshooting

| Symptom | Check |
| --- | --- |
| `wfdb` cannot find records | run `bash scripts/prepare_data.sh` and set `data.root` |
| accuracy near chance after quick start | expected; the smoke run trains two steps only |
| export error about output type | use the console `export` command, which wraps the logits path |
| type or lint failure | run `make lint` and `make type` to see the exact line |
| slow private step | reduce `train.batch_size`; per-sample gradients scale with the batch |

---

## 12. Self-test

```
make test
```

The self-test covers signal shape, passband response, attention normalisation, private
budget allocation, mixture density, metric correctness, single-batch overfitting,
gradient flow, and a two-step end-to-end run.
