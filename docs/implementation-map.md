# Implementation Map

Source modules carry no comments or docstrings; all paper provenance lives here.
Package layout follows an embedded-firmware signal-stack metaphor: `afe` (analog
front-end), `acquisition` (sensor drivers), `fabric` (compute fabric / encoder),
`governor` (privacy governor), `salience` (attention/saliency), `watchdog`
(adversarial watchdog), `decoder` (classification head), `assembly` (full image),
`scheduler` (training loop), `bench` (test bench / evaluation), `instrument`
(utilities), `console` (CLI).

## Equations

| Paper item | Location | File | Symbol / function |
|---|---|---|---|
| Eq 1  z = SPM(x) | §III.B | `afe/conditioning.py` | `SignalFrontEnd.forward` |
| Eq 2  h = FEB-APBA(z) | §III.B | `fabric/backbone.py` | `ConvFabric.forward` |
| Eq 3  (h',A) = PAAM(h) | §III.B | `salience/attention.py` | `PrivacyAwareAttention.forward` |
| Eq 4  (ŷ,s_adv) = CH-LADL(h') | §III.B | `decoder/head.py`, `watchdog/detector.py` | `ClassifierHead.forward`, `AnomalyWatchdog.score` |
| Eq 5  z-score of BPF/BW | §III.C | `afe/conditioning.py` | `bandpass`, `remove_baseline`, `normalize` |
| Eq 6  S^(k) gradient sensitivity | §III.D | `governor/sensitivity.py` | `gradient_sensitivity` |
| Eq 7  ε^(k) inverse-sensitivity budget | §III.D | `governor/allocation.py` | `allocate_budget` |
| Eq 8  σ^(k)_t noise scale | §III.D | `governor/allocation.py` | `noise_scales` |
| Eq 9  aggregated noisy gradient | §III.D | `governor/engine.py` | `PrivateStep.aggregate` |
| Alg 1 APBA training | §III.D / Fig | `governor/engine.py`, `scheduler/loop.py` | `PrivateStep`, `train` |
| Eq 10 scaled dot-product attention | §III.E | `salience/attention.py` | `scaled_dot_product` |
| Eq 11 head_j projections | §III.E | `salience/attention.py` | `PrivacyAwareAttention` heads |
| Eq 12 Ã = softmax(QKᵀ/√d_k + N_att) | §III.E | `salience/attention.py` | `_perturb_scores` |
| Eq 13 Δ_att attention sensitivity / λ_att | §III.E | `salience/perturbation.py` | `attention_sensitivity`, `laplace_scale` |
| Eq 14 S saliency map | §III.E | `salience/saliency.py` | `saliency_map` |
| Eq 15 GMM p(h'|y=c) | §III.F | `watchdog/mixture.py` | `DiagonalGMM.log_prob` |
| Eq 16 s_adv = -log p + β·D_MD | §III.F | `watchdog/detector.py` | `AnomalyWatchdog.score` |

## Architecture / hyperparameters

| Paper item | Location | File |
|---|---|---|
| Encoder: 4 Conv1d blocks 32/64/128/128, k=7, s=2, BN, ReLU, GAP→128 | Table XV, §IV.G | `fabric/backbone.py` |
| Attention: H=4, d_k=32 | §IV.G | `salience/attention.py`, `configs/model/*.yaml` |
| Classifier: FC 128→64→C softmax | Table XV | `decoder/head.py` |
| Pan-Tompkins R-peak, resample 256 centred on R | §III.C | `afe/rpeak.py`, `afe/segment.py` |
| 4th-order Butterworth [0.5,45] Hz, median-filter BW | §III.C | `afe/conditioning.py` |
| Adam, lr 1e-3, cosine, batch 256, 100 epochs, wd 1e-4 | Table XVI | `configs/train/*.yaml`, `scheduler/loop.py` |
| APBA warmup 5, α 0.5, clip 90th percentile | Table XVI, §IV.B | `governor/*`, `configs/train/*.yaml` |
| GMM M=2, β 0.5, τ 95th percentile | Table XVI, §IV.F | `watchdog/*`, `configs/model/*.yaml` |
| RDP orders {2,5,10,20,50,100}→ε 4.2 @ α*=20, δ 1e-5 | §III.D | `governor/accountant.py` |

## Metrics and evaluation

| Paper item | Location | File |
|---|---|---|
| ACC / PR / REC / F1 macro / AUROC | §IV.D, Tab II | `bench/classification.py` |
| Privacy budget ε (δ=1e-5) | §IV.D, Tab V | `governor/accountant.py` |
| Adversarial ACC/F1 under PGD/FGSM/SAP | §IV.D, Tab III | `bench/robustness.py`, `bench/attacks.py` |
| ADR / FPR | §IV.D, Tab IV | `bench/detection.py` |
| IoU / Pointing-Game vs cardiologist annotation | §IV.D, Tab VI | `bench/interpretability.py` |
| Model size / latency / power (int8) | §IV.D, Tab XIII | `bench/footprint.py` |
| Bootstrap 95% CI (Tab XIV) | §IV.K | `bench/intervals.py` |
| McNemar + Wilcoxon + Bonferroni α_adj=0.0017 | §IV.K | `bench/significance.py` |

## Attacks (Table XVII)

| Attack | ε | iters | step | norm | File |
|---|---|---|---|---|---|
| FGSM | 0.1 | 1 | 0.1 | L∞ | `bench/attacks.py::fgsm` |
| PGD | 0.1 | 40 | 0.01 | L∞ | `bench/attacks.py::pgd` |
| SAP | 0.1 | 40 | 0.01 | L∞ (σ=3 smoothing) | `bench/attacks.py::sap` |

## Experiments → configs

| Reported experiment | Table | Config |
|---|---|---|
| Main clean classification | II | `experiment/main.yaml` |
| Adversarial robustness | III | `experiment/main.yaml` (eval attacks) |
| LADL detection | IV | `experiment/main.yaml` |
| Privacy-utility sweep ε∈{1,2,4,8,∞} | V | `experiment/supplementary_privacy_sweep.yaml` |
| Interpretability | VI | `experiment/main.yaml` |
| Ablation component contribution | VII | `experiment/ablation_components_*.yaml` |
| APBA exponent α∈{0.25,0.5,0.75,1.0} | VIII | `experiment/ablation_alpha_*.yaml` |
| LADL config (M, β) | IX | `experiment/ablation_ladl_*.yaml` |
| Per-class MIT-BIH / CPSC | X, XI | `experiment/main.yaml`, `experiment/main_cpsc.yaml` |
| SOTA comparison | XII | `experiment/main.yaml` |
| Deployment quantization | XIII | `experiment/main.yaml` (export) |
| Cross-dataset transfer | XVIII | `experiment/supplementary_cross_dataset.yaml` |
| Noise-level sensitivity SNR | XIX | `experiment/supplementary_noise_sensitivity.yaml` |

## Ablation switches

`assembly/config.py::TrustHeartConfig` exposes `use_apba`, `use_paam`, `use_ladl`
(Table VII rows: Base / +APBA / +PAAM / +LADL / pairwise / Full). Allocation exponent
α and LADL (M, β) are config scalars for Tables VIII–IX.
