# Project Context

project_name       : trustheart                                  [HIGH]
domain             : consumer wearable ECG arrhythmia classification with
                     differential privacy, explainable attention, and
                     adversarial anomaly detection (IoMT edge)        [HIGH]
framework          : PyTorch 2.0 (plain torch.nn); hand-written DP-SGD [HIGH]
venue              : IEEE Transactions on Consumer Electronics        [HIGH]
primary_datasets   : 2 datasets (see §6)                              [HIGH]
compute_target     : RTX 3090 (train) / ARM Cortex-M4 168 MHz (deploy) [HIGH]
hparams_reference  : Table XVI (Appendix B) + Methods §III–IV         [HIGH]
supp_path          : none (single-file manuscript; appendix A–E inline)
extra_signals      : 16 numbered equations + Algorithm 1; Tables II–XIX;
                     code-availability sentence present; ethics: public
                     anonymised data only

NEEDS_USER_DECISION: 0
Status: resolved.

## 1. project_name

`trustheart` (snake_case from the method name "TrustHeart"). Title L1, Abstract L1.
Confidence HIGH.

## 2. supp_path

`none`. The manuscript is self-contained: Appendix A (network spec, Table XV),
Appendix B (hyperparameters, Table XVI), Appendix C (attack configs, Table XVII),
Appendix D (cross-dataset, Table XVIII), Appendix E (noise sensitivity, Table XIX).
No external supplement file is cross-referenced.

## 3. domain

Computational cardiology on consumer wearables: single/few-lead ECG beat
classification into arrhythmia groups, under three coupled requirements —
(ε,δ)-differential privacy during training, privacy-aware interpretable attention,
and lightweight adversarial-input detection — targeting microcontroller deployment.
Methods §III.A (problem formulation), Abstract L1–L8. HIGH.

## 4. framework

PyTorch 2.0 (Implementation Details §IV.B: "experiments are conducted using
PyTorch 2.0 and Opacus 1.4"). The reported DP engine is Opacus, but APBA requires
per-layer adaptive noise scales σ^(k) (Eq 7–9, Alg 1) that the standard Opacus
PrivacyEngine does not express; per R1 the DP-SGD loop and per-layer noise are
implemented directly on torch tensors, with Rényi-DP accounting written by hand
(§III.D, ref [39]). Deployment path uses STM32CubeAI + TensorFlow Lite Micro
(§IV.B, §IV.G); reproduced here as ONNX export + documented int8 quantization.
HIGH.

## 5. venue

IEEE Transactions on Consumer Electronics. Evidence: filename prefix "TCE", the
Index Terms list "Consumer Electronics", the abstract and discussion frame every
contribution around consumer wearable deployment, and the two-column IEEE
Transactions layout. HIGH.

## 6. primary_datasets

### MIT-BIH Arrhythmia Database
- name: MIT-BIH Arrhythmia Database
- version: 1.0.0
- license: Open Data Commons Attribution License v1.0 (PhysioNet canonical page)
- access URL: https://physionet.org/content/mitdb/1.0.0/
- details: 48 half-hour 2-channel records, 360 Hz, digitised at 11-bit over a 10 mV
  range; AAMI 5-group mapping N/S/V/F/Q; ~110k annotated beats (Table I: 109,494);
  inter-patient De Chazal protocol, 22 training / 22 testing records (§IV.A, ref [41]).

### CPSC2018
- name: China Physiological Signal Challenge 2018
- version: 2018 release
- license: CC BY-NC-SA 4.0 (challenge data-use terms, canonical challenge page)
- access URL: http://2018.icbeb.org/Challenge.html
- details: 6,877 12-lead recordings, 500 Hz, 6–60 s; 9 classes
  N/AF/I-AVB/LBBB/RBBB/PAC/PVC/STD/STE; five-fold patient-level cross-validation
  (§IV.A, Table I).

Both datasets are public and de-identified; no private/restricted data is used.

## 7. compute_target

- Training: NVIDIA RTX 3090 GPUs (§IV.B "All experiments are performed using NVIDIA
  RTX 3090 GPUs").
- Deployment: ARM Cortex-M4 @ 168 MHz via STM32CubeAI and TensorFlow Lite Micro
  (§IV.B, §IV.G).
- Footprint (Table XIII): Float32 1,180 KB / 142 ms; Int8 PTQ & QAT 295 KB / 35 ms /
  64 KB RAM / 2.3 mW; +LADL overhead +2.5 KB / +0.9 ms / +4 KB / +0.1 mW.
- Wall-clock for full training is not reported per run; 100 epochs over MIT-BIH
  (~110k beats) and CPSC2018 (6,877 records) at batch 256.

## 8. hparams_reference

Table XVI (Appendix B) is the authoritative set, corroborated by Methods prose:
optimizer Adam; initial lr 1e-3; cosine-annealing schedule; batch size 256;
100 training epochs; weight decay 1e-4; APBA warmup 5 epochs; APBA allocation
exponent α = 0.5; gradient clipping threshold = 90th percentile of warmup gradient
norms; LADL GMM components M = 2; LADL balance β = 0.5; attention noise λ_att
adaptive (Eq 13). Privacy: δ = 1e-5; main result ε = 4.2 obtained from RDP
composition over orders α ∈ {2,5,10,20,50,100} with tightest bound at α* = 20
(§III.D). Privacy sweep ε ∈ {1,2,4,8,∞} (Table V). Attack configs ε = 0.1 with
PGD/SAP 40 iterations, step 0.01, L∞ (Table XVII).

## 9. extra_signals

- 16 numbered equations (Eq 1–16) and one algorithm box (Algorithm 1, APBA training).
- Reported tables II–XIX, including ablation (VII), exponent sensitivity (VIII),
  LADL config (IX), per-class (X, XI), SOTA comparison (XII), deployment (XIII),
  confidence intervals (XIV), cross-dataset (XVIII), noise sensitivity (XIX).
- No proprietary tokenizer; no released checkpoints.
- Code-availability sentence (Reproducibility Statement): "To support
  reproducibility, source code and evaluation scripts will be released upon
  acceptance." Retained here for provenance; per release instruction it is omitted
  from the README.
- Ethics: only public, anonymised datasets with appropriate approvals; no new human
  subject data; no conflict of interest.
- Three board-certified cardiologists (>10 years) annotated 200 beats, inter-annotator
  Cohen's κ = 0.82, consensus labels as IoU/Pointing-Game ground truth (§IV.D).
