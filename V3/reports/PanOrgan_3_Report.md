# PanOrgan 3 — Technical Implementation & Benchmark Report

## 1. Overview & Objectives
`PanOrgan 3` (`PanOrgan 3 Cohort`) scaled the framework from 3 cohorts to **7 clinical cohorts** (and `8` total sharded cohorts including `prostatex_mri` for SSL), introducing a three-stage training curriculum (Stage 1 SSL, Stage 2A Joint Multi-Task, Stage 2B Head Calibration) to dramatically improve classification F1, calibration error (`ECE`), and cross-organ segmentation fidelity.

## 2. Architecture & Modeling
- **Unified Backbone + `OmniModalityStem` + `OntologyPromptEncoder`:** Retained the shared multimodal architecture while expanding task coverage across neurology MRI, thoracic CXR, breast ultrasound, and brachial plexus ultrasound.
- **Checkpoint Footprint:**
  - `v3_stage2b_calibrated/best.pt`: `240 MB` (final calibrated release weights)
  - `v3_stage2a_joint/best.pt`: `715 MB` (joint training checkpoint)
  - Total checkpoint directory footprint across stages: `~3.54 GB` on `/Volumes/Shoaib/PanOrgan_V3/checkpoints/`.

## 3. Datasets & Splitting Strategy
`PanOrgan 3` expanded the zero-leakage manifest (`master_manifest.parquet` on `/Volumes/Shoaib/PanOrgan_V3/preprocessed_shards/`) across 4 modalities (`MRI`, `CXR`, `US`, plus unlabeled `CT`/`MG` in SSL):
- **Supervised Test Evaluation Pool:** `n = 1,425` classification test samples, `n = 1,246` segmentation test samples (`tcga_lgg`, `pulmonology_cxr`, `busi`, `ultrasound_nerve`, `nih_cxr14`).

## 4. Training & Calibration Protocol
1. **Stage 1 (Self-Supervised Learning):** Masked reconstruction pretraining (`best_val_loss = 0.003693`).
2. **Stage 2A (Joint Training):** `batch_size = 8`, `grad_accum = 4` on Apple M4 (`MPS`), achieving `Ep 17 best_composite = 0.8679` (`val_dice = 0.8350`, `val_f1 = 0.9009`, `val_acc = 88.90%`).
3. **Stage 2B (Head Calibration):** Focused classification head calibration (`Ep 1 best_val_f1 = 0.9017`, `val_acc = 89.18%`), followed by temperature scaling (`T = 0.9390`) and `MC2` conformal calibration.
4. **Post-Hoc Threshold Tuning:** Evaluated `High-Sens (0.6, 1.3, 1.3, 1.3)` operating point (`Acc = 88.07%`, `F1 = 0.8938`, `Sens = 89.54%`, `Spec = 94.61%`).

## 5. Verified Evaluation Metrics
Source: `/Volumes/Shoaib/PanOrgan_V3/experiments/thesis_reports/v3_complete_thesis_report.json`

### Classification & Calibration (`n = 1,425`)
| Metric | Uncalibrated | Temperature Scaled (`T = 0.9390`) |
| :--- | :---: | :---: |
| ROC-AUC | 0.9715 | 0.9713 |
| Accuracy | 87.93% | 87.93% |
| Macro-F1 | 0.8905 | 0.8905 |
| Macro Precision | 0.8956 | 0.8956 |
| Macro Sensitivity (Recall) | 0.8879 | 0.8879 |
| Macro Specificity | 0.9442 | 0.9442 |
| ECE | 0.0792 | 0.0598 |
| Brier Score | 0.2105 | 0.2077 |

### Segmentation (`n = 1,246`)
| Cohort | Samples | Dice (95% CI) | IoU | HD95 (mm) |
| :--- | :---: | :---: | :---: | :---: |
| `pulmonology_cxr` | 334 | 0.9637 (0.9600–0.9672) | 0.9316 | 9.61 |
| `tcga_lgg` | 537 | 0.8753 (0.8541–0.8951) | 0.8413 | 6.90 |
| `busi` | 109 | 0.7790 (0.7280–0.8240) | 0.6983 | 28.78 |
| `ultrasound_nerve` | 266 | 0.6175 (0.5620–0.6680) | 0.5863 | 9.77 |
| **Overall** | **1,246** | **0.8355 (0.8180–0.8510)** | **0.7986** | **10.16** |

### Conformal Coverage & MNAR Stress Test
- **MC2 Conformal Coverage:** Marginal coverage `93.33%`, average set size `1.5277`, abstention rate `20.98%`.
- **MNAR Stress Curve (Overall Dice):** `0.8355` (0%) -> `0.8380` (20%) -> `0.7908` (40%).

## 6. Known Limitations
1. **Missing Supervised `CT` and `MG` Labels in Release 3:** `tcga_kirc_ct` (`CT`) and `cbis_ddsm` (`MG`) were not yet integrated with supervised clinical labels during Release 3 training, limiting supervised evaluation to `MRI`, `CXR`, and `US`.
2. **Absence of MixUp & Explicit `val_loss` Logging:** Stage 2A/2B scripts did not apply `MixUp` regularization on classification-only batches or log per-epoch `val_loss` separately alongside `val_f1`/`val_dice`.
3. **Ultrasound Nerve Segmentation Ceiling:** `ultrasound_nerve` achieved `Dice = 0.6175`, substantially lower than `pulmonology_cxr` (`0.9637`) and `tcga_lgg` (`0.8753`).
4. **Unlabeled Prostate MRI:** `prostatex_mri` (`1,198` samples) remained unlabeled (`label = -1`) due to restricted TCIA challenge credentials.
