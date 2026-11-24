# PanOrgan 3.1 — Final Technical & Benchmark Report

## 1. Overview & Objectives
`PanOrgan 3.1` (`PanOrgan 3 Cohort`, Version `3.1`) resolves the primary gaps identified in `PanOrgan 3`:
1. Integrates verified official supervised labels for **`CT` (`tcga_kirc_ct`)** and **`MG` (`cbis_ddsm`)**, completing **5-modality supervised training** (`MRI`, `CT`, `CXR`, `US`, `MG`).
2. Adds **`MixUp` augmentation** (`alpha = 0.2`, `p = 0.25`) on classification-only batches to improve generalization and calibration.
3. Adds explicit **`val_loss` tracking** per epoch alongside `val_f1`, `val_acc`, and `val_dice`.
4. Warm-starts from the calibrated Release 3 weights (`/Volumes/Shoaib/PanOrgan_V3/checkpoints/v3_stage2b_calibrated/best.pt`).

## 2. Architecture & Modeling
- **Warm-Start Weights:** Initialized from `v3_stage2b_calibrated/best.pt` (`240 MB`).
- **Final 3.1 Checkpoint:** `/Volumes/Shoaib/PanOrgan_V3/checkpoints/v31_5modal_mixup/best.pt` (`715 MB`, selected at Epoch 5 with best composite score `0.8671`).
- **Training Script:** `V3/scripts/train_v31_upgrade.py` (`--epochs 5 --checkpoint-every-steps 20`).
- **Evaluation Report:** `/Volumes/Shoaib/PanOrgan_V3/experiments/thesis_reports/v31_complete_thesis_report.json`

## 3. Datasets & Official Label Additions
Updating `master_manifest.parquet` via `V3/scripts/add_v31_labeled_cohorts.py` produced **11,735 total sharded samples** (`8,078 train / 1,839 val / 1,818 test`) across 8 cohorts (`3,777` unique patients, zero-leakage verified):
- **New `tcga_kirc_ct` (`CT`, `1,200` samples):** `797 train / 214 val / 189 test`, `15` unique patients. Labeled with official renal tumor grades (`G1/G2` vs `G3/G4`) from the NCI GDC API (`tcga_kirc_official_clinical.csv`, `537` patients).
- **New `cbis_ddsm` (`MG`, `88` samples):** `59 train / 14 val / 15 test`, `45` unique patients. Labeled with official pathology (`BENIGN` vs `MALIGNANT`) from `mass_case_description_train/test_set.csv` and `calc_case_description_train/test_set.csv`.
- **Active 3.1 Supervised Pool:** `7,266 train / 1,642 val / 1,818 test` (evaluation `n = 1,629` classification, `n = 1,246` segmentation).

## 4. Training Progression (5 Epochs on Apple M4 MPS)

| Epoch | Train Loss | Val Loss | Val Dice | Val Macro-F1 | Val Accuracy | Composite |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 2.6559 | 3.7232 | 0.8089 | 0.8707 | 84.90% | 0.8398 |
| 2 | 2.6242 | 3.8593 | 0.8339 | 0.8643 | 84.90% | 0.8491 |
| 3 | 2.4957 | 4.3587 | 0.8213 | 0.8219 | 81.91% | 0.8216 |
| 4 | 2.4389 | 3.9015 | 0.8338 | 0.8832 | 86.48% | 0.8585 |
| **5** | **2.4089** | **3.8846** | **0.8404** | **0.8938** | **87.15%** | **0.8671** |

## 5. Verified Held-Out Test Metrics

### Classification & Calibration (`n = 1,629`)
| Metric | Uncalibrated | Temperature Scaled (`T = 1.0354`) |
| :--- | :---: | :---: |
| AUROC | 0.9590 | 0.9593 |
| Accuracy | 85.57% | 85.57% |
| Macro-F1 | 0.8814 | 0.8814 |
| Macro Precision | 0.8832 | 0.8832 |
| Macro Sensitivity (Recall) | 0.8802 | 0.8802 |
| Macro Specificity | 0.9358 | 0.9358 |
| ECE | 0.0611 | 0.0784 |
| Brier Score | 0.2483 | 0.2499 |

### Classification Confusion Matrix (`4 x 4`, `n = 1,629`)
```text
[[723,  83,  12,   1],
 [113, 487,   5,   1],
 [ 15,   4, 105,   0],
 [  1,   0,   0,  79]]
```

### Segmentation (`n = 1,246`, Mean & 95% Bootstrap CI)
| Cohort | Samples | Dice (95% CI) | IoU (95% CI) | HD95 (mm) |
| :--- | :---: | :---: | :---: | :---: |
| `pulmonology_cxr` | 334 | 0.9664 (0.9631–0.9695) | 0.9366 (0.9309–0.9419) | 8.24 |
| `tcga_lgg` | 537 | 0.8901 (0.8714–0.9087) | 0.8560 (0.8353–0.8753) | 6.13 |
| `busi` | 109 | 0.7677 (0.7165–0.8155) | 0.6871 (0.6340–0.7385) | 37.30 |
| `ultrasound_nerve` | 266 | 0.6402 (0.5875–0.6897) | 0.6028 (0.5505–0.6524) | 10.91 |
| **Overall** | **1,246** | **0.8465 (0.8294–0.8627)** | **0.8088 (0.7914–0.8249)** | **10.30** |

### Conformal Prediction (`MC2`, `alpha = 0.10`)
- **Overall:** Marginal Coverage `92.51%`, Avg Set Size `1.7397`, Abstention Rate `29.59%`.
- **Per-Cohort Breakdown:**
  - `nih_cxr14` (`n = 179`): `100.0%` coverage, `0.0%` abstention, set size `1.00`
  - `tcga_lgg` (`n = 537`): `94.97%` coverage, `0.0%` abstention, set size `1.00`
  - `busi` (`n = 109`): `91.74%` coverage, `11.01%` abstention (down from `30.28%` in Release 3), set size `1.13`
  - `pulmonology_cxr` (`n = 334`): `87.72%` coverage, `0.0%` abstention, set size `1.00`
  - `cbis_ddsm` (`n = 15`): `93.33%` coverage, `100.0%` abstention, set size `3.93`
  - `tcga_kirc_ct` (`n = 189`): `85.19%` coverage, `100.0%` abstention, set size `3.41`
  - `ultrasound_nerve` (`n = 266`): `93.98%` coverage, `100.0%` abstention, set size `3.60`

## 6. Known Limitations
1. **Small Mammography Sample Pool:** `cbis_ddsm` contributes `88` labeled samples (`45` patients; `15` test), yielding `93.33%` conformal coverage but `100%` abstention.
2. **Low Unique Patient Count in `tcga_kirc_ct`:** `1,200` CT slices originate from `15` unique patients (`189` test slices), yielding `85.19%` conformal coverage with `100%` abstention.
3. **Temperature Scaling ECE:** Calibrated ECE (`0.0784` at `T = 1.0354`) was slightly higher than uncalibrated ECE (`0.0611`).
4. **Unlabeled `prostatex_mri`:** `prostatex_mri` (`1,198` samples) remains excluded from supervised loss (`label = -1`) due to restricted TCIA challenge credentials.
