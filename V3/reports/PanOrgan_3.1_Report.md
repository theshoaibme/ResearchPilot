# PanOrgan 3.1 — Technical Upgrade & Active Run Report

## 1. Overview & Objectives
`PanOrgan 3.1` (`PanOrgan 3 Cohort`, Version `3.1`) directly resolves the primary gaps identified in `PanOrgan 3`:
1. Integrates verified official supervised labels for **`CT` (`tcga_kirc_ct`)** and **`MG` (`cbis_ddsm`)**, completing **5-modality supervised training** (`MRI`, `CT`, `CXR`, `US`, `MG`).
2. Adds **`MixUp` augmentation** (`alpha = 0.2`, `p = 0.25`) on classification-only batches to improve generalization and calibration.
3. Adds explicit **`val_loss` tracking** per epoch alongside `val_f1`, `val_acc`, and `val_dice`.
4. Warm-starts from the calibrated Release 3 weights (`/Volumes/Shoaib/PanOrgan_V3/checkpoints/v3_stage2b_calibrated/best.pt`) rather than restarting from scratch.

## 2. Architecture & Modeling
- **Base Weights:** Warm-started from `v3_stage2b_calibrated/best.pt` (`240 MB`).
- **Training Script:** `V3/scripts/train_v31_upgrade.py` (`--epochs 5 --checkpoint-every-steps 20`).
- **Output Artifacts:**
  - Checkpoints: `/Volumes/Shoaib/PanOrgan_V3/checkpoints/v31_5modal_mixup/`
  - Logs: `/Volumes/Shoaib/PanOrgan_V3/experiments/v31_stdout.log` and `v31_pipeline.log`
  - Target Evaluation Report: `/Volumes/Shoaib/PanOrgan_V3/experiments/thesis_reports/v31_complete_thesis_report.json`

## 3. Datasets & Official Label Additions
Updating `master_manifest.parquet` via `V3/scripts/add_v31_labeled_cohorts.py` produced **11,735 total sharded samples** (`8,078 train / 1,839 val / 1,818 test`) across 8 cohorts (`3,777` unique patients, zero-leakage verified):
- **New `tcga_kirc_ct` (`CT`, `1,200` samples):** `797 train / 214 val / 189 test`, `15` unique patients. Labeled with official renal tumor grades (`G1/G2` vs `G3/G4`) from the NCI GDC API (`tcga_kirc_official_clinical.csv`, `537` patients).
- **New `cbis_ddsm` (`MG`, `88` samples):** `59 train / 14 val / 15 test`, `45` unique patients. Labeled with official pathology (`BENIGN` vs `MALIGNANT`) from `mass_case_description_train/test_set.csv` and `calc_case_description_train/test_set.csv`.
- **Active 3.1 Supervised Pool:** `7,266 train / 1,642 val / 1,818 test`.

## 4. Training & Calibration Protocol
- **MixUp Regularization:** Applied with `alpha = 0.2`, `p = 0.25` on classification-only batches (`908` steps/epoch).
- **Validation & Checkpointing:** Explicit `val_loss` logged every epoch; step checkpoints saved every `20` steps for power-loss resilience.
- **Execution:** Running on Apple M4 GPU (`MPS`) with `caffeinate -dimsu` protection.

## 5. Current Run Status & Evaluation Metrics
- **Training Status:** **In Progress** (`PID 5831`, currently in `Epoch 1/5` at `~15%` / `~3.8s/it`, `908` batches/epoch).
- **Final Test Metrics (`v31_complete_thesis_report.json`):** Pending completion of all 5 epochs and post-training evaluation. No projected or synthetic metrics are reported prior to run completion.

## 6. Known Limitations
1. **Small Mammography Sample Pool:** `cbis_ddsm` currently contributes `88` labeled samples (`45` patients; `15` test samples), which limits statistical power on `MG` relative to `MRI` and `CXR`.
2. **Low Unique Patient Count in `tcga_kirc_ct`:** Although `tcga_kirc_ct` provides `1,200` CT slices, they originate from `15` unique patients (`189` test slices).
3. **Unlabeled `prostatex_mri`:** `prostatex_mri` (`1,198` samples) remains excluded from supervised loss (`label = -1`) because its official clinical findings file requires restricted TCIA challenge access.
4. **Evaluation Pending:** `3.1` test-set `AUROC`, `Macro-F1`, `Dice`, `ECE`, and `MC2` coverage cannot be finalized until `PID 5831` completes.
