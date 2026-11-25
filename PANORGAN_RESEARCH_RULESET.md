# PanOrgan — Publication-Grade Multimodal Medical AI Rule Set

> **Purpose:** Master engineering, data, training, evaluation, and publication checklist for **PanOrgan 4.1+** and future thesis/journal submissions (*MICCAI, IEEE TMI, Medical Image Analysis, Nature Digital Medicine*).  
> **Core Principle:** Never hardcode unrealistic universal targets (e.g., forcing "95% everywhere" ignores task-intrinsic Bayes error, imaging physics, and label noise). Instead, enforce **zero leakage, balanced representation, strong calibration, rigorous statistical validation, and transparent reporting of limitations**.

---

## 1. Data Governance, Provenance & Hygiene

1. **Incremental Gap-Filling (Keep Strong Cohorts, Fix Weak Ones):**
   - Retain all validated strong cohorts (`CXR`, `Brain MRI`, `Breast US`) and selectively augment only data-starved, imbalanced, or low-resolution cohorts (`cbis_ddsm`, `cmmd_mg`, `kidney_ct_abn`, `tcga_kirc_ct`).
2. **Strict Patient-Level Zero-Leakage Splitting:**
   - All `train / val / test` splits MUST group by `patient_id` (`leakage_count == 0`). Never split 2D slices or multiple views from the same patient across splits.
   - Verify split integrity programmatically before every training run.
3. **Source Provenance, Licensing & Ethics Audit:**
   - Every cohort in the manifest must record: **exact source URL**, **license** (`CC BY 4.0`, `CC BY 3.0`, `CC BY-NC-SA 4.0`, etc.), **citation**, and **research/commercial permission**.
4. **Class Balance & Slice Quality Filtering:**
   - Audit class ratios per cohort before training. Target balanced representation (or inverse-sqrt class-weighted loss + balanced batch sampling) so no head collapses to predicting only the majority/positive class.
   - Filter blank, corrupted, or near-zero-variance edge slices (`std < 0.05` after normalization) during CT/MRI preprocessing so empty slices do not pollute training or sample grids.
5. **External Zero-Shot Holdout:**
   - Reserve at least one completely unseen external cohort per major modality (e.g., `pulmonology_cxr_ext`) strictly for out-of-distribution (OOD) generalization testing.

---

## 2. Modality Harmonization, Resolution & 2.5D/3D Physics

1. **Modality-Appropriate Intensity Normalization:**
   - **CT:** Apply multi-window Hounsfield Unit (HU) clipping (e.g., soft-tissue, lung, bone/contrast channels) instead of naive min-max scaling.
   - **MRI / US / MG:** Apply robust percentile clipping (e.g., `1st–99.5th` percentile) and z-score standardization per scan.
2. **Resolution & ROI Preservation:**
   - Do not crush high-frequency pathology (e.g., mammography microcalcifications, subtle renal lesions) down to `256×256` full-frame images.
   - Use **`384×384` or `512×512` lesion/organ ROI crops** for Mammography and CT grading tasks.
3. **True 2.5D / 3D Volumetric Representation:**
   - Architecture support alone (`OmniModalityStem.stem_25d`) is not enough—verify `is_3d == True` in the manifest for volumetric CT/MRI cohorts and feed **5-slice volumetric slabs (`slab_depth = 5`)** conditioned on physical voxel spacing `(log dx, log dy, log dz)` and `body_coord_z` via `PhysicsFiLM`.
4. **3D Volumetric Reconstruction Metrics:**
   - For volumetric segmentation, evaluate both 2D slice Dice and reconstructed 3D patient-level **Dice** and **HD95 (mm)** where physical spacing is available.

---

## 3. Architecture & Multi-Task Interference Control

1. **Cohort-Isolated Classification Heads:**
   - Never share a single binary/multiclass output logit across clinically distinct cohorts (e.g., `kidney_ct_abn` abnormality vs `tcga_kirc_ct` tumor grade). Use cohort-isolated heads over a shared multimodal representation.
2. **Multi-Scale Spatial Feature Pooling:**
   - Combine hierarchical feature maps (`32×32`, `16×16`, `8×8` Avg+Max pooling) with the global embedding so small focal lesions are not washed out by global average pooling.
3. **Preventing Negative Transfer:**
   - Monitor per-cohort validation curves independently during joint training. If one modality degrades another, apply gradient norm balancing, LoRA/adapter isolation, or staged backbone + specialist head fine-tuning.

---

## 4. Anti-Overfitting, Optimization & Training Stability

1. **Conservative Learning Rate & Schedule:**
   - Use **low backbone LR (`1.5e-5` to `3e-5`)** and moderate head LR (`8e-5` to `1.5e-4`) with **linear warmup (5–10% of steps)** followed by **cosine annealing**.
2. **Strong Regularization Stack:**
   - `Weight Decay`: `0.05 – 0.08`
   - `DropPath`: `0.20 – 0.25`
   - `Head Dropout`: `0.30 – 0.40`
   - `Label Smoothing`: `0.03 – 0.05`
   - `MixUp / CutMix`: `p = 0.25 – 0.35` (for classification; disable or mask-align for segmentation)
   - `EMA (Exponential Moving Average)`: decay `0.998` for evaluation weights.
3. **Honest Early Stopping & Checkpointing:**
   - Select checkpoints by **Validation Macro-F1 + Segmentation Dice** (never test set metrics!).
   - Track `train_loss` and `val_loss` using the **exact same loss formula and reduction** so train/val curves are directly comparable without artificial scale gaps.

---

## 5. Calibration, Balanced Decisions & Uncertainty (Crucial for Clinical Trust)

1. **No Accuracy-Only Reporting & No Always-Positive Collapse:**
   - Always report **Weighted Accuracy**, **Balanced Accuracy**, **Macro-F1**, **AUROC**, **AUPRC (PR-AUC)**, and **per-class Precision / Recall (Sensitivity & Specificity)**.
2. **Post-Hoc Probability Calibration:**
   - Fit **Temperature Scaling** on the validation set per head to minimize **Expected Calibration Error (`ECE < 0.05`)** and **Brier Score**. Plot **Reliability Diagrams**.
3. **Principled Decision Thresholding:**
   - Tune decision thresholds on the validation set using **Macro-F1 or Youden's J (`Sensitivity + Specificity - 1`)**—never leave an uncalibrated threshold that causes one class's recall to collapse (e.g., `ultrasound_nerve` at `0.63` or `tcga_kirc_ct` at `0.40`).
4. **Conformal Uncertainty & Abstention:**
   - Apply **Modality-Conditional Conformal Prediction (`MC2`)** targeting `90%` coverage and report `coverage`, `avg_set_size`, and `abstention_rate`.

---

## 6. Robustness, Explainability & Clinical Sanity Audits (Often Missed!)

1. **Stress & Occlusion Testing:**
   - Run **MNAR Field-of-View (FOV) Occlusion** (`0%`, `20%`, `40%`), noise/contrast perturbation, and missing-modality resilience tests.
2. **Visual & Saliency Verification:**
   - Inspect **Grad-CAM / Attention maps** and **Segmentation Mask Overlays** across all cohorts (20+ representative test samples) to verify the model attends to pathology rather than hospital watermarks, borders, or text markers (shortcut learning check).
3. **Subgroup & Domain-Shift Audit:**
   - Break down performance by scanner/site/cohort and check for spurious correlations (e.g., image size, brightness, or contrast differences between benign and malignant cases).

---

## 7. Statistical Rigor, Realistic Targets & Publication Checklist

1. **Realistic Cohort-Specific Benchmarks (Not Hardcoded 95%):**
   - **High-SNR Structural Tasks** (`CXR 4-class`, `Brain MRI LGG`, `Breast US BUSI`): Target **`90–95%+`** Accuracy/AUROC.
   - **Fine-Grained / Low-Contrast Tasks** (`Mammography CMMD/CBIS`, `Kidney CT Abn/Grading`, `Nerve US`): Target **`78–88%+` AUROC / Macro-F1** with high-res ROI + 2.5D slabs.
2. **Confidence Intervals & Significance Testing:**
   - Compute **95% Bootstrap Confidence Intervals (`n = 1,000` resamples)** for all headline metrics.
   - Use **DeLong's test** (for AUROC) and **McNemar's test** (for paired predictions) when claiming superiority over baselines.
3. **Mandatory Ablation Studies for Reviewers:**
   - Compare unified multimodal model vs. single-modality / standard 2D backbone (`ConvNeXt-Small` baseline).
   - Ablate `OmniModalityStem + PhysicsFiLM`, `2.5D slabs vs 2D slices`, `Multi-Scale Heads`, `TTA`, and `Temperature Scaling`.
4. **Reproducibility & Honest Limitations:**
   - Lock random seeds (`seed=42`), save exact configs/manifests/hashes, plot full-scale training curves (no misleading zoomed Y-axes), and explicitly document remaining limitations (e.g., sample size, 2D vs 3D, resolution trade-offs) in the paper and model card.
