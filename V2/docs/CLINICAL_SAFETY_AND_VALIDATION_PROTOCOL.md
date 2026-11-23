# Clinical Safety, Engineering Safeguards, and Validation Protocol (V2)

Training machine learning models for medical imaging introduces distinct safety, statistical, and clinical validity requirements that do not arise in standard natural-image benchmarks. Every component of `Pan-Organ Net V2` is engineered around **10 mandatory clinical safeguards**.

---

## The 10 Clinical ML Safeguards Implemented in V2

### 1. Strict Patient-Level Isolation (Zero Data Leakage)
- **Risk:** Splitting volumetric slices or repeated patient exams randomly across train and test sets allows deep networks to memorize patient-specific body habitus or scanner noise, inflating test accuracy by 10–25%.
- **V2 Enforcement:** `build_patient_splits()` uses official patient splits (`NIH ChestX-ray14`, `CBIS-DDSM`) or patient-grouped `StratifiedGroupKFold` (`AMOS22`, `BUSI`, `CAMUS`, `TCGA-LGG`, `Ultrasound Nerve`). `verify_zero_patient_overlap()` asserts zero intersection across splits and emits `split_audit_report.json`.

### 2. Authentic Clinical Labels & Hospital-Shortcut Resistance
- **Risk:** Proxy tasks (e.g., scanner modality classification) or multi-source hospital merges (e.g., merging normal X-rays from pediatric hospitals with diseased X-rays from adult ICUs) cause models to learn hospital artifacts instead of pathology (DeGrave et al., 2021).
- **V2 Enforcement:** Every dataset uses verified clinical pathology or expert-drawn anatomical/tumor masks. Cross-hospital generalization is explicitly measured on an untouched external validation cohort (`RSNA Pneumonia` for CXR; `AMOS22 MRI` cross-modality for abdominal segmentation).

### 3. Clinical Cost Asymmetry (Minimizing False Negatives)
- **Risk:** In oncology and acute screening, a False Negative (missed malignancy or pneumonia) carries severe clinical harm compared to a False Positive recall. Plain accuracy is misleading under class imbalance.
- **V2 Enforcement:** Classification uses cost-sensitive **Asymmetric Focal Loss** and **Focal CE**; segmentation combines **Soft Dice + Focal + Boundary L1**. Reporting prioritizes **Sensitivity at 90% and 95% Specificity**, **Specificity**, **PPV/NPV**, **AUPRC**, **AUROC**, **Macro-F1**, **3D Dice**, and **physical-mm 95th-percentile Hausdorff Distance (HD95)**.

### 4. Modality-Specific Physics Preservation
- **Risk:** Generic 8-bit rescaling destroys quantitative CT Hounsfield Units, flattens soft-tissue MRI contrast, and turns Ultrasound acoustic speckle into dominant false edges.
- **V2 Enforcement:** `panorgan/data/harmonizer.py` applies multi-window CT HU normalization (`[-150,250]`, `[-1000,200]`, `[200,1000]`), foreground-masked MRI Z-score clipping, 16-bit safe X-Ray/Mammography scaling, and median-despeckled Ultrasound structural entropy, while conditioning the network on physical voxel spacing via `PhysicsFiLM`.

### 5. Balanced Multi-Modality Optimization (`MT-MoE` + Temperature Sampling)
- **Risk:** Large 2D datasets (112K CXRs) overwhelm smaller volumetric cohorts (500 CTs), while competing task gradients degrade individual organ performance.
- **V2 Enforcement:** `BalancedModalityBatchSampler` samples datasets proportional to $N_d^{0.5}$, paired with **Sparse Top-2 `MT-MoE`** routing to decouple modality- and task-specific feature pathways.

### 6. Probability Calibration & Selective Abstention (`MC2-CP`)
- **Risk:** Uncalibrated deep classifiers output overconfident probabilities on ambiguous or out-of-distribution clinical scans.
- **V2 Enforcement:** Post-hoc **Temperature Scaling** minimizes validation NLL (quantified via **ECE** and **Brier Score**), wrapped with **Modality-Conditional Conformal Prediction (`MC2-CP`)** providing finite-sample $(1-\alpha)$ coverage guarantees and explicit abstention rates (`|C(x)| > 1`) flagging cases for radiologist escalation.

### 7. Quantitative Explainability Faithfulness (Beyond Visual Heatmaps)
- **Risk:** Visual saliency maps can appear subjectively convincing while failing to reflect the true features driving predictions.
- **V2 Enforcement:** Every visual explanation (Grad-CAM++ / feature-norm attribution) is quantitatively verified via **Deletion AUC** (confidence must fall steeply when highlighted lesion regions are occluded) and **Insertion AUC** (confidence must rise steeply when highlighted regions are revealed).

### 8. Missing-Not-at-Random (MNAR) Partial-FOV Stress Testing
- **Risk:** Real hospital scans are frequently truncated to target regions rather than standardized whole-torso views.
- **V2 Enforcement:** Models are systematically stress-tested at **0%, 20%, 40%, and 50%** contiguous spatial FOV truncation, reporting measured relative degradation:
  $$\Delta_{\text{MNAR}} = \frac{\text{Metric}_{\text{Full}} - \text{Metric}_{\text{Truncated}}}{\text{Metric}_{\text{Full}}} \times 100\%$$

### 9. Statistical Significance & Uncertainty Intervals
- **Risk:** Claiming superiority from single scalar test numbers without uncertainty bounds or hypothesis testing fails peer review.
- **V2 Enforcement:** Every reported metric includes **1,000-iteration patient-level bootstrap 95% confidence intervals**. Comparisons against identical-split baselines (`DenseNet121`, `ConvNeXt-S`, `UNet-2.5D`) and ablations are verified via **DeLong's test** (AUROC), **Wilcoxon signed-rank test** (patient-level Dice & HD95), and **McNemar's test** (binary diagnostic accuracy).

### 10. Privacy, Governance, and Clinical Deployment Scope
- **Data Governance:** Exclusively de-identified public cohorts with documented licenses; `strip_dicom_phi()` removes direct patient identifiers in memory whenever DICOM files are parsed.
- **Clinical Scope:** Framed strictly as an **assistive Clinical Decision Support (CDS)** and triage/pre-segmentation system—not an autonomous diagnostic substitute—benchmarked with per-case inference latency (ms) and peak GPU VRAM (GB).

---

## Controlled Baseline & Ablation Experimental Matrix

All models are evaluated on **identical held-out patient splits**:

| Experiment ID | Configuration | Scientific Hypothesis Tested |
|---|---|---|
| `B1` | Single-Task `DenseNet121` (2D Cls) | Narrow CNN classification baseline |
| `B2` | Single-Task `ConvNeXt-Small` (2D Cls) | Modern matched-backbone narrow classification baseline |
| `B3` | Single-Task `UNet-2.5D` (Seg) | Narrow segmentation baseline across CT/MRI/US |
| `A1` | `Pan-Organ Net V2` (Random Masking) | Effect of `MA-UGM` vs. standard uniform MAE masking |
| `A2` | `Pan-Organ Net V2` (Gradient-Only SGM) | Effect of modality-adaptive despeckling + EMA teacher vs. raw gradients |
| `A3` | `Pan-Organ Net V2` (w/o `MT-MoE`) | Effect of Sparse Top-2 MoE on cross-modality gradient conflict |
| `A4` | `Pan-Organ Net V2` (w/o `SACR`) | Effect of `SACR` coordinate anchoring under 0–50% MNAR FOV truncation |
| `FULL` | **`Pan-Organ Net V2` (Complete + `MC2-CP`)** | Full unified model + conformal calibration & external validation |
