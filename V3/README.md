---
license: apache-2.0
library_name: pytorch
tags:
  - medical-imaging
  - multimodal
  - segmentation
  - classification
  - conformal-prediction
  - foundation-model
datasets:
  - tcga-lgg
  - tcga-kirc
  - nih-chest-xray14
  - cbis-ddsm
  - busi
  - prostatex
---

# PanOrgan 3 Cohort

`PanOrgan 3 Cohort` is a multi-modal, multi-task medical imaging model for joint anatomical segmentation and diagnostic classification across heterogeneous clinical cohorts. It unifies five imaging modalities (`MRI`, `CT`, `CXR`, `US`, `MG`) under a shared backbone conditioned on clinical ontology text prompts, with post-hoc temperature scaling and conformal prediction support.

## Architecture Summary

- **Modality Harmonization (`OmniModalityStem`):** Normalizes heterogeneous 2D/3D inputs across `MRI`, `CT`, `CXR`, `US`, and `MG` into a unified token representation.
- **Clinical Prompt Conditioning (`OntologyPromptEncoder`):** Encodes structured clinical prompts (organ, modality, target task) to condition downstream predictions. This module accepts clinical prompt strings rather than conversational chat.
- **Joint Heads:** Shared feature representation feeding a dense segmentation decoder and a multi-class/multi-label classification head.
- **Uncertainty & Calibration:** Post-hoc temperature scaling (`T = 0.9390` in release `3`) and Mondrian/Marginal Conformal Calibration (`MC2`) targeting 90% coverage.
- **Checkpoint Footprint:** Final calibrated release weights (`best.pt`) occupy `240 MB` (joint stage checkpoint: `715 MB`).

## Training Cohorts (11,735 Samples)

All splits (`8,078 train / 1,839 val / 1,818 test`) are strictly grouped by `patient_id` to prevent patient-level data leakage. Only verified clinical labels from official sources are used.

| Cohort | Modality | Samples | Patients | Supervised Labels | Primary Task |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `tcga_lgg` | MRI | 3,929 | 110 | 3,929 | Brain tumor joint segmentation and classification |
| `pulmonology_cxr` | CXR | 2,000 | 2,000 | 2,000 | Lung segmentation and pulmonary classification |
| `ultrasound_nerve` | US | 1,500 | 47 | 1,500 | Brachial plexus nerve segmentation and presence |
| `tcga_kirc_ct` | CT | 1,200 | 15 | 1,200 | Renal grade classification (`G1/G2` vs `G3/G4`, NCI GDC) |
| `prostatex_mri` | MRI | 1,198 | 25 | 0 | Unlabeled self-supervised representation learning |
| `nih_cxr14` | CXR | 1,167 | 882 | 1,167 | Multi-label thoracic pathology classification |
| `busi` | US | 653 | 653 | 653 | Breast ultrasound lesion segmentation and classification |
| `cbis_ddsm` | MG | 88 | 45 | 88 | Breast mammography pathology (`BENIGN` vs `MALIGNANT`) |

## Evaluation Results (Release 3 Test Set)

All metrics below are measured on the held-out `test` split (`n = 1,425` classification, `n = 1,246` segmentation) prior to the `3.1` MixUp and 5-modality fine-tuning pass (currently running).

### Classification & Calibration

| Metric | Uncalibrated | Temperature Scaled (`T = 0.9390`) |
| :--- | :---: | :---: |
| ROC-AUC | 0.9715 | 0.9713 |
| Accuracy | 87.93% | 87.93% |
| Macro-F1 | 0.8905 | 0.8905 |
| Macro Sensitivity (Recall) | 88.79% | 88.79% |
| Macro Specificity | 94.42% | 94.42% |
| Macro Precision | 89.56% | 89.56% |
| Expected Calibration Error (ECE) | 0.0792 | 0.0598 |
| Brier Score | 0.2105 | 0.2077 |

### Segmentation (Mean with 95% Bootstrap CI)

| Cohort | Samples | Dice | IoU | HD95 (mm) |
| :--- | :---: | :---: | :---: | :---: |
| `pulmonology_cxr` | 334 | 0.9637 | 0.9316 | 9.61 |
| `tcga_lgg` | 537 | 0.8753 | 0.8413 | 6.90 |
| `busi` | 109 | 0.7790 | 0.6983 | 28.78 |
| `ultrasound_nerve` | 266 | 0.6175 | 0.5863 | 9.77 |
| **Overall** | **1,246** | **0.8355** | **0.7986** | **10.16** |

### Conformal Coverage & MNAR Missingness Stress Test

- **Conformal Prediction (`alpha = 0.10`):** Marginal coverage `93.33%`, average prediction set size `1.5277`, abstention rate `20.98%`.
- **MNAR Stress Curve (Overall Dice):** `0.8355` (0% missing) -> `0.8380` (20% missing) -> `0.7908` (40% missing).

## Training & Version Progression

1. **Stage 1 (Self-Supervised Pretraining):** Masked anatomical reconstruction across labeled and unlabeled cohorts (`best_val_loss = 0.003693`).
2. **Stage 2A (Joint Multi-Task Training):** Joint segmentation and classification optimization (`val_dice = 0.8350`, `val_f1 = 0.9009`, `val_acc = 88.90%`).
3. **Stage 2B (Head Calibration):** Classification head refinement (`val_f1 = 0.9017`, `val_acc = 89.18%`) followed by temperature calibration.
4. **Release 3.1 (In Progress):** Warm-started fine-tuning integrating official `tcga_kirc_ct` (NCI GDC) and `cbis_ddsm` (CBIS-DDSM) supervised labels across all 5 modalities with `MixUp` (`alpha = 0.2`, `p = 0.25`) and explicit validation loss tracking.

## Usage

```python
import torch

ckpt = torch.load("best.pt", map_location="cpu")
state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
```

To run inference or evaluation within this repository, pass clinical modality and organ prompts through the `OntologyPromptEncoder` pipeline defined in the project configuration.

## Limitations

- **Research Use Only:** Not cleared for clinical diagnosis or autonomous medical decision-making.
- **Unlabeled Cohort:** `prostatex_mri` (`1,198` samples) lacks public clinical findings labels without restricted TCIA challenge credentials and is therefore used strictly for self-supervised representation learning.
- **Ultrasound Boundary Variance:** `ultrasound_nerve` (`Dice = 0.6175`) and `busi` (`HD95 = 28.78 mm`) remain more challenging than projection X-ray (`pulmonology_cxr`) and brain MRI (`tcga_lgg`) due to acoustic speckle and operator-dependent probe geometry.
- **Sample Size Imbalance:** `cbis_ddsm` (`88` samples) and `tcga_kirc_ct` (`15` unique patients, `1,200` slices) represent smaller patient pools than `pulmonology_cxr` (`2,000` patients).
