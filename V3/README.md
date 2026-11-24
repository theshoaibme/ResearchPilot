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

`PanOrgan 3 Cohort` is a multi-modal, multi-task medical imaging model for joint anatomical segmentation and diagnostic classification across heterogeneous clinical cohorts. Version `3.1` unifies all five imaging modalities (`MRI`, `CT`, `CXR`, `US`, `MG`) under a shared backbone conditioned on clinical ontology text prompts, with MixUp regularization, temperature scaling, and conformal prediction support.

## Architecture Summary

- **Modality Harmonization (`OmniModalityStem`):** Normalizes heterogeneous 2D/3D inputs across `MRI`, `CT`, `CXR`, `US`, and `MG` into a unified token representation.
- **Clinical Prompt Conditioning (`OntologyPromptEncoder`):** Encodes structured clinical prompts (organ, modality, target task) to condition downstream predictions. This module accepts clinical prompt strings rather than conversational chat.
- **Joint Heads:** Shared feature representation feeding a dense segmentation decoder and a multi-class/multi-label classification head.
- **Uncertainty & Calibration:** Post-hoc temperature scaling (`T = 0.9390` in Release `3`; `T = 1.0354` in Release `3.1`) and Marginal Conformal Calibration (`MC2`) targeting 90% coverage.
- **Checkpoint Footprint:** Release `3` calibrated weights occupy `240 MB`; Release `3.1` 5-modality MixUp weights (`best.pt`) occupy `715 MB`.

## Training Cohorts (11,735 Samples)

All splits (`8,078 train / 1,839 val / 1,818 test`) are strictly grouped by `patient_id` (`leakage_count = 0`) to prevent patient-level data leakage. Only verified clinical labels from official sources are used.

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

## Evaluation Results (Version 3.1 Held-Out Test Set)

All metrics below are measured on the held-out `test` split (`n = 1,629` classification across all 5 modalities, `n = 1,246` segmentation) after 5 epochs of MixUp fine-tuning (`train_loss = 2.4089`, `val_loss = 3.8846`, `val_dice = 0.8404`, `val_f1 = 0.8938`, `val_acc = 87.15%`).

### Classification & Calibration (`n = 1,629`)

| Metric | Uncalibrated | Temperature Scaled (`T = 1.0354`) |
| :--- | :---: | :---: |
| ROC-AUC | 0.9590 | 0.9593 |
| Accuracy | 85.57% | 85.57% |
| Macro-F1 | 0.8814 | 0.8814 |
| Macro Sensitivity (Recall) | 0.8802 | 0.8802 |
| Macro Specificity | 0.9358 | 0.9358 |
| Macro Precision | 0.8832 | 0.8832 |
| Expected Calibration Error (ECE) | 0.0611 | 0.0784 |
| Brier Score | 0.2483 | 0.2499 |

### Segmentation (Mean with 95% Bootstrap CI, `n = 1,246`)

| Cohort | Samples | Dice | IoU | HD95 (mm) |
| :--- | :---: | :---: | :---: | :---: |
| `pulmonology_cxr` | 334 | 0.9664 | 0.9366 | 8.24 |
| `tcga_lgg` | 537 | 0.8901 | 0.8560 | 6.13 |
| `busi` | 109 | 0.7677 | 0.6871 | 37.30 |
| `ultrasound_nerve` | 266 | 0.6402 | 0.6028 | 10.91 |
| **Overall** | **1,246** | **0.8465** | **0.8088** | **10.30** |

### Conformal Coverage (`MC2`, `alpha = 0.10`)

- **Overall:** Marginal coverage `92.51%`, average prediction set size `1.7397`, abstention rate `29.59%`.
- **BUSI Improvement:** Breast ultrasound abstention dropped from `30.28%` (Release `3`) to `11.01%` (Release `3.1`) at `91.74%` coverage.

## Version Progression Summary

1. **Stage 1 (Self-Supervised Pretraining):** Masked anatomical reconstruction (`best_val_loss = 0.003693`).
2. **Stage 2A (Joint Multi-Task Training):** Joint segmentation and classification (`val_dice = 0.8350`, `val_f1 = 0.9009`, `val_acc = 88.90%`).
3. **Stage 2B (Release 3 Calibration):** Head calibration (`val_f1 = 0.9017`, `val_acc = 89.18%`, `test AUROC = 0.9715`, `ECE = 0.0598`, `Overall Dice = 0.8355`).
4. **Release 3.1 (5-Modality MixUp Upgrade):** Added official `tcga_kirc_ct` (`CT`) and `cbis_ddsm` (`MG`) labels with `MixUp` (`alpha = 0.2`, `p = 0.25`) and explicit `val_loss` tracking (`Overall Dice = 0.8465`, `test AUROC = 0.9593`, `Macro-F1 = 0.8814`).

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
- **Sample Size Imbalance on New Modalities:** `cbis_ddsm` (`88` samples, `15` test) and `tcga_kirc_ct` (`15` unique patients, `189` test slices) exhibit `100%` conformal abstention in Release `3.1`, indicating that larger patient pools are needed for `CT` and `MG`.
- **Ultrasound Boundary Variance:** `ultrasound_nerve` (`Dice = 0.6402`) and `busi` (`HD95 = 37.30 mm`) remain more challenging than projection X-ray (`pulmonology_cxr`) and brain MRI (`tcga_lgg`).
