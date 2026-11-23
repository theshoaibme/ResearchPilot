# PanOrgan 2 — Technical Implementation & Benchmark Report

## 1. Overview & Objectives
`PanOrgan 2` transitioned the project from architectural design into a fully executable, reproducible medical AI pipeline. It introduced patient-level zero-leakage `.npz` sharding, external SSD storage routing (`/Volumes/Shoaib/PanOrgan_V2/`), multi-stage training on Apple M4 (`MPS`), temperature scaling, and conformal prediction evaluation across three core cohorts.

## 2. Architecture & Modeling
- **Modality Harmonizer (`OmniModalityStem`):** Standardized heterogeneous `MRI`, `CXR`, and `US` inputs into consistent spatial tokens.
- **Prompt Conditioning (`OntologyPromptEncoder`):** Added structured clinical prompt embeddings for organ and modality conditioning.
- **Multi-Task Heads:** Joint segmentation decoder and classification head trained end-to-end.
- **Uncertainty Quantification:** Added post-hoc temperature scaling and Marginal/Mondrian Conformal Calibration (`MC2`).

## 3. Datasets & Splitting Strategy
`PanOrgan 2` trained and evaluated on **3 verified cohorts** with strict patient-level zero-leakage splitting (`split_audit_report.json`):
- `tcga_lgg` (`MRI`, neurology oncology — `n = 537` test samples)
- `busi` (`US`, breast ultrasound — `n = 109` test samples)
- `nih_cxr14` (`CXR`, multi-label thoracic classification — `n = 179` test samples)
- **Total Evaluation Pool:** `n = 825` classification test samples, `n = 646` segmentation test samples.

## 4. Training & Calibration Protocol
- **Hardware & Precision:** Local Apple M4 GPU (`MPS`) with external SSD sharding and HuggingFace/PyTorch cache redirection.
- **Post-Hoc Calibration:** Optimized temperature parameter `T = 1.6201` on validation logits, followed by `MC2` conformal calibration targeting `90%` coverage (`alpha = 0.10`).

## 5. Verified Evaluation Metrics
Source: `/Volumes/Shoaib/PanOrgan_V2/experiments/thesis_reports/complete_thesis_report.json`

### Classification & Calibration (`n = 825`)
| Metric | Uncalibrated | Temperature Scaled (`T = 1.6201`) |
| :--- | :---: | :---: |
| AUROC | 0.9062 | 0.9002 |
| Accuracy | 69.82% | 69.82% |
| Macro-F1 | 0.6011 | 0.6011 |
| Macro Precision | 0.6020 | 0.6020 |
| Macro Sensitivity (Recall) | 0.7838 | 0.7838 |
| Macro Specificity | 0.8663 | 0.8663 |
| ECE | 0.1296 | 0.1609 |
| Brier Score | 0.4304 | 0.4208 |

### Segmentation (`n = 646`)
| Cohort | Samples | Dice (95% CI) | IoU | HD95 (mm) |
| :--- | :---: | :---: | :---: | :---: |
| `tcga_lgg` | 537 | 0.8081 (0.7829–0.8331) | 0.7653 | 10.62 |
| `busi` | 109 | 0.6356 (0.5805–0.6855) | 0.5211 | 53.11 |
| **Overall** | **646** | **0.7790 (0.7548–0.8034)** | **0.7241** | **17.74** |

### Conformal Coverage & MNAR Stress Test
- **MC2 Conformal Coverage:** Marginal coverage `91.03%`, average set size `1.4752`, abstention rate `28.61%` (`nih_cxr14` abstention `100%`).
- **MNAR Missingness Curve (Overall Dice):** `0.7790` (0%) -> `0.7886` (20%) -> `0.7589` (40%).

## 6. Known Limitations
1. **Moderate Classification F1 & Accuracy:** Macro-F1 (`0.6011`) and Accuracy (`69.82%`) lagged behind AUROC (`0.9062`) due to class imbalance and uncalibrated decision thresholds.
2. **Temperature Scaling ECE Regression:** Post-hoc temperature scaling (`T = 1.6201`) slightly worsened ECE (`0.1296` -> `0.1609`) and caused `100%` conformal abstention on `nih_cxr14`.
3. **High Ultrasound Boundary Error:** `busi` segmentation achieved `Dice = 0.6356` with `HD95 = 53.11 mm`.
4. **Limited Cohort & Modality Diversity:** Only 3 cohorts (`MRI`, `CXR`, `US`) were active; `CT` and `MG` were absent.
