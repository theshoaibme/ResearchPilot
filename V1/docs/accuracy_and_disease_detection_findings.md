# Pan-Organ Net: Accuracy & Disease Detection Summary

This document consolidates key diagnostic performance statistics, disease classifications, and segmentation accuracies for the **Pan-Organ Net** foundation model compared to task-specific architectures and baseline networks.

---

## 1. Classification and Segmentation Metrics
The model is evaluated using two primary clinical validation metrics:
*   **Area Under the ROC Curve (AUC):** Used to measure diagnostic classification accuracy for disease presence/absence.
*   **Dice Similarity Coefficient (DSC):** Used to measure spatial boundary overlap accuracy for organ and anomaly segmentations.

---

## 2. Disease Classification Accuracy
Evaluated on clinical benchmarks for brain and lung pathologies:

| Diagnostic Target | Dataset / Cohort | Task-Specific CNN | BiomedCLIP (2D) | Pan-FM Baseline | **Pan-Organ Net (Ours)** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Lung Nodule Detection** | LUNA16 | $81.2\%$ | $84.1\%$ | $88.7\%$ | **$91.2\%$** |
| **Brain Lesion Classification** | TCIA (LGG/GBM) | $78.8\%$ | $74.3\%$ | $86.5\%$ | **$90.4\%$** |

---

## 3. Structural & Anatomical Segmentation Accuracy
Evaluated on the TotalSegmentator dataset containing 117 organs and structural boundaries under simulated complete and partial body scan settings:

| Scan Completion | Missing Organ Ratio | Task-Specific CNN | Pan-FM Baseline | **Pan-Organ Net (Ours)** |
| :--- | :---: | :---: | :---: | :---: |
| **Complete Scan** | $0\%$ | $84.2\%$ | $86.4\%$ | **$89.8\%$** |
| **Moderate Partial Scan** | $20\%$ | $71.0\%$ | $78.1\%$ | **$88.2\%$** |
| **Extreme Partial Scan** | $50\%$ | $48.0\%$ | $61.2\%$ | **$85.4\%$** |

---

## 4. Key Performance Insights
1. **SGM-Driven Stability:** When $50\%$ of scans are omitted (extreme partial scans), standard random masking baselines collapse by **$25.3\%$**, dropping to $61.2\%$ segmentation accuracy. Pan-Organ Net retains **$85.4\%$** accuracy (dropping only $5.5\%$) due to Saliency-Guided Masking (SGM).
2. **Unified Representation:** Pan-Organ Net outperforms narrow CNNs and general 2D VLMs (BiomedCLIP) by projecting multiple scans (CT, MRI) and patient history into a single cohesive embedding space.

---

## 5. Systematic Disease Classification & Diagnostic Targets
Below is the classification of all clinical conditions, anomalies, and diseases covered by **Pan-Organ Net**'s multi-organ training cohort (mapped from MIMIC-CXR, TCIA, LUNA16, and 3DLAND):

### A. Neurological Pathologies (CNS)
*   **Target Lesions:** Lower-Grade Glioma (LGG), Glioblastoma Multiforme (GBM), and structural brain masses.
*   **Data Source:** TCIA Brain Cohorts.
*   **Diagnostic Objective:** Contrast-enhanced MRI segmentation and classification of tumor progression stages.

### B. Thoracic & Pulmonary Pathologies
*   **Target Lesions:** Benign/malignant pulmonary nodules, consolidations, ground-glass opacities, pneumonia, and pleural effusions.
*   **Data Sources:** LUNA16, MIMIC-CXR.
*   **Diagnostic Objective:** Detection and zero-shot categorization of nodule malignancy; multimodal radiographic diagnosis from paired X-ray + clinical notes.

### C. Abdominal & Gastrointestinal Anomalies
*   **Target Lesions:** Hepatic (liver) tumors/masses, renal (kidney) cysts and carcinomas (e.g., KIRC), splenomegaly, pancreatic lesions, gallbladder inflammation, and stomach tumors.
*   **Data Sources:** TCIA (TCGA-KIRC), 3DLAND, AMOS, TotalSegmentator.
*   **Diagnostic Objective:** High-precision boundary segmentation (Dice/HD95) for surgical margin delineation; multi-organ lesion correlation.

### D. Cardiovascular & Systemic Anomalies
*   **Target Lesions:** Cardiac chamber enlargement, coronary calcification, aortic aneurysm/dissection, and thoracic lymphatic metastases.
*   **Data Sources:** TotalSegmentator, MIMIC-CXR.
*   **Diagnostic Objective:** Anatomical alignment, vascular integrity mapping, and metastatic lymph node staging.

