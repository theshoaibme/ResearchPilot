# FYDP (Title Phase) Evaluation Report (Version 2.0)

## Project Overview

- **Project Title:** The Pan-Organ Diagnostic Paradigm: A High-Capacity Foundation Model for Multi-Modality Medical Screening

---

## Core Summary

### 1. Problem Statement & Motivation
* **Existing Limitations:** Traditional AI diagnostic models are typically single-organ, single-disease, or single-modality frameworks. They fail to replicate real-world clinical workflows where radiologists must simultaneously evaluate multiple organ systems across various imaging types.
* **Core Challenges Addressed:**
  - Fragmented, narrow diagnostic AI models.
  - Lack of effective multi-modal data fusion strategies.
  - Absence of a unified pan-organ screening framework.
  - Weak AI interpretability and transparency.
  - Scalability constraints across diverse patient populations and imaging protocols.

### 2. Project Objectives
1. Conduct a comprehensive literature review and analyze problem gaps in medical AI.
2. Acquire, curate, and preprocess multi-modal imaging datasets (MRI, CT, X-ray) across multiple organ systems.
3. Design a high-capacity deep learning foundation model for simultaneous multi-organ screening.
4. Implement multi-modal data fusion strategies.
5. Integrate Explainable AI (XAI) using Gradient-Weighted Class Activation Mapping (Grad-CAM) to generate visual heatmaps for clinical transparency.
6. Train, optimize, and validate the model against standard clinical benchmarks (Accuracy, Sensitivity, Specificity, AUC-ROC).

### 3. Key Novel Contributions
* **Pan-Organ Architecture:** Performs multi-organ screening within a single end-to-end inference pipeline.
* **Multi-Modal Data Fusion:** Merges imaging data across MRI, CT, and X-ray modalities.
* **Broad Disease Taxonomy:** Covers neurodegenerative, neoplastic, vascular, demyelinating, infectious, traumatic, epileptic, and autoimmune conditions.
* **Explainability:** Generates localized visual explanations to support clinical decision-making.

### 4. Scope & Limitations
* **In Scope:** Model architecture design, data fusion, pretraining/transfer learning, XAI integration, and benchmark performance evaluation.
* **Limitations:**
  - Dependent on public multi-modal dataset availability.
  - Rare disease coverage may be limited due to sparse data.
  - Validation is restricted to offline benchmark datasets (no real-time live hospital deployment).

### 5. Supervisor Feedback & Completed Phases Execution

Handwritten feedback notes suggested focusing next steps on 5 key phases, which have now been systematically executed and documented:

#### Phase 1: Literature Review (Completed)
* Synthesized state-of-the-art architectures (3D U-Net, BiomedCLIP, TotalSegmentator, Pan-FM).
* Identified key baseline performance standards and established comparative evaluation criteria.
* Documented in: [Phase_1_Literature_Review.md](phases/Phase_1_Literature_Review.md)

#### Phase 2: Dataset Collection (Completed)
* Cataloged multi-modal, multi-organ datasets: **TotalSegmentator** (1,204 CT volumes / 117 organs), **MIMIC-CXR** (377,110 chest X-rays), and **TCIA** (~120,000 multi-parametric MRI & CT series).
* Documented in: [Phase_2_Dataset_Collection.md](phases/Phase_2_Dataset_Collection.md)

#### Phase 3: Data Preprocessing (Completed)
* Specified isotropic voxel spatial resampling ($1.5\text{mm} \times 1.5\text{mm} \times 1.5\text{mm}$).
* Formulated intensity normalization protocols (Hounsfield Unit windowing for CT, Z-score scaling for MRI).
* Documented in: [Phase_3_Data_Preprocessing.md](phases/Phase_3_Data_Preprocessing.md)

#### Phase 4: Gaps Finding (Completed)
* Identified **Dominant-Organ Shortcutting** in standard MAE random masking.
* Identified **Missing Not at Random (MNAR) Vulnerability** during partial scan inputs.
* Identified **Actionability Disconnect** in standard probabilistic outputs.
* Documented in: [Phase_4_Gaps_Finding.md](phases/Phase_4_Gaps_Finding.md)

#### Phase 5: Method Development (Completed)
* Formulated **Modality-Aware Tokenizer** with physical voxel spacing meta-embeddings.
* Designed **Saliency-Guided Masking (SGM)** to retain high-entropy anatomical feature tokens.
* Integrated **Grad-CAM Explainability** and **Action Planning Heads** for clinical transparency.
* Documented in: [Phase_5_Method_Development.md](phases/Phase_5_Method_Development.md)
