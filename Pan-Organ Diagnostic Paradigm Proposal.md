# The Pan-Organ Diagnostic Paradigm: A High-Capacity Foundation Model for Multi-Modality Medical Screening

**Daffodil International University**  
**Department of Computer Science and Engineering**  
**Course Code:** CSE499	**Title:** FYDP (Title Defense)

**Proposal Approval Form**

| 1st Member information |  |  |  |
| ----- | :---- | :---- | :---- |
| **Student ID:** | **0242310005101484** | **Student Name:** | **MD Shoaib Khan** |
| **2nd Member information**  |  |  |  |
| **Student ID:** | **0242310005101412** | **Student Name:** | **Asmita Rahman** |
| **Semester:** | **Spring 2026** | **Date:** | **18.02.2026** |
| **III. Proposed Title:** | **The Pan-Organ Diagnostic Paradigm: A High-Capacity Foundation Model for Multi-Modality Medical Screening** |  |  |
| **IV. Updated Title (**If there is any update**):** |  |  |  |
|  This is the approval for the Final Year Design Project (FYDP) proposal and title submitted by the above-mentioned student who is from the Department of CSE under my supervision. Upon thorough review and evaluation, it is my professional opinion that the **proposed project title *\[mentioned in Sec III and IV\]*** is both relevant and aligned with the academic goals and standards of our department. I (Supervisor) have endorsed the proposed proposal and title. Under our department's guidelines, the student has fulfilled all the necessary prerequisites and will attach this Proposal Approval Form to their FYDP Title Phase Evaluation Report. |  |  |  |
| **Supervisor Name:**  | **Dr. Md. Ali Hossain** |  |  |
| **Supervisor Designation:**  | **Associate Professor**   |  |  |
| **Approved by:  \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ Supervisor Signature with Date** |  |  |  |

# **V. Detailed Proposal**

# **1. Objective**

The primary objective is to develop **"Pan-Organ Net,"** a high-capacity, multi-modal transformer-based foundation model. Unlike traditional clinical models restricted to single-modality pixels, this project integrates diverse data sources—specifically **multi-modality medical imaging** (CT, MRI, Ultrasound, and X-ray), **unstructured clinical notes/text**, and **structured laboratory reports**—into a unified embedding space. By cross-referencing systemic structural and physiological data, the model delivers zero-shot and few-shot diagnostic predictions and automatically generates actionable, next-step clinical recommendation reports across the whole human body.

# **2. Motivation**

Current clinical AI is plagued by the "One Model, One Task" limitation, which fails to capture a patient's overall health profile. Crucial parameters from blood tests or physical summaries are routinely isolated from visual exams. By developing a multi-modal pan-organ paradigm, we can:

* **Capture Complete Patient Health Profiles:** Cross-reference lab values (e.g., elevated creatinine) with imaging features (e.g., renal masses) to avoid localized diagnostic errors.
* **Reduce Data Scarcity:** Leverage self-supervised learning on massive unlabeled multi-modal datasets to assist in diagnosing rare diseases where labeled data is minimal.  
* **Enhance Diagnostic Accuracy:** Allow the model to learn "universal" anatomical features and clinical correlations, improving its ability to detect subtle systemic anomalies that span across organ boundaries.

# **3. Background Study**

Recent literature highlights a shift from supervised CNNs to **Self-Supervised Learning (SSL)** using Vision Transformers (ViTs) and Vision-Language Models (VLMs). While models like *RadImageNet* or *BiomedCLIP* have made strides, they often remain modality-specific or lack the "high-capacity" required for simultaneous multi-organ segmentation, classification, and text integration. This research aims to fill the gap by implementing a **Multi-Modal Masked Autoencoder (MAE)** framework that treats medical imaging, clinical text, and structured lab records as a unified language.

# **4. Gap Analysis**

Existing solutions suffer from four primary gaps:

1. **Modality & Text Isolation (Non-Unified Latent Space):** Most models cannot align a CT scan, an ultrasound, clinical summaries, and laboratory panels simultaneously without separate, non-aligned networks.
2. **Missingness Robustness & Shortcut Learning:** Clinical records are frequently incomplete. A patient might have a chest CT and lab tests, but no abdominal MRI. Standard models suffer from *dominant-organ shortcut learning* (over-indexing on highly available organs at the expense of others).
3. **Domain Shift & Spectral Collapse:** When training models on scarce or heterogeneous medical subsets, domain differences (variations in scanner parameters) and finite-sample noise often collapse the covariance of the representation space, reducing downstream zero-shot capability.
4. **The Actionability Gap (The "Detection-Action" Disconnect):** Modern models excel at identifying an anomaly (e.g., "Lung nodule detected") but do not provide clinical decision-support or next-step recommendation pathways (e.g., suggesting a biopsy, predicting prognosis, or highlighting specific lab tests).

# **5. Methodology**

We will employ a **State-of-the-Art (SOTA) Multi-Modal Foundation Model Architecture** based on the 3D Swin Transformer (for visual patches) and a transformer-based text encoder (for clinical notes and laboratory panels).

* **Techniques/Tools:** We will utilize **PyTorch** for model development, **Hugging Face Accelerate** for distributed training, and **Saliency-Guided Masked Autoencoders (SGM)** as the primary self-supervised pretext task to prevent dominant-organ shortcutting. The latent space will be aligned via a dual-objective loss function combining structural reconstruction loss and contrastive semantic alignment across text, labs, and pixels.
* **Data Collection:** Data will be aggregated from large-scale public repositories including **TCIA (The Cancer Imaging Archive)**, **MIMIC-CXR** (paired with clinical reports), and **TotalSegmentator**, ensuring a dataset size of 500,000+ images across 10+ organ systems. Pre-processing will involve isotropic voxel resampling to $1.5\text{mm}^3$ and Z-score intensity normalization.
* **Data Analysis:** Evaluation will involve **Linear Probing** and **Full Fine-Tuning** benchmarks. Key metrics include:  
  * **Dice Similarity Coefficient (DSC)** for segmentation accuracy.  
  * **Hausdorff Distance (HD95)** to ensure surgical boundary precision.
  * **Area Under the ROC Curve (AUC-ROC)** for multi-class diagnostic classification.  
  * **Robustness Evaluations** under simulated missing-organ scenarios (MNAR).

# **6. Expected Outcomes**

**A Pre-trained Multi-Modal Foundation Model:** A high-capacity weight file ready for downstream clinical fine-tuning, capable of processing images, notes, and lab reports.

**State-of-the-Art Performance:** Outperforming narrow-AI models in at least three organ-specific tasks (e.g., lung nodule detection, liver segmentation, and cardiac pathology classification) even under simulated missing-data environments.

**Clinical Decision Actionability:** Decoupled Action Planning heads that output structured diagnostic pathways, automated diagnostic reports, and next-step clinical recommendations (e.g., suggesting follow-up scans or specific biopsy targets).

**Open-Source Contribution:** A GitHub repository containing the pre-processing pipelines, pre-trained weights, and the training code to foster further academic research in the CSE department.

# **7. Conclusion**

The "Pan-Organ Diagnostic Paradigm" represents a shift from reactive, task-specific AI to **proactive, general-purpose medical intelligence**. By consolidating multi-modality screening, clinical notes, and lab reports into a single high-capacity model with SGM robustness and actionable heads, this project not only meets the academic rigor of the CSE499 course but also provides a scalable solution to global healthcare challenges.