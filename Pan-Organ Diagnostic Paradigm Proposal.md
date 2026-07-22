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

The primary objective is to develop **"Pan-Organ Net,"** a high-capacity, transformer-based foundation model capable of performing zero-shot and few-shot diagnostic tasks across multiple human organs and imaging modalities (MRI, CT, X-ray, and Ultrasound). Unlike traditional "narrow" AI that focuses on a single pathology, this project aims to solve the problem of **model fragmentation** in clinical settings by creating a unified feature extractor that generalizes across diverse anatomical structures.

# **2. Motivation**

Current clinical AI is plagued by the "One Model, One Task" limitation, which is computationally expensive and difficult to integrate into hospital workflows. By developing a pan-organ paradigm, we can:

* **Reduce Data Scarcity:** Leverage self-supervised learning on massive unlabeled datasets to assist in diagnosing rare diseases where labeled data is minimal.  
* **Enhance Diagnostic Accuracy:** Allow the model to learn "universal" anatomical features, improving its ability to detect subtle anomalies that cross-organ boundaries.

# **3. Background Study**

Recent literature highlights a shift from supervised CNNs to **Self-Supervised Learning (SSL)** using Vision Transformers (ViTs). While models like *RadImageNet* or *BiomedCLIP* have made strides, they often remain modality-specific or lack the "high-capacity" required for simultaneous multi-organ segmentation and classification. This research aims to fill the gap by implementing a **Multi-Modal Masked Autoencoder (MAE)** framework that treats medical imaging as a unified language.

# **4. Gap Analysis**

Existing solutions suffer from three primary gaps:

1. **Modality Isolation & Non-Unified Latent Space:** Most current models process different modalities (e.g., CT vs. Ultrasound) via entirely separate pathways, failing to align the semantic anatomical representation in a unified latent space.
2. **Missingness Robustness & Shortcut Learning:** In clinical practice, patient records are incomplete. A patient might have a chest CT but no abdominal MRI. Standard models suffer from *dominant-organ shortcut learning* (over-indexing on highly available organs at the expense of others).
3. **Domain Shift & Spectral Collapse:** When training models on scarce or heterogeneous medical subsets, domain differences (variations in scanner parameters) and finite-sample noise often collapse the covariance of the representation space, reducing downstream zero-shot capability.
4. **The Actionability Gap (The "Detection-Action" Disconnect):** Modern models excel at identifying an anomaly (e.g., "Lung nodule detected") but do not provide clinical decision-support or next-step recommendation pathways (e.g., suggesting a biopsy, predicting prognosis, or highlighting specific lab tests).

# **5. Methodology**

We will employ a **State-of-the-Art (SOTA) Foundation Model Architecture** based on the 3D Swin Transformer framework to ensure high spatial resolution across organ boundaries.

* **Techniques/Tools:** We will utilize **PyTorch** for model development, **Hugging Face Accelerate** for distributed training, and **Saliency-Guided Masked Autoencoders (SGM)** as the primary self-supervised pretext task to prevent dominant-organ shortcutting. The latent space will be aligned via a dual-objective loss function combining structural reconstruction loss and contrastive semantic alignment.
* **Data Collection:** Data will be aggregated from large-scale public repositories including **TCIA (The Cancer Imaging Archive)**, **MIMIC-CXR**, and **TotalSegmentator**, ensuring a dataset size of 500,000+ images across 10+ organ systems. Pre-processing will involve isotropic voxel resampling to $1.5\text{mm}^3$ and Z-score intensity normalization.
* **Data Analysis:** Evaluation will involve **Linear Probing** and **Full Fine-Tuning** benchmarks. Key metrics include:  
  * **Dice Similarity Coefficient (DSC)** for segmentation accuracy.  
  * **Hausdorff Distance (HD95)** to ensure surgical boundary precision.
  * **Area Under the ROC Curve (AUC-ROC)** for multi-class diagnostic classification.  
  * **Robustness Evaluations** under simulated missing-organ scenarios (MNAR).

# **6. Expected Outcomes**

**A Pre-trained Foundation Model:** A high-capacity weight file ready for downstream clinical fine-tuning.

**State-of-the-Art Performance:** Outperforming narrow-AI models in at least three organ-specific tasks (e.g., lung nodule detection, liver segmentation, and cardiac pathology classification) even under simulated missing-data environments.

**Clinical Decision Actionability:** Decoupled Action Planning heads that output structured diagnostic pathways and next-step clinical recommendations.

**Open-Source Contribution:** A GitHub repository containing the pre-processing pipelines, pre-trained weights, and the training code to foster further academic research in the CSE department.

# **7. Conclusion**

The "Pan-Organ Diagnostic Paradigm" represents a shift from reactive, task-specific AI to **proactive, general-purpose medical intelligence**. By consolidating multi-modality screening into a single high-capacity model with SGM robustness and actionable heads, this project not only meets the academic rigor of the CSE499 course but also provides a scalable solution to global healthcare challenges.