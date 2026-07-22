**Proposed Title:** | **The Pan-Organ Diagnostic Paradigm: A High-Capacity Foundation Model for Multi-Modality Medical Screening** |  |  |
| **IV. Updated Title (**If there is any update**):** |  |  |  |
|  This is the approval for the Final Year Design Project (FYDP) proposal and title submitted by the above-mentioned student who is from the Department of CSE under my supervision. Upon thorough review and evaluation, it is my professional opinion that the **proposed project title *\[mentioned in Sec III and IV\]*** is both relevant and aligned with the academic goals and standards of our department. I (Supervisor) have endorsed the proposed proposal and title. Under our department's guidelines, the student has fulfilled all the necessary prerequisites and will attach this Proposal Approval Form to their FYDP Title Phase Evaluation Report. |  |  |  |
| **Supervisor Name:**  | **[Supervisor Name]** |  |  |
| **Supervisor Designation:**  | **Associate Professor**   |  |  |
| **Approved by:  \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ Supervisor Signature with Date** |  |  |  |

# **V. Detailed Proposal**

# **1\. Objective**

The primary objective is to develop **"Pan-Organ Net,"** a high-capacity, transformer-based foundation model capable of performing zero-shot and few-shot diagnostic tasks across multiple human organs and imaging modalities (MRI, CT, X-ray, and Ultrasound). Unlike traditional "narrow" AI that focuses on a single pathology, this project aims to solve the problem of **model fragmentation** in clinical settings by creating a unified feature extractor that generalizes across diverse anatomical structures.

# 

# **2\. Motivation**

Current clinical AI is plagued by the "One Model, One Task" limitation, which is computationally expensive and difficult to integrate into hospital workflows. By developing a pan-organ paradigm, we can:

* **Reduce Data Scarcity:** Leverage self-supervised learning on massive unlabeled datasets to assist in diagnosing rare diseases where labeled data is minimal.  
* **Enhance Diagnostic Accuracy:** Allow the model to learn "universal" anatomical features, improving its ability to detect subtle anomalies that cross-organ boundaries.

# **3\. Background Study**

Recent literature highlights a shift from supervised CNNs to **Self-Supervised Learning (SSL)** using Vision Transformers (ViTs). While models like *RadImageNet* or *BiomedCLIP* have made strides, they often remain modality-specific or lack the "high-capacity" required for simultaneous multi-organ segmentation and classification. This research aims to fill the gap by implementing a **Multi-Modal Masked Autoencoder (MAE)** framework that treats medical imaging as a unified language

# **4\. Gap Analysis**

Existing solutions suffer from three primary gaps:

1. **Modality Isolation:** Most models cannot process a CT scan and an Ultrasound simultaneously without retraining.  
2. **Domain Shift:** Models trained on one hospital's data often fail on another’s due to hardware variances.  
3. **High Computational Overhead:** Running 50 separate models for 50 different screenings is unsustainable for real-time diagnostics. Our project addresses these by using a **Unified Embedding Space**, allowing one model to handle varied inputs seamlessly.

.

# **5\. Methodology**

We will employ a **State-of-the-Art (SOTA) Foundation Model Architecture** based on the Swing Transformer or InternImage framework to ensure high spatial resolution across organ boundaries.

* **Techniques/Tools:** We will utilize **PyTorch** for model development, **Hugging Face Accelerate** for distributed training, and **Masked Image Modeling (MIM)** as the primary self-supervised pretext task.  
* **Data Collection:** Data will be aggregated from large-scale public repositories including **TCIA (The Cancer Imaging Archive)**, **MIMIC-CXR**, and **TotalSegmentator**, ensuring a dataset size of 500,000+ images across 10+ organ systems. Pre-processing will involve Z-score normalization and affine augmentations to handle modality variance.  
* **Data Analysis:** Evaluation will involve **Linear Probing** and **Full Fine-Tuning** benchmarks. Key metrics include:  
  * **Dice Similarity Coefficient (DSC)** for segmentation accuracy.  
  * **Area Under the ROC Curve (AUC-ROC)** for multi-class diagnostic classification.  
  * **Inference Latency** to ensure clinical viability.

# **6\. Expected Outcomes**

**A Pre-trained Foundation Model:** A high-capacity weight file ready for downstream clinical fine-tuning.

**State-of-the-Art Performance:** Outperforming narrow-AI models in at least three organ-specific tasks (e.g., Lung Nodule detection, Liver segmentation, and Cardiac arrhythmia screening).

**Open-Source Contribution:** A GitHub repository containing the training pipeline to foster further academic research in the CSE department.

# **7\. Conclusion**

The "Pan-Organ Diagnostic Paradigm" represents a shift from reactive, task-specific AI to **proactive, general-purpose medical intelligence**. By consolidating multi-modality screening into a single high-capacity model, this project not only meets the academic rigor of the CSE499 course but also provides a scalable solution to global healthcare challenges.