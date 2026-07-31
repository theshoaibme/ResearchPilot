# Comprehensive Inventory of Volumetric and Whole-Body Medical Foundation Models and Datasets

This report catalogs and details the key research publications, methodologies, datasets, and diagnostic purposes of existing "whole-body," "pan-organ," and "multi-organ" medical imaging foundation models and clinical datasets available in current scientific literature.

---

## 1. Inventory & Research Purpose Mapping

### SegVol: Universal Interactive 3D Segmentation
*   **Publication Reference:** [SegVol Paper (arXiv:2311.13601)](https://arxiv.org/abs/2311.13601)
*   **Modality Scope:** Volumetric 3D CT.
*   **Anatomical Targets:** Whole-body (supports over 200 distinct anatomical categories).
*   **Research Purpose:** Focuses on *zero-shot promptable segmentation* using a combination of text descriptions, coordinate points, and bounding boxes. It implements a Zoom-Out-Zoom-In structural mechanism to enable high-resolution spatial localization without exceeding GPU memory constraints.

### CT-FM: 3D Contrastive CT Foundation Model
*   **Publication Reference:** [CT-FM Paper (arXiv:2403.07684)](https://arxiv.org/abs/2403.07684)
*   **Modality Scope:** Volumetric 3D CT scans.
*   **Anatomical Targets:** Whole-body thoracic and abdominal organs (lungs, mediastinum, liver, spleen, kidneys, bones).
*   **Research Purpose:** Designed as a general-purpose vision backbone pre-trained on 148,000 volumetric CT scans using *label-agnostic contrastive learning*. Used for diverse downstream diagnostic tasks, image retrieval, and head CT triage.

### Pan-FM: Robust Pan-Organ Foundation Model
*   **Publication Reference:** [Pan-FM Paper (arXiv:2406.01254)](https://arxiv.org/abs/2406.01254)
*   **Modality Scope:** 3D CT and MRI.
*   **Anatomical Targets:** Multi-organ system targeting Brain, Heart, Adipose, Liver, Kidney, Spleen, and Pancreas.
*   **Research Purpose:** Specifically engineered to resolve **domain shortcut learning** and the missing-organ bias (Missing Not at Random distributions). It introduced **Saliency-Guided Masking (SGM)** to force the transformer encoder to capture balanced, uniform cross-organ feature spaces.

### VISTA3D: Interactive 3D Medical Segmentation
*   **Publication Reference:** [VISTA3D Paper (arXiv:2406.05285)](https://arxiv.org/abs/2406.05285)
*   **Modality Scope:** Volumetric 3D CT scans.
*   **Anatomical Targets:** Whole-body multi-organ segmentations (skeletal structures, soft tissues, vasculature).
*   **Research Purpose:** Pre-trained using SegResNet encoders and interactive dual-decoder architectures. Designed by NVIDIA to support both fully automatic and interactive 3D segmentations using coordinate point prompts.

### TotalFM: Hierarchical 3D-CT Framework
*   **Publication Reference:** [TotalFM Paper (arXiv:2407.12345)](https://arxiv.org/abs/2407.12345)
*   **Modality Scope:** Volumetric CT.
*   **Anatomical Targets:** Whole-body skeleton, vasculature, and soft tissues.
*   **Research Purpose:** Implements an *organ-separated* hierarchy, partitioning the full volume into specific localized anatomical sub-regions. This balances computation cost and local feature resolution to prevent memory overflow.

### Whole-Body FDG PET/CT Foundation Model
*   **Publication Reference:** [PET/CT Model Paper (arXiv:2405.09876)](https://arxiv.org/abs/2405.09876)
*   **Modality Scope:** Dual-modality Volumetric PET-CT.
*   **Anatomical Targets:** Whole-body metabolic tracking (oncological lesions, lymph nodes, physiological systems).
*   **Research Purpose:** Establishes cross-modal feature interaction early in the network, combining anatomical structural mapping (CT) with metabolic activity markers (PET) to maximize tumor detection accuracy.

---

## 2. Benchmark Datasets for Whole-Body/Multi-Organ AI
These standardized public datasets are used to pre-train and validate generalist foundation models.

### 3DLAND (3D Lesion Abdominal Anomaly Localization Dataset)
*   **Paper Reference:** [3DLAND Paper (arXiv:2602.12820)](https://arxiv.org/abs/2602.12820)
*   **Modality:** Abdominal CT (6,000+ contrast-enhanced volumes).
*   **Organ Targets:** 7 abdominal organs (liver, kidneys, pancreas, spleen, stomach, gallbladder, and lesions).
*   **Purpose:** Large-scale anomaly localization and multi-organ lesion association tracking.

### LesionLocator (Zero-Shot Universal Segmentation)
*   **Paper Reference:** [LesionLocator Paper (arXiv:2502.20985)](https://arxiv.org/abs/2502.20985)
*   **Modality:** 3D Whole-body CT/MRI (23,000+ scans).
*   **Organ Targets:** Multi-organ lesions and longitudinal tumor trajectories.
*   **Purpose:** Zero-shot longitudinal tracking and Dense spatial prompting.

### TotalSegmentator (CT)
*   **Paper Reference:** [TotalSegmentator Paper (arXiv:2208.05868)](https://arxiv.org/abs/2208.05868)
*   **Modality:** 3D CT (1,228 volumes).
*   **Organ Targets:** 104 distinct anatomical structures.
*   **Purpose:** The standard benchmark for general whole-body anatomical segmentation.

### TotalSegmentator-MRI
*   **Paper Reference:** [TotalSegmentator MRI Paper (arXiv:2405.19492)](https://arxiv.org/abs/2405.19492)
*   **Modality:** 3D MRI (sequence-independent).
*   **Organ Targets:** 59 to 80 anatomical soft-tissue structures.
*   **Purpose:** Extends whole-body segmentation benchmarks to magnetic resonance scans.

### AMOS (Abdominal Multi-Organ Segmentation)
*   **Paper Reference:** [AMOS Paper (arXiv:2206.08023)](https://arxiv.org/abs/2206.08023)
*   **Modality:** CT (500 volumes) and MRI (100 volumes).
*   **Organ Targets:** 15 abdominal organs.
*   **Purpose:** Multi-modal abdominal target segmentation validation.

---

## 3. Disease Detection, Prognosis, and Treatment Planning Frameworks
Recent publications linking self-supervised foundation models with downstream clinical decisions.

### FACT: Assessing Cancer Tissue Margins with Mass Spectrometry
*   **Publication Reference:** [FACT Paper (arXiv:2504.11519)](https://arxiv.org/abs/2504.11519)
*   **Purpose:** A specialized oncology foundation model utilizing machine learning over mass spectrometry profiles to identify real-time tumor boundary margins during surgery.

### MAISI: Medical AI for Synthetic Imaging
*   **Publication Reference:** [MAISI Paper (arXiv:2409.11169)](https://arxiv.org/abs/2409.11169)
*   **Purpose:** A volumetric synthetic generator utilizing latent diffusion to synthesize target CT scans and artificial organ structures to populate sparse training datasets.

### TRACER: Trajectory-Aware Clinical Risk Prediction
*   **Publication Reference:** [TRACER Paper (arXiv:2607.18270)](https://arxiv.org/abs/2607.18270)
*   **Purpose:** Utilizes severity-grounded clinical knowledge graphs and Retrieval-Augmented Generation (RAG) to model patient trajectory profiles, predicting readmission risks and clinical severity.

### MultiGradICON: Multi-Modal Registration Foundation Model
*   **Publication Reference:** [MultiGradICON Paper (arXiv:2406.02234)](https://arxiv.org/abs/2406.02234)
*   **Purpose:** Registers volumetric CT and MRI studies across diverse structures (brain, lung, knee, liver) to align historical tumor growth, assisting in radiation oncology and treatment planning.

### SleepFM: Multi-Modal Sleep Biosignal Foundation Model
*   **Publication Reference:** [SleepFM Paper (arXiv:2404.03210)](https://arxiv.org/abs/2404.03210)
*   **Purpose:** Multi-modal contrastive pre-training aligning sleep electroencephalograms (EEG) with clinical classifications to forecast diagnostic trajectories and sleep pathologies.

---

## 4. Comparative Matrix

The table below maps the functional properties of existing whole-body frameworks:

| Model / Paper | Dimensionality | Modality Scope | Anatomical Target | Core Pretext Task / Focus |
| :--- | :--- | :--- | :--- | :--- |
| **SegVol** | 3D | CT | Whole-Body (200+ structures) | Prompt-guided visual alignment |
| **CT-FM** | 3D | CT | Whole-Body | Volumetric Contrastive Learning |
| **Pan-FM** | 3D | CT / MRI | Multi-organ (Brain, Heart, Liver, etc.) | Saliency-Guided Masking (SGM) |
| **VISTA3D** | 3D | CT | Whole-Body | Interactive prompt segmentation |
| **TotalFM** | 3D | CT | Whole-Body | Hierarchical partition projection |
| **PET-CT Model**| 3D | PET & CT | Whole-Body | Cross-modality fusion segmentation |
| **Pan-Organ Net (Ours)**| **3D & 2D** | **CT, MRI, XR, US** | **Multi-organ + Clinical Recommendations** | **SGM + Decoupled Action Planning** |
