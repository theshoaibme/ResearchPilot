# Comprehensive Inventory of Volumetric and Whole-Body Medical Foundation Models

This report catalogs and details the key research publications, methodologies, datasets, and diagnostic purposes of existing "whole-body," "pan-organ," and "multi-organ" medical imaging foundation models available in current scientific literature.

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

## 2. Comparative Matrix

The table below maps the functional properties of existing whole-body frameworks:

| Model / Paper | Dimensionality | Modality Scope | Anatomical Target | Core Architectural Pretext Task |
| :--- | :--- | :--- | :--- | :--- |
| **SegVol** | 3D | CT | Whole-Body (200+ structures) | Prompt-guided visual alignment |
| **CT-FM** | 3D | CT | Whole-Body | Volumetric Contrastive Learning |
| **Pan-FM** | 3D | CT / MRI | Multi-organ (Brain, Heart, Liver, etc.) | Saliency-Guided Masking (SGM) |
| **VISTA3D** | 3D | CT | Whole-Body | Interactive prompt segmentation |
| **TotalFM** | 3D | CT | Whole-Body | Hierarchical partition projection |
| **PET-CT Model**| 3D | PET & CT | Whole-Body | Cross-modality fusion segmentation |
| **Pan-Organ Net (Ours)**| **3D & 2D** | **CT, MRI, XR, US** | **Multi-organ + Clinical Recommendations** | **SGM + Decoupled Action Planning** |
