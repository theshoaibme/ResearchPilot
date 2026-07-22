# Classification of Existing Multi-Modality and Whole-Body Foundation Models

This document aggregates and organizes current state-of-the-art generalist foundation models and volumetric whole-body frameworks in the medical imaging literature.

---

## 1. Volumetric Whole-Body Foundation Models (3D CT/MRI/PET)
These models process full 3D volumes to capture spatial landmarks across multiple anatomical organs.

### SegVol (Universal Segmenter)
*   **Imaging Modalities:** 3D CT.
*   **Target Organs:** 100+ structural organs and anatomical systems (abdominal, thoracic, pelvic).
*   **Description:** Supports zero-shot segmentation of arbitrary volumetric structures using interactive prompt tokens.

### CT-FM (CT Foundation Model)
*   **Imaging Modalities:** 3D CT.
*   **Target Organs:** Whole-body (lungs, mediastinum, liver, kidneys, skeleton).
*   **Description:** Pre-trained on over 100,000 CT volumes using self-supervised 3D Masked Image Modeling (MIM), showing strong down-stream adaptation.

### VISTA3D (Volumetric Imaging System)
*   **Imaging Modalities:** 3D CT / MRI.
*   **Target Organs:** Multi-organ segmentations (Brain, liver, kidneys, lungs, cardiovascular structures).
*   **Description:** A generalized volumetric segmentation framework robust under diverse window sizes.

---

## 2. Multi-Modal Vision-Language Medical Models (VLM)
These models combine diverse 2D/3D scans (X-Rays, Ultrasound, CT slices) with textual prompts for clinical report generation and visual question answering (VQA).

### BiomedCLIP (Biomedical Contrastive Language-Image Pretraining)
*   **Imaging Modalities:** 2D X-Ray, CT slices, Pathology slides, Ultrasound.
*   **Target Organs:** Whole-body representations (Thoracic, Abdominal, Brain).
*   **Description:** Standard baseline for language-image alignment, highly capable in zero-shot medical classification.

### LLaVA-NeXT-Med
*   **Imaging Modalities:** Multi-modality (X-Ray, CT, MRI, Ultrasound, Pathology).
*   **Target Organs:** Whole-body.
*   **Description:** Instantiates clinical dialogue capabilities, interpreting multiple scan types simultaneously for cross-exam diagnostic support.

---

## 3. Modality and Target Classification Matrix

The table below maps the existing models against their target structures and modalities to highlight functional coverage:

| Model | Modality Type | Dimensionality | Target Coverage | Key Pretext Task |
| :--- | :--- | :--- | :--- | :--- |
| **SegVol** | CT | 3D Volumetric | Multi-organ (100+ structures) | Interactive Prompting |
| **CT-FM** | CT | 3D Volumetric | Whole-body | Masked Image Modeling (MIM) |
| **BiomedCLIP** | Multi-modality | 2D Projection/Slices | Whole-body / Pathology | Contrastive Learning (CLIP) |
| **VISTA3D** | CT / MRI | 3D Volumetric | Multi-organ | Dense segmentation |
| **LLaVA-NeXT-Med**| Multi-modality | 2D / Volumetric Slices | Whole-body | Autoregressive language modeling |
| **Pan-Organ Net (Ours)**| **CT, MRI, XR, US** | **3D & 2D Unified** | **Multi-organ (Brain, Heart, Liver, Kidneys, Lungs)** | **Saliency-Guided Masking (SGM)** |
