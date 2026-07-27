# Phase 1: Literature Review & Baseline Analysis

---

## 1. Overview of Medical AI Landscape
Traditional deep learning diagnostic models operate under the **"One Model, One Task"** limitation. They focus strictly on isolated anatomical regions (e.g., single-organ segmentation or single-modality disease classification). Recent foundation models attempt to broaden this scope, but key challenges remain.

---

## 2. State-of-the-Art Baseline Matrix

| Model | Modality Support | Spatial Dimension | Pretext Task / Training | Primary Limitation |
| :--- | :--- | :--- | :--- | :--- |
| **3D U-Net** | Single (CT or MRI) | 3D Volumetric | Supervised Segmentation | Lacks transferability across modalities; prone to domain shift. |
| **BiomedCLIP** | 2D Projections (X-Ray/Histology) | 2D | Contrastive Text-Image Pairing | Incapable of parsing 3D volumetric spatial dependencies across organs. |
| **TotalSegmentator** | CT | 3D Volumetric | Supervised Multi-Organ | Requires exhaustive, manual voxel-level annotations for every target organ. |
| **Pan-FM** | CT / MRI | 3D Volumetric | Random Masked Autoencoding (MAE) | Vulnerable to **dominant-organ shortcutting** and performance collapse under incomplete body scans. |
| **Pan-Organ Net (Proposed)** | CT, MRI, X-ray, Ultrasound | 2D / 3D Heterogeneous | Saliency-Guided MAE + Meta-Tokenization | Directly addresses shortcutting, MNAR data distributions, and non-actionable outputs. |
