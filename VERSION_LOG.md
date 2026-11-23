# ResearchPilot — Version Control & Evolution Log

This repository follows strict directory-level versioning (`V1/`, `V2/`, `V3/`, ...) alongside Git history so every architectural iteration, dataset correction, and experimental release remains isolated, auditable, and reproducible.

---

## `V1/` — Initial Exploratory Phase & Isolated 2D Baselines (Frozen Archive)
- **Scope:** Early feasibility study with separate 2D CNN pipelines (`DenseNet121`, `EfficientNetV2-S`) on individual folders (`BUSI`, `Ultrasound_Nerve`, `COVID19_XRay`, `PROSTATEx`, `TCGA-KIRC`, `CBIS-DDSM`) plus conceptual proposal drafts.
- **Audit Findings Resolved in V2:**
  1. Legacy preprocessing for `TCGA-KIRC` (15 patients), `PROSTATEx` (25 patients), and `CBIS-DDSM` (88 files) assigned scan-type/modality proxy labels instead of clinical pathology or voxel-segmentation labels.
  2. Pulmonology split relied on image-level random splits over multi-source COVID-19 radiography data without cross-hospital external validation.
  3. No unified pan-organ multi-modality neural architecture existed in `V1/core/models/`.
  4. Five specialty subdirectories (`cardiology`, `infectious`, `neuro`, `radiology`, `rheumatology`) were unpopulated.
- **Status:** Frozen for traceability and thesis evolution comparison.

---

## `V2/` — Pan-Organ Net V2: Unified Multi-Modality Clinical Foundation Model (Active Release)
- **Architecture:** Single unified **Pan-Organ Net V2** (`OmniStem` 2D + 2.5D volumetric slabs + `PhysicsFiLM` + Modality-Adaptive Uncertainty-Guided Masking `MA-UGM` + Sparse Top-2 Modality-Task Mixture-of-Experts `MT-MoE` + Self-Supervised Anatomical Coordinate Regressor `SACR` + Frozen Biomedical Text-Prompted Mask & Pathology Decoders + Modality-Conditional Conformal Prediction `$MC^2$-CP`).
- **Datasets (8 Verified Open-Access Clinical Benchmarks, 5 Modalities, 6 Specialties):**
  - `AMOS22` (3D CT + 3D MRI, 15 abdominal organs — internal + cross-modality validation)
  - `NIH ChestX-ray14` (2D CXR, 14 pathologies, official patient split) + `RSNA Pneumonia` (external hospital validation)
  - `CBIS-DDSM` Full Curated (2D Mammography, real `Benign` vs `Malignant` pathology + ROI masks)
  - `BUSI` (2D Breast Ultrasound, deduplicated, joint pathology + tumor segmentation)
  - `CAMUS` (2D Cardiac Echo, LV endocardium/epicardium & LA segmentation + Ejection Fraction)
  - `TCGA-LGG` (Brain FLAIR MRI, glioma segmentation + presence detection)
  - `Ultrasound Nerve` (2D Neck Ultrasound, Brachial Plexus segmentation, subject-level split)
- **Clinical Safeguards:** Automated zero-patient-leakage gate, class-imbalance/FN-cost-aware losses, physics-preserving normalization, temperature scaling + `$MC^2$` conformal uncertainty sets, Attention/Grad-CAM++ faithfulness (Insertion/Deletion AUC), MNAR partial-FOV truncation tests (0–50%), 1,000-bootstrap 95% CIs, and DeLong/Wilcoxon/McNemar significance testing.
- **Infrastructure:** Google Drive (`5 TB`) + Google Colab Pro (`48 GB RAM / 100 GB local SSD`) using 256 MB sequential WebDataset/NPZ shards and a rolling 20 GB local-SSD LRU cache with fault-tolerant atomic checkpointing.

---

## `V3/` — Post-Defense Full-3D Scaling Roadmap (Planned)
- Extends the exact `V2` sharding, `MT-MoE`, `SACR`, and `$MC^2$-CP` framework to native 3D volumetric crops (`128×128×96`) across `TotalSegmentator v2` (117 CT structures), `TotalSegmentator-MRI` (80 MRI structures), and `KiTS23` (489 renal tumor volumes).
