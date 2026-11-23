# PanOrgan 1 — Technical Architecture & Prototype Report

## 1. Overview & Objectives
`PanOrgan 1` was the foundational design and research-scoping release of the PanOrgan project. Its primary objective was to address clinical **model fragmentation**—where isolated single-organ, single-modality neural networks fail to generalize across hospital workflows—by designing a unified multi-modal, multi-task medical foundation framework.

## 2. Architecture & Modeling
- **Core Architectural Concept:** Proposed the **Hierarchical Anatomical Tokenizer (HAT)** to map heterogeneous volumetric (`3D` CT/MRI) and projection/acoustic (`2D` X-Ray, Ultrasound) inputs into a shared token space.
- **Self-Supervised Pretraining Design:** Formulated the **Saliency-Guided Masked Autoencoder (SGM)** objective to reduce dominant-organ shortcut learning during masked reconstruction.
- **Target Clinical Scope:** Designed around 7 clinical domains (Urology, Pulmonology, Cardiology, Hepatology, Nephrology, Gastroenterology, Musculoskeletal) with SNOMED CT and RadLex ontology alignment.

## 3. Datasets & Staging Strategy
- **3-Tier Allocation Plan:** Established a structured data curation plan separating self-supervised pretraining cohorts, supervised multi-task cohorts, and held-out external validation cohorts.
- **Prototype Workspace:** Initial raw dataset staging and core prototype modules were organized under `V1/dataset` and `V1/core` (later archived to `/Volumes/Shoaib/PanOrgan_V2/V1_archive/` to keep internal storage lightweight).

## 4. Training & Calibration Protocol
- `PanOrgan 1` served as the architectural blueprint and experimental staging phase. No unified multi-cohort production training run or held-out test calibration benchmark was finalized in this version; instead, its design directly informed the executable `V2` sharding and training pipeline.

## 5. Verified Evaluation Metrics
- **Status:** Proposal, literature synthesis, and prototype staging completed (`V1/docs/FINAL_PROJECT_PROPOSAL_AND_MASTER_REPORT.md`).
- **Quantitative Benchmark:** No unified multi-cohort test evaluation JSON was produced at stage `1` (quantitative benchmarking began in `PanOrgan 2`).

## 6. Known Limitations
1. **Conceptual / Pre-Benchmark Stage:** `PanOrgan 1` did not execute end-to-end multi-cohort training or produce held-out test metrics (`Dice`, `AUROC`, `ECE`).
2. **Storage & Compute Constraints:** Initial local storage layout placed datasets and caches on the host drive without external SSD sharding or Apple Silicon (`MPS`) memory-profile optimization.
3. **Lack of Patient-Level Split Auditing:** Automated zero-leakage `patient_id` split auditing and conformal uncertainty calibration (`MC2`) were not yet implemented.
4. **Broad Modality Target vs. Real Data Availability:** The 12-modality / 35-dataset vision exceeded locally curated, verified clinical data available for immediate training.
