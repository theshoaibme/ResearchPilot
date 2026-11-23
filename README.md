# Pan-Organ Net: Unified Multi-Modality Clinical Medical Imaging Framework

A versioned medical imaging research repository implementing **Pan-Organ Net V2**, a unified 2D + 2.5D volumetric multi-modality foundation framework spanning **5 imaging modalities** (CT, MRI, Chest X-Ray, Ultrasound, Mammography) and **6 clinical specialties** (Abdominal Radiology, Pulmonology, Oncology, Cardiology, Neurology, Nephrology).

## Repository Structure

| Directory | Role | Status |
|---|---|---|
| [`V1/`](V1/) | Frozen archive of Phase-1 exploratory baselines, raw local datasets, and initial proposal drafts | Frozen Archive |
| [`V2/`](V2/) | **Active Release:** Unified `Pan-Organ Net V2` package (`panorgan/`), leakage-audited multi-modality preprocessing, Colab Pro notebooks, automated test suite, and complete thesis documentation | Active |
| [`V3/`](V3/) | Planned post-defense native 3D whole-body scaling roadmap (`TotalSegmentator v2`, `TotalSegmentator-MRI`, `KiTS23`) | Planned |
| [`VERSION_LOG.md`](VERSION_LOG.md) | Complete changelog tracking every transition across `V1 → V2 → V3` | Maintained |

## Quick Start (`V2` — Local Verification & Colab Pro Training)

```bash
# 1. Install V2 package in editable mode
pip install -e V2/

# 2. Run full automated unit & leakage verification suite
pytest V2/tests -v
```

### Google Colab Pro Pipeline (`V2/notebooks/`)
Run the 5 notebooks sequentially in Google Colab Pro with Google Drive mounted:
1. `01_Data_Download_and_Harmonize_Colab.ipynb` — Downloads open-access clinical benchmarks straight to Drive, runs physics-aware harmonization, verifies 0% patient leakage, and writes 256 MB sequential shards.
2. `02_Stage1_SSL_Pretrain_Colab.ipynb` — Self-supervised pretraining with Modality-Adaptive Uncertainty-Guided Masking (`MA-UGM`) and Self-Supervised Anatomical Coordinate Regression (`SACR`).
3. `03_Stage2_Multitask_PanOrgan_Train_Colab.ipynb` — End-to-end unified multi-modality segmentation + pathology training with Sparse Top-2 `MT-MoE` and atomic Drive checkpointing.
4. `04_Baselines_and_Ablations_Colab.ipynb` — Identical-split baselines (`DenseNet121`, `ConvNeXt-S`, `UNet-2.5D`) and controlled architectural ablations.
5. `05_Eval_Calibration_XAI_Paper_Tables_Colab.ipynb` — Internal + external hospital validation, MNAR FOV-truncation stress curves, `$MC^2$` conformal calibration, XAI faithfulness, bootstrap 95% CIs, and LaTeX tables.

## Documentation (`V2/docs/`)
- [`DATASET_PROVENANCE_AND_SPLITS.md`](V2/docs/DATASET_PROVENANCE_AND_SPLITS.md) — Complete dataset sources, licenses, clinical ground truths, and patient-level split guarantees.
- [`ARCHITECTURE_AND_MATHEMATICS.md`](V2/docs/ARCHITECTURE_AND_MATHEMATICS.md) — Formal architectural specification, equations, complexity profile, and SOTA comparison.
- [`CLINICAL_SAFETY_AND_VALIDATION_PROTOCOL.md`](V2/docs/CLINICAL_SAFETY_AND_VALIDATION_PROTOCOL.md) — 10-pillar clinical ML engineering protocol, statistical testing, and conformal uncertainty bounds.
- [`VIVA_DEFENSE_QA_HANDBOOK.md`](V2/docs/VIVA_DEFENSE_QA_HANDBOOK.md) — Structured technical defense reference covering datasets, design trade-offs, ablations, and clinical limitations.
- [`VERIFIED_REFERENCES.bib`](V2/docs/VERIFIED_REFERENCES.bib) — Verified peer-reviewed BibTeX bibliography.
