# Dataset Provenance, Clinical Ground Truths, and Split Protocol (V2)

## 1. Cohort Design Rationale & V1-to-V2 Audit Corrections

Building a credible multi-modality pan-organ clinical model requires that every included cohort satisfy four non-negotiable standards:
1. **Authentic Clinical Ground Truth:** Supervision must correspond to clinically actionable pathology diagnoses or expert-delineated anatomical/lesion segmentation masks—never acquisition metadata proxies.
2. **Traceable Open-Access Licensing:** Every dataset must come from an established peer-reviewed public registry with documented licensing and de-identification provenance.
3. **Patient-Level Isolation:** All train, validation, and test splits must strictly separate patients/subjects (`assert set(train_ids).isdisjoint(val_ids | test_ids)`).
4. **Cross-Institutional External Validation:** At least one modality must be evaluated zero-shot or out-of-domain on an independent hospital registry to rule out scanner or site shortcutting.

### Summary of V1 Audit Findings Addressed in V2

| V1 Dataset | V1 Defect Identified During Audit | V2 Remediation |
|---|---|---|
| `TCGA-KIRC` (15 patients) | Preprocessor assigned modality label (`CT` vs `MR`) rather than renal tumor pathology; no segmentation masks available. | Replaced by **AMOS22** (500 CT + 100 MRI with 15 abdominal organs including bilateral kidneys, liver, spleen, pancreas, aorta) for multi-organ volumetric segmentation. |
| `PROSTATEx` (25 patients) | Preprocessor assigned sequence label (`T2` vs `DWI/ADC`) rather than clinically significant prostate cancer (`ClinSig`). | Archived in `V1/`; scheduled for `V3/` native 3D pairing with official `ProstateX-Findings-Train.csv`. |
| `CBIS-DDSM` (88 files) | Preprocessor classified file artifact type (`Lesion_Crop` vs `ROI_Mask` vs `Full_Mammogram`). | Upgraded to **Full Curated CBIS-DDSM** (~3,100 cases) joined with official `mass/calc_case_description` CSVs using true biopsy-verified **Benign vs. Malignant** pathology + paired ROI masks. |
| `COVID-19 Radiography` | Image-level random split across merged public hospital repositories known to induce non-clinical source shortcuts (DeGrave et al., 2021). | Replaced as primary pulmonology cohort by **NIH ChestX-ray14** (official patient-level splits, 14 pathologies) with **RSNA Pneumonia** as an independent external validation cohort. |
| `Ultrasound Nerve` | Trained only as a binary presence classifier despite having pixel-level ground-truth masks. | Upgraded to **joint pixel-level Brachial Plexus segmentation (Dice/HD95) + presence classification** under strict 47-subject splits. |
| `BUSI` | Included known near-duplicate scans and trained classification-only. | Added 8×8 perceptual-hash deduplication, multi-mask union, and upgraded to **joint Benign/Malignant/Normal classification + tumor segmentation**. |
| Empty Specialty Folders (`cardiology/`, `neuro/`) | Directories had zero data or models in V1. | Populated with **CAMUS** (500-patient cardiac echocardiography) and **TCGA-LGG** (110-patient brain glioma FLAIR MRI segmentation). |

---

## 2. Complete V2 Multi-Modality Dataset Registry

| ID | Dataset | Modality | Specialty & Target Organ(s) | Clinical Task & Ground Truth | Scale | Split Strategy | Source & License |
|---|---|---|---|---|---|---|---|
| `DS-01` | **AMOS22 (CT)** | 3D CT (2.5D 5-slice slabs + 3D eval) | Abdominal Radiology (15 organs: spleen, R/L kidney, gallbladder, esophagus, liver, stomach, aorta, IVC, pancreas, R/L adrenal, duodenum, bladder, prostate/uterus) | 16-class dense voxel segmentation + `SACR` body coordinate + MNAR FOV truncation test | 500 CT volumes | Patient-level 70/15/15 | Zenodo `7155725` (Ji et al., NeurIPS 2022) — CC BY 4.0 |
| `DS-02` | **AMOS22 (MRI)** | 3D MRI (2.5D 5-slice slabs + 3D eval) | Abdominal Radiology (same 15 organs) | Cross-modality CT↔MRI multi-organ generalization & segmentation | 100 MRI volumes | Patient-level holdout / external | Zenodo `7155725` (Ji et al., NeurIPS 2022) — CC BY 4.0 |
| `DS-03` | **NIH ChestX-ray14** | 2D Chest X-Ray | Pulmonology / Thoracic | 14-label multi-label pathology classification (Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia) | 112,120 X-rays (30,805 patients) | Official patient-level `train_val` / `test_list.txt` | NIH Clinical Center / Kaggle — Public Domain (CC0) |
| `DS-04` | **RSNA Pneumonia** | 2D Chest X-Ray | Pulmonology (External Hospital Cohort) | Radiologist-adjudicated Pneumonia opacity detection (External Validation Only) | ~26,684 patients | **100% External Test** (never trained on) | RSNA / Kaggle — Open Research |
| `DS-05` | **CBIS-DDSM (Curated)** | 2D Digital Mammography | Breast Oncology | Biopsy-verified **Benign vs. Malignant** classification + ROI lesion mask segmentation | ~3,100 lesion cases | Official patient-level train/test (+ patient val split) | TCIA / Lee et al. (Scientific Data 2017) — CC BY 3.0 |
| `DS-06` | **BUSI** | 2D Breast Ultrasound | Breast Oncology | 3-class pathology (`Normal`, `Benign`, `Malignant`) + pixel-level tumor segmentation (deduplicated) | 780 scans (~600 patients) | Deduplicated patient-grouped 70/15/15 | Al-Dhabyani et al. (Data in Brief 2020) — CC BY 4.0 |
| `DS-07` | **CAMUS** | 2D Cardiac Echocardiography | Cardiology | 4-class structural segmentation (LV Endocardium, LV Epicardium, Left Atrium) across ED & ES phases | 500 patients (~2,000 frames) | Patient-level 70/15/15 | CREATIS / Leclerc et al. (IEEE TMI 2019) — Open Research |
| `DS-08` | **TCGA-LGG MRI** | 2D/2.5D Brain FLAIR MRI | Neuro-Oncology | Manual lower-grade glioma tumor segmentation + slice tumor detection | 110 patients (3,929 slices) | Patient-level 70/15/15 | TCIA / Buda et al. (CBM 2019) — CC0 |
| `DS-09` | **Ultrasound Nerve** | 2D Neck Ultrasound | Peripheral Neurology | Brachial Plexus nerve pixel segmentation + presence detection | 47 subjects (5,635 scans) | Subject-level 70/15/15 | Kaggle Ultrasound Nerve Segmentation — Research |

---

## 3. Harmonized Storage & Zero-Leakage Audit Protocol

All datasets are processed via `V2/notebooks/01_Data_Download_and_Harmonize_Colab.ipynb` into:
- **Sequential 256 MB `.tar` Shards** stored permanently on Google Drive (`/MyDrive/ResearchPilot_PanOrgan/V2/preprocessed_shards/`). Each entry stores compressed `float16` normalized images, `uint8` ground-truth masks, and JSON physics/clinical metadata.
- **Unified Manifest (`master_manifest.parquet`)**: Single tabular index linking every `sample_id` to `patient_id`, `dataset`, `modality`, `split`, `spacing_mm`, `body_coord_z`, `prompt`, `label`/`multilabel`, and exact shard location.
- **Automated Leakage Certificate (`split_audit_report.json`)**: Generated by `verify_zero_patient_overlap()` (`panorgan/data/shard_builder.py`). Pipeline execution halts with `PatientLeakageError` if even a single `patient_id` overlaps between `train`, `val`, `test`, or `external_test`.
