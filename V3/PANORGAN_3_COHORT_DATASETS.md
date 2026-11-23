# PanOrgan 3 Cohort — Official Dataset & Modality Inventory

- **Model Name:** `PanOrgan 3 Cohort` (Version `3.1`)
- **Master Manifest:** `/Volumes/Shoaib/PanOrgan_V3/preprocessed_shards/master_manifest.parquet`
- **Total Sharded Samples:** `11,735` (Patient-level Zero-Leakage `train / val / test` split)
- **Imaging Modalities (5):** `MRI`, `CT`, `CXR`, `US`, `MG`

## Cohort Summary Table

| # | Cohort (`dataset`) | Modality | Samples | Unique Patients | Split (`Train / Val / Test`) | Labeled Samples | Role / Task | Official Data & Label Source |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| 1 | `tcga_lgg` | `MRI` | 3,929 | 110 | 2,829 / 563 / 537 | 3,929 | `joint_seg_cls` (Neurology / Brain MRI) | TCIA TCGA-LGG |
| 2 | `pulmonology_cxr` | `CXR` | 2,000 | 2,000 | 1,333 / 333 / 334 | 2,000 | Lung Segmentation & Classification | Pulmonology CXR (Montgomery / Shenzhen) |
| 3 | `ultrasound_nerve` | `US` | 1,500 | 47 | 1,002 / 232 / 266 | 1,500 | Nerve Segmentation & Presence | Kaggle Ultrasound Nerve Segmentation |
| 4 | `tcga_kirc_ct` | `CT` | 1,200 | 15 | 797 / 214 / 189 | 1,200 | Renal Grade Classification (`G1/G2` vs `G3/G4`) | TCIA TCGA-KIRC + Official NCI GDC API (`tcga_kirc_official_clinical.csv`) |
| 5 | `prostatex_mri` | `MRI` | 1,198 | 25 | 812 / 197 / 189 | 0 | Unlabeled SSL / Anatomy Pretraining | TCIA / SPIE-AAPM-NCI PROSTATEx |
| 6 | `nih_cxr14` | `CXR` | 1,167 | 882 | 811 / 177 / 179 | 1,167 | `multilabel_classification` (14 Thoracic Findings) | NIH Clinical Center ChestX-ray14 |
| 7 | `busi` | `US` | 653 | 653 | 435 / 109 / 109 | 653 | `joint_seg_cls` (Breast Ultrasound) | Mendeley BUSI (Al-Dhabyani et al.) |
| 8 | `cbis_ddsm` | `MG` | 88 | 45 | 59 / 14 / 15 | 88 | Breast Mammography Pathology (`BENIGN` vs `MALIGNANT`) | TCIA CBIS-DDSM (`mass/calc_case_description_train/test_set.csv`) |
| **Total** | **8 Cohorts** | **5 Modalities** | **11,735** | **3,777** | **8,078 / 1,839 / 1,818** | **10,537** | **Multimodal Segmentation + Classification + SSL** | **100% Official Clinical Sources (Zero Synthetic Labels)** |

## Notes on Data Integrity & Storage
- **Zero-Leakage Splitting:** All splits are strictly grouped by `patient_id` so no patient appears across `train`, `val`, and `test`.
- **No Fabricated Labels:** `prostatex_mri` (`1,198` samples) is kept strictly unlabeled (`label = -1`) for self-supervised representation learning because its official clinical findings table requires restricted TCIA challenge authentication.
- **External SSD Storage:** All raw datasets, preprocessed `.npz` shards, model checkpoints, and HuggingFace/PyTorch caches reside on `/Volumes/Shoaib/PanOrgan_V3/` to keep the internal Mac storage lightweight.
