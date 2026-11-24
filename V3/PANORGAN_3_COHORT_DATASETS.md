# PanOrgan 3 Cohort — Official Dataset & Modality Inventory (`3.1` & `3.2`)

- **Model Name:** `PanOrgan 3 Cohort` (Versions `3.1` & `3.2`)
- **Master Manifest:** `/Volumes/Shoaib/PanOrgan_V3/preprocessed_shards/master_manifest.parquet`
- **Imaging Modalities (5):** `MRI`, `CT`, `CXR`, `US`, `MG`

---

## 1. Current `PanOrgan 3.1` Cohorts (`11,735` Samples — Zero Data Removed)

All splits (`8,078 train / 1,839 val / 1,818 test`) are strictly grouped by `patient_id` (`leakage_count = 0`).

| # | Cohort (`dataset`) | Modality | Samples | Patients | Split (`Train / Val / Test`) | Labeled | Task | Official Data Source |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| 1 | `tcga_lgg` | `MRI` | 3,929 | 110 | 2,829 / 563 / 537 | 3,929 | `joint_seg_cls` (Brain MRI) | [TCIA TCGA-LGG](https://www.cancerimagingarchive.net/collection/tcga-lgg/) |
| 2 | `pulmonology_cxr` | `CXR` | 2,000 | 2,000 | 1,333 / 333 / 334 | 2,000 | Lung Seg & Pulmonary Cls | [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) |
| 3 | `ultrasound_nerve` | `US` | 1,500 | 47 | 1,002 / 232 / 266 | 1,500 | Nerve Seg & Presence | [Kaggle Ultrasound Nerve](https://www.kaggle.com/c/ultrasound-nerve-segmentation) |
| 4 | `tcga_kirc_ct` | `CT` | 1,200 | 15 | 797 / 214 / 189 | 1,200 | Renal Grade (`G1/G2` vs `G3/G4`) | [TCIA TCGA-KIRC](https://www.cancerimagingarchive.net/collection/tcga-kirc/) + [NCI GDC API](https://api.gdc.cancer.gov/) |
| 5 | `prostatex_mri` | `MRI` | 1,198 | 25 | 812 / 197 / 189 | 0 | Unlabeled SSL Pretraining | [TCIA PROSTATEx](https://www.cancerimagingarchive.net/collection/prostatex/) |
| 6 | `nih_cxr14` | `CXR` | 1,167 | 882 | 811 / 177 / 179 | 1,167 | 14-Finding Thoracic Cls | [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) |
| 7 | `busi` | `US` | 653 | 653 | 435 / 109 / 109 | 653 | Breast US `joint_seg_cls` | [Mendeley BUSI](https://data.mendeley.com/datasets/wmy84gzngw/1) |
| 8 | `cbis_ddsm` | `MG` | 88 | 45 | 59 / 14 / 15 | 88 | Mammography Pathology | [TCIA CBIS-DDSM](https://www.cancerimagingarchive.net/collection/cbis-ddsm/) |
| **Total (`3.1`)** | **8 Cohorts** | **5 Modalities** | **11,735** | **3,777** | **8,078 / 1,839 / 1,818** | **10,537** | **Seg + Cls + SSL** | **100% Official Clinical Sources** |

---

## 2. `PanOrgan 3.2` Balanced Expansion Plan (`~5.12 GB` New Downloads)

To eliminate class imbalance (`Class 2 = 6.5%`, `Class 3 = 4.7%` in `3.1`) and fix `100%` conformal abstention on `MG` & `CT` without removing any existing data, `PanOrgan 3.2` adds:

| Target Gap | Dataset / Source | Exact Size | Target Class / Samples Added | Official Source Link |
| :--- | :--- | :---: | :--- | :--- |
| **Mammography (`MG`)** | `CMMD` (`core-img-cls-mammography-cmmd`) | **1.60 GB** (HF) | `+1,500–2,000` `MG` scans (`BENIGN` / `MALIGNANT`) | [multabench/core-img-cls-mammography-cmmd](https://huggingface.co/datasets/multabench/core-img-cls-mammography-cmmd) |
| **Kidney `CT`** | `Kidney CT Abnormality` | **3.52 GB** (HF) | `+2,000+` `CT` scans (expands beyond 15 KIRC pts) | [Euniceyeee/kidney-ct-abnormality](https://huggingface.co/datasets/Euniceyeee/kidney-ct-abnormality) |
| **Class `2` & Class `3` Balance (`CXR`)** | Unused `COVID-19 Radiography` (`Lung_Opacity` = 6,012, `Viral Pneumonia` = 1,345, `COVID` = 3,616) | **0 GB** (Already on SSD) | Balances `Class 2` (`690 -> 2,500+`) & `Class 3` (`500 -> 1,345+`) | [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) |

- **Total New Download Size:** **`~5.12 GB`** (fits easily in `/Volumes/Shoaib/`, `144 GiB` free).
- **Balancing Policy:** Combined with **Class-Balanced Weighted Sampling** during `3.2` fine-tuning so all 4 classes (`0`, `1`, `2`, `3`) receive balanced gradient updates.
