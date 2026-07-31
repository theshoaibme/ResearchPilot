# Automated Dataset Download & Access Guide for Pan-Organ Net

This guide provides step-by-step automated CLI commands to download each dataset directly onto your system.

---

## 1. Kaggle Datasets (Automated via Kaggle API)

### Prerequisites:
1. Obtain your `kaggle.json` API token from [Kaggle Account Settings](https://www.kaggle.com/settings).
2. Save `kaggle.json` to: `C:\Users\HP\.kaggle\kaggle.json`

### Commands:
```bash
# BUSI Breast Ultrasound
kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset -p ./datasets/BUSI --unzip

# COVID-19 Radiography Database
kaggle datasets download -d tawsifurrahman/covid19-radiography-database -p ./datasets/COVID19_XRay --unzip

# RSNA Pulmonary Embolism CT Challenge
kaggle competitions download -c rsna-str-pulmonary-embolism-detection -p ./datasets/RSNA_PE

# Ultrasound Nerve Segmentation
kaggle competitions download -c ultrasound-nerve-segmentation -p ./datasets/Ultrasound_Nerve

# PANDA Prostate Grade WSI
kaggle competitions download -c prostate-cancer-grade-assessment -p ./datasets/PANDA_WSI
```

---

## 2. PhysioNet Datasets (MIMIC-CXR & VinDr-CXR)

### Prerequisites:
1. Complete PhysioNet CITI training certification.
2. Install PhysioNet client: `pip install physionet-build`

### Commands:
```bash
# MIMIC-CXR-JPG (requires credentials)
wget -r -N -c np --user <YOUR_USERNAME> --ask-password https://physionet.org/files/mimic-cxr-jpg/2.0.0/ -P ./datasets/MIMIC_CXR

# VinDr-CXR
wget -r -N -c np https://physionet.org/files/vindr-cxr/1.0.0/ -P ./datasets/VinDr_CXR
```

---

## 3. The Cancer Imaging Archive (TCIA Datasets via NBIA Data Retriever)

Download the [NBIA Data Retriever CLI](https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever+Command+Line+Interface+User+Guide) or use `tcia-utils`:

```bash
pip install tcia-utils
```

### Python Script (`download_tcia.py`):
```python
from tcia_utils import nbia

# Download Brain LGG Series
nbia.downloadSeries(collection="TCGA-LGG", path="./datasets/TCGA_LGG")

# Download Renal KIRC Series
nbia.downloadSeries(collection="TCGA-KIRC", path="./datasets/TCGA_KIRC")

# Download Prostate PROSTATEx
nbia.downloadSeries(collection="PROSTATEx", path="./datasets/PROSTATEx")

# Download Mammography CBIS-DDSM
nbia.downloadSeries(collection="CBIS-DDSM", path="./datasets/CBIS_DDSM")
```

---

## 4. Open Access Web Datasets (Direct Browser / Curl Links)

| Dataset | Direct Web Access Page | Format |
| :--- | :--- | :--- |
| **TotalSegmentator** | [https://zenodo.org/records/10047292](https://zenodo.org/records/10047292) | NIfTI (.nii.gz) |
| **BraTS 2023** | [https://www.synapse.org/#!Synapse:syn51156910](https://www.synapse.org/#!Synapse:syn51156910) | NIfTI (.nii.gz) |
| **AbdomenCT-1K** | [https://github.com/dwyang/AbdomenCT-1K](https://github.com/dwyang/AbdomenCT-1K) | NIfTI (.nii.gz) |
| **LUNA16** | [https://luna16.grand-challenge.org/](https://luna16.grand-challenge.org/) | MHD / Raw |
| **OASIS-3** | [https://www.oasis-brains.org/](https://www.oasis-brains.org/) | DICOM / NIfTI |
| **CAMELYON17** | [https://camelyon17.grand-challenge.org/](https://camelyon17.grand-challenge.org/) | WSI TIF |
| **InBreast** | [https://data.mendeley.com/datasets/ywsfp3v2bc/1](https://data.mendeley.com/datasets/ywsfp3v2bc/1) | DICOM |
