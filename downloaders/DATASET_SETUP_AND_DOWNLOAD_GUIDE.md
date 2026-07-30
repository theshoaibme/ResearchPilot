# Pan-Organ Net: Complete Medical Dataset Setup & Download Guide

This document provides step-by-step instructions and automated Python scripts to download, verify, and organize all medical imaging datasets required for **Pan-Organ Net**.

---

## Directory Structure Overview

All dataset downloader scripts are organized inside the `downloaders/` folder, and all dataset files are stored directly in `./dataset/`:

```
ResearchPilot/
├── downloaders/
│   ├── __init__.py
│   ├── main.py                     # Master execution runner
│   ├── hf_downloader.py            # Hugging Face (PanOrganNet) Downloader
│   ├── kaggle_downloader.py        # Kaggle Datasets Downloader (with Progress Bars)
│   ├── tcia_downloader.py          # TCIA Datasets Downloader
│   ├── physionet_downloader.py    # PhysioNet Datasets Downloader
│   └── open_access_downloader.py   # TotalSegmentator v2 (Zenodo) Downloader
└── dataset/
    ├── PanOrganNet/                # Hugging Face Repository Dataset
    ├── COVID19_XRay/               # COVID-19 Radiography Database (Kaggle)
    ├── BUSI/                       # Breast Ultrasound Images (Kaggle)
    ├── Ultrasound_Nerve/           # Ultrasound Nerve Segmentation (Kaggle)
    ├── RSNA_PE/                    # RSNA Pulmonary Embolism CT (Kaggle)
    ├── PANDA_WSI/                  # PANDA Prostate Cancer Grade WSI (Kaggle)
    ├── TotalSegmentator/           # TotalSegmentator v2 (Zenodo)
    ├── TCGA_KIRC/                  # Clear Cell Renal CT (TCIA)
    ├── TCGA_LGG/                   # Brain Glioma MRI (TCIA)
    ├── PROSTATEx/                  # Prostate Multi-parametric MRI (TCIA)
    ├── CBIS_DDSM/                  # Breast Mammography (TCIA)
    ├── MIMIC_CXR/                  # Chest Radiography (PhysioNet)
    └── VinDr_CXR/                  # Vietnamese Chest X-Ray (PhysioNet)
```

---

## Quick Start: Master Downloader Suite

To run all automated dataset downloads in sequence:

```bash
python3 downloaders/main.py
```

---

## Individual Downloader Scripts

### 1. Hugging Face Dataset (`PanOrganNet`)
- **Script**: `downloaders/hf_downloader.py`
- **Repo**: [the-shoaib2/PanOrganNet](https://huggingface.co/datasets/the-shoaib2/PanOrganNet)
- **Token**: `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`

```bash
python3 downloaders/hf_downloader.py
```

---

### 2. Kaggle Datasets
- **Script**: `downloaders/kaggle_downloader.py`
- **Token**: `KGAT_7779efdbcabb625d5aaffdb2a39465c4`
- Includes real-time `tqdm` progress bars and automatic zip extraction.

```bash
python3 downloaders/kaggle_downloader.py
```

---

### 3. TCIA Datasets (The Cancer Imaging Archive)
- **Script**: `downloaders/tcia_downloader.py`
- **Collections**: `TCGA-KIRC`, `TCGA-LGG`, `PROSTATEx`, `CBIS-DDSM`

```bash
python3 downloaders/tcia_downloader.py
```

---

### 4. PhysioNet Datasets (`MIMIC-CXR` & `VinDr-CXR`)
- **Script**: `downloaders/physionet_downloader.py`
- Prompts for PhysioNet user credentials or reads `PHYSIONET_USER` and `PHYSIONET_PASS`.

```bash
python3 downloaders/physionet_downloader.py
```

---

### 5. Zenodo / Open Access Datasets (`TotalSegmentator`)
- **Script**: `downloaders/open_access_downloader.py`
- Downloads **TotalSegmentator v2** (23.5 GB) into `./dataset/TotalSegmentator/`.

```bash
python3 downloaders/open_access_downloader.py
```
