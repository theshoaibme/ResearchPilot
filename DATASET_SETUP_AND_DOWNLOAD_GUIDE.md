# Pan-Organ Net: Complete Medical Dataset Setup, Download & Preprocessing Guide

This document provides step-by-step instructions and automated Python suites to download, verify, and preprocess all medical imaging datasets required for **Pan-Organ Net**.

---

## Directory Architecture

All core code scripts are centralized inside the `core/` directory, separated into `core/downloaders/` and `core/processors/`. Preprocessed tensors are output into `dataset/processed/`:

```
ResearchPilot/
├── core/
│   ├── downloaders/                # Dataset Acquisition Suite
│   │   ├── __init__.py
│   │   ├── main.py                 # Master Downloader Suite Runner
│   │   ├── hf_downloader.py        # Hugging Face (PanOrganNet) Downloader
│   │   ├── kaggle_downloader.py    # Kaggle Datasets Downloader (with Progress Bars)
│   │   ├── tcia_downloader.py      # TCIA Datasets Downloader
│   │   ├── physionet_downloader.py# PhysioNet Datasets Downloader
│   │   ├── open_access_downloader.py# TotalSegmentator v2 (Zenodo) Downloader
│   │   └── generate_catalog_csv.py # Medical Dataset Catalog CSV Generator
│   │
│   └── processors/                 # Dataset Preprocessing Pipeline Suite
│       ├── __init__.py
│       ├── main.py                 # Master Preprocessor Suite Runner
│       ├── base_processor.py       # CLAHE, Resampling, Windowing Utilities
│       ├── busi_processor.py       # BUSI Breast Ultrasound Processor
│       ├── covid_processor.py      # COVID-19 Radiography X-Ray Processor
│       ├── hf_processor.py         # PanOrganNet Dataset Processor
│       └── totalseg_processor.py   # TotalSegmentator 3D CT Volume Processor
│
└── dataset/                        # Dataset Storage Directory
    ├── processed/                  # Preprocessed Tensor Arrays (.npz)
    │   ├── BUSI/
    │   ├── COVID19_XRay/
    │   ├── PanOrganNet/
    │   └── TotalSegmentator/
    ├── BUSI/                       # Raw Downloaded Datasets
    ├── COVID19_XRay/
    ├── PanOrganNet/
    └── TotalSegmentator/
```

---

## 1. Master Download & Processing Suites

- **Run All Dataset Downloaders**:
  ```bash
  python3 core/downloaders/main.py
  ```

- **Run All Dataset Preprocessors**:
  ```bash
  python3 core/processors/main.py
  ```

---

## 2. Individual Downloader Execution

```bash
# Hugging Face
python3 core/downloaders/hf_downloader.py

# Kaggle Datasets (BUSI, COVID19, RSNA PE, Ultrasound Nerve, PANDA)
python3 core/downloaders/kaggle_downloader.py

# TCIA Datasets (TCGA-KIRC, TCGA-LGG, PROSTATEx, CBIS-DDSM)
python3 core/downloaders/tcia_downloader.py

# PhysioNet Datasets (MIMIC-CXR, VinDr-CXR)
python3 core/downloaders/physionet_downloader.py

# Zenodo Open Access (TotalSegmentator v2)
python3 core/downloaders/open_access_downloader.py
```

---

## 3. Preprocessing Specifications (Phase 3 Alignment)

- **2D Projections (X-Ray & Ultrasound)**:
  - **Resolution**: Resized to $512 \times 512$ grid.
  - **Normalization**: Contrast Limited Adaptive Histogram Equalization (CLAHE) + Min-Max Scaling $[0.0, 1.0]$.
  - **Masks**: Nearest-neighbor spatial interpolation to preserve binary segmentations.

- **3D CT Volumes (TotalSegmentator)**:
  - **Windowing**: Hounsfield Unit (HU) clipping to Soft Tissue Window $[-150, 250]\,\text{HU}$ scaled to $[0.0, 1.0]$.
  - **Resampling**: Isotropic voxel spacing $(1.5\,\text{mm}, 1.5\,\text{mm}, 1.5\,\text{mm})$.
