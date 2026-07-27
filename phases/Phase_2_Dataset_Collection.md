# Phase 2: Comprehensive Multi-Modal, Multi-Organ Dataset Collection & Inventory

---

## 1. Data Repository Catalog & Cohort Breakdown

To train and validate **Pan-Organ Net** across all 7 medical specialist domains and 8 disease categories, Phase 2 aggregates **5 primary multi-modal, multi-organ datasets** comprising over 500,000 patient examinations:

```
├── TotalSegmentator Dataset (v2)
│   ├── Modality: 3D Computed Tomography (CT)
│   ├── Scale: 1,204 high-resolution CT volumes
│   ├── Target: 117 anatomical organs, bones, vessels, and tissue structures
│   ├── Licensing: Open Access (CC BY 4.0)
│   └── Primary Role: Ground-truth pan-organ spatial segmentation benchmark
│
├── MIMIC-CXR Database (v2.0.0)
│   ├── Modality: 2D Chest Radiography (X-Ray)
│   ├── Scale: 377,110 chest radiographs across 227,835 imaging studies
│   ├── Target: High-throughput thoracic screening & 14 disease label classifications
│   ├── Licensing: PhysioNet Credentialed Health Data License
│   └── Primary Role: Pretraining projection spatial representations & thoracic disease screening
│
├── The Cancer Imaging Archive (TCIA) Multi-Parametric Cohort
│   ├── Modality: 3D Multi-parametric MRI & Contrast CT
│   ├── Scale: ~120,000 volumetric series (~5,200 patient studies)
│   ├── Sub-Cohorts:
│   │   ├── TCGA-LGG / TCGA-GBM (Brain Low/High-Grade Glioma MRI)
│   │   ├── TCGA-KIRC (Clear Cell Renal Carcinoma CT/MRI)
│   │   ├── CPTAC-LUAD / LIDC-IDRI (Lung Adenocarcinoma & Nodule CT)
│   │   └── PROSTATEx (Prostate Multi-parametric MRI)
│   ├── Licensing: Open Access TCIA Data Usage Policy
│   └── Primary Role: Soft-tissue tumor characterization & neoplastic lesion segmentation
│
├── BraTS (Brain Tumor Segmentation Challenge 2023)
│   ├── Modality: Multi-sequence 3D MRI (T1, T1Gd, T2, FLAIR)
│   ├── Scale: 4,500 volumetric MRI scans
│   ├── Target: Glioma sub-regions (Enhancing tumor, Tumor core, Whole tumor, Edema)
│   ├── Licensing: Synapse Data Access License
│   └── Primary Role: Neuro-oncology baseline for central nervous system screening
│
└── LUNA16 (LUng Nodule Analysis 2016 / LIDC-IDRI)
    ├── Modality: Low-dose 3D Chest CT
    ├── Scale: 888 thoracic CT scans
    ├── Target: 1,018 lung nodule ground-truth annotations with radiologist agreement
    ├── Licensing: Public Benchmark Dataset
    └── Primary Role: High-resolution pulmonary nodule detection & malignancy scoring
```

---

## 2. Modality & Organ System Coverage Matrix

| Organ System | Target Organs & Structures | Modalities | Primary Dataset Source | Specialist Alignment |
| :--- | :--- | :--- | :--- | :--- |
| **Central Nervous System (CNS)** | Brain parenchyma, Ventricles, White matter, Spinal cord | MRI (T1, T2, FLAIR, dTI), CT | BraTS, TCIA (TCGA-LGG/GBM) | Neurologist, Neurosurgeon, Oncologist |
| **Thoracic & Respiratory** | Lungs, Trachea, Bronchi, Pleura, Mediastinum | Chest X-Ray, CT (HRCT) | MIMIC-CXR, LUNA16, TotalSegmentator | Pulmonologist, Infectious Disease |
| **Cardiovascular System** | Heart chambers, Aorta, Pulmonary arteries, Vena cava | CT Angiography, Cardiac MRI | TotalSegmentator, TCIA | Cardiologist, Vascular Surgeon |
| **Abdominal & Digestive** | Liver, Spleen, Pancreas, Gallbladder, Stomach | CT, Multi-Parametric MRI | TotalSegmentator, TCIA (TCGA-KIRC) | Oncologist, Infectious Disease |
| **Urinary & Renal** | Left/Right Kidneys, Renal pelvis, Ureters, Bladder | Contrast CT, MRI | TotalSegmentator, TCIA | Nephrologist, Oncologist |
| **Pelvic & Reproductive** | Prostate, Uterus, Ovaries | Multi-Parametric MRI, CT | PROSTATEx (TCIA), TotalSegmentator | Oncologist, Radiologist |
| **Musculoskeletal System** | Vertebrae (C1-S1), Ribs, Sternum, Femur, Pelvic bones | CT, Musculoskeletal MRI | TotalSegmentator, TCIA | Rheumatologist, Orthopedic Surgeon |

---

## 3. Dataset Curation, Licensing & Access Protocol

1. **Patient Privacy & De-Identification:** All 5 datasets are fully anonymized in compliance with HIPAA Safe Harbor standards. DICOM headers are scrubbed of PII.
2. **Train / Validation / Test Splitting Strategy:**
   - **Pre-Training Set (80%):** Unlabeled CT, MRI, and X-Ray volumes for Saliency-Guided Masked Autoencoding (SGM).
   - **Validation Set (10%):** Patient-stratified split for hyperparameter tuning.
   - **Test Benchmark Set (10%):** Held-out patient studies for zero-shot and few-shot evaluation against ground-truth segmentations.

---

## 4. Phase 2 Verification & Data Readiness Status

- **Completeness Check:** Covers all 7 medical specialist domains, 8 disease categories, 4 imaging modalities (CT, MRI, X-Ray, US), and 5 benchmark datasets.
- **Phase 3 Integration:** Preprocessed arrays (NIfTI / DICOM) are handed over to Phase 3 for spatial resampling and intensity normalization.
