# Phase 2: Dataset Collection & Inventory

---

## 1. Data Repository Catalog

To train and validate a pan-organ foundation model, we curate multi-modal data covering diverse anatomical regions:

```
├── TotalSegmentator Dataset
│   ├── Modality: 3D Computed Tomography (CT)
│   ├── Scale: 1,204 high-resolution CT volumes
│   ├── Target: 117 organ and anatomical structure ground-truth masks
│   └── Role: Primary benchmark for multi-organ spatial integrity & segmentation
│
├── MIMIC-CXR Database
│   ├── Modality: 2D Chest X-Ray (Projection Radiography)
│   ├── Scale: 377,110 projection radiography images
│   ├── Target: High-throughput thoracic screening & disease classification
│   └── Role: Pretraining projection spatial representations
│
└── The Cancer Imaging Archive (TCIA)
    ├── Modality: Multi-parametric MRI & CT
    ├── Scale: ~120,000 volumetric series
    ├── Cohorts: Brain (LGG/GBM), Kidney (TCGA-KIRC), Lung, Prostate
    └── Role: Multi-parametric soft-tissue characterization & lesion identification
```

---

## 2. Modality & Organ Coverage Summary
- **Thoracic System:** Lungs, Heart, Mediastinum, Ribs (CT, X-Ray)
- **Abdominal System:** Liver, Kidneys, Spleen, Pancreas (CT, MRI)
- **Central Nervous System:** Brain soft-tissue structures, Glioma sub-regions (MRI)
- **Pelvic & Musculoskeletal:** Prostate, Pelvic Bones (MRI, CT)
