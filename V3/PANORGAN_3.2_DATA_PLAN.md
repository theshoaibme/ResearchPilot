# PanOrgan 3.2 — Data Gap Audit & Class Expansion Plan

**Goal:** Resolve class imbalance, low unique-patient diversity, and `100%` conformal abstention cohorts observed in `PanOrgan 3.1` (`v31_complete_thesis_report.json`), and evaluate adding new clinical classes.

---

## 1. Exact Data Gaps in `PanOrgan 3.1` (`11,735` Samples)

### A. Severe 4-Class Imbalance (`master_manifest.parquet`)
| Unified Class | Train | Val | Test | Total | Share | Current Sources | Gap Severity |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **Class `0`** | 3,880 | 896 | 819 | **5,595** | 53.1% | `tcga_lgg`, `nih_cxr14`, `ultrasound_nerve`, `pulmonology_cxr`, `tcga_kirc_ct`, `busi`, `cbis_ddsm` | Adequate |
| **Class `1`** | 2,593 | 553 | 606 | **3,752** | 35.6% | `tcga_lgg`, `tcga_kirc_ct`, `ultrasound_nerve`, `pulmonology_cxr`, `busi`, `cbis_ddsm` | Adequate |
| **Class `2`** | 460 | 106 | 124 | **690** | **6.5%** | Only `pulmonology_cxr` (500) + `busi` (190) | **Critical Shortage** |
| **Class `3`** | 333 | 87 | 80 | **500** | **4.7%** | Only `pulmonology_cxr` (500) | **Critical Shortage** |
| **Unlabeled (`-1`)** | 812 | 197 | 189 | **1,198** | — | `prostatex_mri` (SSL only) | No supervised labels |

### B. Cohort & Patient-Diversity Bottlenecks
1. **`cbis_ddsm` (`MG`):** Only **`88` samples (`45` patients; `15` test)** -> `100%` conformal abstention.
2. **`tcga_kirc_ct` (`CT`):** `1,200` slices from only **`15` unique patients (`3` test patients)** -> `85.19%` coverage (below `90%` target) and `100%` abstention.
3. **`ultrasound_nerve` (`US`):** `1,500` images from **`47` patients** -> `Dice = 0.6402`, `100%` abstention.
4. **`busi` (`US`):** Only `102` normal cases (`Class 0`) vs `361` benign / `190` malignant; `HD95 = 37.30 mm`.
5. **`prostatex_mri` (`MRI`):** `1,198` samples (`25` patients) remain unlabeled (`-1`).

---

## 2. `PanOrgan 3.2` Data Expansion Targets

| Target Gap | Recommended Public Clinical Datasets (Official Labels) | Target Gain | Impact on `3.2` |
| :--- | :--- | :--- | :--- |
| **Mammography (`MG`)** | Full **CBIS-DDSM** (`~2,600+` scans) + **VinDr-Mammo** / **CMMD** | `88` -> **`2,500+` scans** (`500+` patients) | Eliminates `15`-sample `MG` bottleneck & reduces `100%` abstention |
| **Renal / Abdominal `CT`** | **KiTS23** (`489` patients, official kidney tumor seg + grade/stage) + more **TCGA-KIRC** patients | `15` -> **`300+` unique patients** | Fixes `85.19%` coverage & adds `CT` segmentation masks |
| **Class `2` & Class `3` (`CXR`/`CT`)** | **CheXpert** / **RSNA Pneumonia** / **COVIDx CXR** (severe/critical thoracic tiers) | Class `2`: `690` -> **`2,500+`**; Class `3`: `500` -> **`2,000+`** | Eliminates `11:1` class imbalance against Class `0` |
| **Ultrasound (`US`)** | **UDIAT** / **BrEaST** (breast US) + **BUSI** boundary-weighted Dice/Focal loss | `653` -> **`1,200+` US patients** | Reduces `BUSI` `HD95` (`37.30 mm`) & improves `US` generalization |
| **Prostate `MRI` (`-1` -> Labeled)** | **PI-CAI** (`1,500` prostate MRI cases with official ISUP/csPCa labels) | Replaces unlabeled `prostatex_mri` gap | Unlocks supervised prostate `MRI` evaluation |

---

## 3. Evaluating Extra Clinical Classes (`4` vs `6–8` Classes)

### Option A — Keep `4` Unified Severity Tiers (Fast Warm-Start from `3.1`)
- Keep `num_classes = 4` (`0: Normal/No-Finding`, `1: Benign/Low-Risk`, `2: Malignant/Moderate`, `3: High-Grade/Critical/Severe`).
- **Pros:** Zero head reset; warm-starts directly from `PanOrgan 3.1` (`715 MB` `best.pt`), immediately fixes Class `2` (`6.5%`) and Class `3` (`4.7%`) scarcity, and supports Mondrian group-wise conformal calibration.

### Option B — Expand to `6` or `8` Clinical Classes (Recommended if Adding New Organs/Pathologies)
- Expand classification head (`num_classes = 6` or `8`):
  1. `Normal / Healthy`
  2. `Benign Lesion / Cyst`
  3. `Malignant Tumor (Oncology)`
  4. `Bacterial Pneumonia / Consolidation`
  5. `Viral / COVID-19 Pneumonia`
  6. `Tuberculosis / Pleural Effusion`
  7. `High-Grade Carcinoma (e.g., KIRC G3/G4, Glioma)`
  8. `Clinically Significant Prostate Lesion (PI-CAI)`
- **Implementation:** Load `PanOrgan 3.1` backbone (`strict=False`), replace only the final `cls_head` linear layer (`4 -> 8`), train with `MixUp` + class-balanced Focal/Cross-Entropy loss + **Group-wise (Mondrian) Conformal Calibration** + **per-modality Temperature Scaling** (fixing the `0.0611 -> 0.0784` ECE regression).

---

## 4. Recommended `PanOrgan 3.2` Execution Order
1. **Curate & Shard Priority Additions on `/Volumes/Shoaib/PanOrgan_V3/`:** `KiTS23` (`CT`), full `CBIS-DDSM` / `CMMD` (`MG`), and Class `2/3` `CXR` cohorts with strict `patient_id` zero-leakage audit.
2. **Calibration Upgrade:** Switch from single global temperature (`T = 1.0354`) to **per-modality temperature scaling** + **group-wise `MC2` thresholds** so every modality meets `>= 90%` coverage.
3. **Train `PanOrgan 3.2`:** Warm-start from `3.1` `best.pt`, verify `val_loss` convergence, and benchmark against `3.1` (`Overall Dice = 0.8465`, `AUROC = 0.9593`).
