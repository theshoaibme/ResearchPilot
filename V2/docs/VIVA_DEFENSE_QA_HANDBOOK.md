# Thesis Defense & Viva Q&A Technical Handbook (Pan-Organ Net V2)

This handbook consolidates anticipated thesis committee questions across clinical motivation, dataset provenance, V1→V2 evolution, architecture mathematics, validation rigor, and limitations.

---

## Section A: Dataset Provenance & Integrity Questions

### Q1. Where did each dataset come from, and why were these specific 8 datasets selected?
**Answer:** Every dataset is a peer-reviewed, open-access de-identified clinical benchmark covering complementary anatomical systems and imaging physics:
- **Abdominal CT & MRI:** `AMOS22` (Ji et al., *NeurIPS* 2022, Zenodo `7155725`, CC BY 4.0) provides 15 expert-annotated abdominal organs across both CT (500 scans) and MRI (100 scans), enabling cross-modality CT↔MRI evaluation.
- **Pulmonology CXR (Internal + External):** `NIH ChestX-ray14` (Wang et al., *CVPR* 2017, 112,120 images, official patient splits) trains multi-label thoracic pathology screening, while `RSNA Pneumonia` serves strictly as an unseen external hospital validation set.
- **Breast Oncology (Dual Modality):** `BUSI` (Al-Dhabyani et al., 2020, Ultrasound) and full curated `CBIS-DDSM` (Lee et al., *Scientific Data* 2017, Mammography) provide biopsy-verified benign/malignant pathology plus lesion ROI masks.
- **Cardiology:** `CAMUS` (Leclerc et al., *IEEE TMI* 2019, 500 patients) covers cardiac chamber segmentation (LV endocardium, epicardium, left atrium) in echocardiography.
- **Neurology (Central & Peripheral):** `TCGA-LGG` (Buda et al., 2019, 110 patients) covers brain glioma segmentation on FLAIR MRI, and `Ultrasound Nerve` (47 subjects) covers peripheral Brachial Plexus segmentation.

### Q2. Why did you archive V1 and change datasets in V2?
**Answer:** During our internal methodology audit prior to final training, we found that three V1 preprocessing scripts (`TCGA-KIRC`, `PROSTATEx`, `CBIS-DDSM`) had labeled scan/file types (e.g., CT vs. MRI or Crop vs. Mask) rather than clinical pathology, and the V1 COVID-19 X-ray split was image-level across merged hospital sources (which DeGrave et al., *Nature Machine Intelligence* 2021 proved causes site-artifact shortcuts). Rather than hiding those early baselines, we froze them transparently in `V1/` and rebuilt `V2/` with strictly verified clinical ground truths, patient-level zero-leakage assertions, and external hospital testing.

### Q3. How do you prove there is zero patient data leakage between train and test splits?
**Answer:** Every dataset parser extracts true patient or subject identifiers (`Patient ID` in NIH-CXR14, `patient_id` in CBIS-DDSM/TCGA-LGG, case ID in AMOS22/CAMUS, subject ID in Nerve US, and perceptual-hash deduplication + case grouping in BUSI). Before any shard is written, `verify_zero_patient_overlap()` asserts that the set intersection of `patient_id` across `train`, `val`, `test`, and `external_test` is strictly empty, halting the pipeline with `PatientLeakageError` otherwise, and persists `split_audit_report.json`.

---

## Section B: Architecture & Novelty Questions

### Q4. How does one single model handle both 2D X-rays/Ultrasound and 3D CT/MRI without breaking spatial dimensions?
**Answer:** Instead of passing $D=1$ 2D images into 3D pooling blocks (which causes depth collapse), `OmniModalityStem` routes 2D projections (3 channels) and 2.5D volumetric slabs ($K=5$ adjacent axial slices) through dedicated modality stems fused with **Physics-Informed FiLM** conditioned on physical voxel spacing $(\log\Delta_x, \log\Delta_y, \log\Delta_z)$, dimensionality, contrast flag, and modality one-hot vector. Volumetric segmentation metrics (3D Dice and physical-mm 3D HD95) are still reconstructed and evaluated at full 3D patient-volume level.

### Q5. Why did you replace raw-gradient Saliency-Guided Masking (SGM) with `MA-UGM`?
**Answer:** Raw intensity gradients $\|\nabla X\|$ assume clean, edge-dominated anatomy. In **Ultrasound**, acoustic speckle creates high gradients uniformly across background tissue; in **MRI**, low-frequency bias fields shift intensity gradients unpredictably. `MA-UGM` solves this by combining modality-specific foreground priors (median-despeckled gradients for Ultrasound, local tissue variance for MRI, body-masked HU gradients for CT) during warmup with **online EMA-teacher reconstruction residual uncertainty**, which automatically discovers hard pathological regions regardless of scanner physics.

### Q6. Why add Sparse Top-2 Mixture-of-Experts (`MT-MoE`)?
**Answer:** Jointly training a shared dense backbone across 5 modalities and both dense pixel segmentation and global multi-label classification causes gradient interference ("tug-of-war", as shown by *Uni-Med* at NeurIPS 2024). Conditioning a lightweight Top-2 MoE router on visual context plus physical acquisition metadata lets the network allocate specialized spatial capacity per modality/task while activating only 2 of 4 experts per forward pass and enforcing balanced utilization via an auxiliary load-balancing loss.

### Q7. How does `SACR` handle truncated (MNAR) partial-FOV scans without knowing a fixed anatomical landmark?
**Answer:** Earlier landmark-distance concepts were circular because a fixed landmark (e.g., the tracheal carina) is absent in isolated abdominal or pelvic scans. `SACR` removes that assumption by training an auxiliary regressor to predict continuous normalized body-atlas coordinates $(z, y, x) \in [0, 1]^3$ directly from local tissue appearance under 20%–50% simulated FOV slab crops. At inference, even on a truncated scan, `SACR` predicts approximate anatomical location and injects that spatial prior into the mask decoder.

### Q8. How does Pan-Organ Net V2 compare with `BiomedParse`, `SAT`, `VISTA3D`, and `TotalSegmentator`?
**Answer:** `TotalSegmentator` and `VISTA3D` excel specifically at 3D CT segmentation; `SAT` handles 3D text-prompted segmentation without 2D projection screening; `BiomedParse` parses 2D slices across 9 modalities without 3D volumetric slab context, MNAR coordinate anchoring, or conformal calibration. `Pan-Organ Net V2` uniquely integrates **2D + 2.5D multi-modality screening (segmentation + pathology classification)** with **Sparse `MT-MoE`**, **`SACR` partial-FOV robustness**, and **Modality-Conditional Conformal Prediction (`MC2-CP`)** trainable within single-GPU academic compute.

---

## Section C: Clinical Safety, Evaluation & Compute Questions

### Q9. Why isn't accuracy enough for medical AI, and how do you handle False Negatives?
**Answer:** On imbalanced screening cohorts, a trivial model predicting "normal" achieves high accuracy while missing life-threatening malignancies. We use **Asymmetric Focal Loss** ($\gamma_- = 3, \gamma_+ = 0$) and **Dice + Focal + Boundary Loss** to heavily penalize missed foreground/positive cases, and evaluate **Sensitivity at 90% and 95% Specificity**, **AUPRC**, **AUROC**, **Macro-F1**, **3D Dice**, and **physical-mm HD95** alongside **1,000-bootstrap 95% confidence intervals**.

### Q10. Why do you need Conformal Prediction (`MC2-CP`) if you already have Temperature Scaling?
**Answer:** Temperature Scaling improves global probability calibration (reducing ECE and Brier Score), but still outputs a single point prediction with no formal error guarantee on individual or degraded scans. `MC2-CP` stratifies nonconformity quantiles per `(modality, coverage_state)` group to output a **prediction set $\mathcal{C}(x)$ with a finite-sample coverage guarantee $\ge 1-\alpha$ (90%)**. When $|\mathcal{C}(x)| > 1$, the system explicitly abstains and flags the scan for specialist radiologist review.

### Q11. How do you verify your XAI heatmaps are clinically faithful rather than cosmetic?
**Answer:** Beyond visual overlays, we measure quantitative **Insertion AUC** and **Deletion AUC**: systematically occluding the highest-attributed pixels must cause predicted pathology probability to drop sharply (low Deletion AUC), whereas revealing only those pixels must recover confidence rapidly (high Insertion AUC).

### Q12. How did you train a multi-modality model on Google Colab Pro (48 GB RAM / 100 GB local SSD) without running out of memory or hitting Drive I/O limits?
**Answer:** Reading tens of thousands of small PNG/DICOM files directly over Google Drive FUSE causes API throttling. We packed all harmonized samples into **256 MB sequential `.tar` shards** stored on 5 TB Google Drive and engineered `ShardCacheManager` to stream a bounded **20 GB rolling LRU cache** onto Colab's local NVMe SSD, paired with `bfloat16`/`float16` AMP, gradient checkpointing, LoRA adapters, and atomic Drive checkpoints every epoch/500 steps for seamless session resume.

### Q13. What are the honest limitations of V2, and what is planned for V3?
**Answer:** Three clear limitations are openly documented:
1. **2.5D Slabs vs. Native 3D Context:** V2 uses 5-slice 2.5D slabs on CT/MRI to stay within single-GPU thesis compute budgets; full native 3D (`128×128×96`) scaling over `TotalSegmentator v2` (117 structures) and `KiTS23` is planned in `V3/`.
2. **Retrospective Public Cohorts:** All benchmarks are retrospective de-identified registries; prospective multi-center clinical reader studies with IRB oversight remain required prior to regulatory clearance.
3. **Assistive Scope:** The system is engineered strictly for Clinical Decision Support (pre-segmentation, triage, calibrated risk flagging), never autonomous diagnosis.
