# PanOrganNet-V3: Production-Grade Multi-Modal Expansion & Retraining Plan

## 1. Diagnosis of V2 Bottlenecks (Why Accuracy Was 69.82%)
1. **Only 3 Cohorts Used in V2 (`5,749` samples):** `tcga_lgg` (Brain MRI), `nih_cxr14` (Pulmonology CXR), and `busi` (Breast US).
2. **Task-Loss Imbalance:** In V2, `seg_loss` (`~0.99`) outweighed `cls_loss` (`~0.01–0.06`) by ~20–90x, causing Early Stopping (`val_loss`) to select **Epoch 3** (`best.pt`) before the classification head converged.
3. **Class Imbalance Over-Prediction:** Test classes were skewed (`557` Class-0, `235` Class-1, `33` Class-2). `AsymmetricFocalLoss` over-predicted Class 1/2 (`220` Class-0 false positives), lowering accuracy to `69.82%` despite `0.9002` ROC-AUC and `78.38%` Sensitivity (meanwhile `Ablation A2` reached `88.00%` accuracy and `0.9648` AUC).

---

## 2. Dataset Expansion (3 ➔ 7+ Multi-Organ Cohorts, Already on SSD)
All additional cohorts below are **already stored locally on `/Volumes/Shoaib/PanOrgan_V2/V1_archive/dataset/raw/`** (zero extra download wait):

| # | Dataset | Modality | Organ / Specialty | Size on SSD | Primary Task |
| :---: | :--- | :--- | :--- | :---: | :--- |
| 1 | `tcga_lgg` | **MRI** | Brain (Neuro-Oncology) | Already Sharded | Joint Seg + Cls |
| 2 | `nih_cxr14` | **CXR** | Lung / Thorax (Pulmonology) | Already Sharded | Multi-label Cls |
| 3 | `busi` | **US** | Breast (Oncology) | `258 MB` | Joint Seg + Cls |
| 4 | `CBIS_DDSM` | **MG** | Breast Mammography | `712 MB` | Joint Seg + Cls |
| 5 | `Ultrasound_Nerve` | **US** | Neck / Brachial Plexus | `2.2 GB` | Segmentation |
| 6 | `PROSTATEx` | **MRI** | Prostate (Multiparametric) | `1.2 GB` | Classification / Seg |
| 7 | `TCGA_KIRC` | **CT / MRI** | Kidney (Renal Carcinoma) | `2.3 GB` | Classification / Seg |
| 8 | `pulmonology` | **CXR** | Pulmonology (Pneumonia/TB) | `865 MB` | Binary / Multi Cls |

### Dataset Organization & Zero-Leakage Protocol
- Patient-level stratified splitting (`GroupShuffleSplit` on `patient_id`) with automated leakage audit (`0` overlap).
- Modality-balanced 256 MB WebDataset shards stored directly in `/Volumes/Shoaib/PanOrgan_V3/preprocessed_shards/`.
- Balanced sampling (`BalancedModalityBatchSampler`, `temperature=0.5`) + inverse-frequency class weights capped at `[0.5, 5.0]`.

---

## 3. Anti-Overfitting & Top-Notch Accuracy/Sensitivity/Specificity Architecture
1. **Task-Normalized Multi-Task Loss:** Normalize `seg_loss` and `cls_loss` scales (or learnable homoscedastic uncertainty weights) so segmentation no longer drowns out classification gradients.
2. **Composite Checkpoint Selection (`best.pt`):** Replace raw `val_loss` with:
   $$\text{Score}_{\text{val}} = 0.5 \cdot \text{Val\_Dice} + 0.5 \cdot \text{Val\_Macro\_F1}$$
   guaranteeing the saved checkpoint is simultaneously strong at Segmentation **and** Classification.
3. **Two-Phase Stage-2 Schedule:**
   - **Phase 2A (Joint Training, 20 epochs, `patience=7`):** Train `OmniModalityStem + LoRA + MT-MoE + SACR + Decoders` with `MixUp/CutMix` (`p=0.3`), `DropPath (0.2)`, `Weight Decay (0.05)`, and `Label Smoothing (0.05)`.
   - **Phase 2B (Frozen-Backbone Head Calibration, 6 epochs):** Freeze `OmniModalityStem`, `Backbone`, and `UniversalMaskDecoder` (preserving 100% of Segmentation Dice & MNAR robustness), and fine-tune `ClinicalPathologyHead` with class-balanced focal cross-entropy + cosine LR.
4. **Post-Training Calibration & Conformal Guarantee:** Fit `TemperatureScaler` + `ModalityConditionalConformal (MC2-CP)` per modality on `val`, then evaluate on untouched `test` (`1,000` bootstrap CIs).

---

## 4. Target Benchmarks & Power-Loss Execution
- **Target Performance:** Segmentation Dice `0.80–0.85+`, Classification Accuracy `88–93%+`, Sensitivity `>= 85%`, Specificity `>= 90%`, ROC-AUC `>= 0.95`, MC2-CP Coverage `>= 90%`.
- **Execution:** Runs 100% locally on **Apple M4 GPU (MPS)** + `/Volumes/Shoaib/PanOrgan_V3/` with 50-step `fsync` crash-resume (`caffeinate` enabled).
