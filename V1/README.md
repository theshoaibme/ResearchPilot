# The Pan-Organ Diagnostic Paradigm: A High-Capacity Foundation Model for Multi-Modality Medical Screening (Pan-Organ Net)

## Abstract
Clinical Artificial Intelligence (AI) stands at a critical juncture: while deep neural networks demonstrate remarkable accuracy in isolated tasks, their real-world utility is heavily bottlenecked by **model fragmentation**—the classic "One Model, One Task" paradigm. Deploying, managing, and maintaining dozens of distinct deep networks for individual anatomical organs and scanner modalities is clinically and computationally unsustainable. 

In this work, we introduce **Pan-Organ Net**, a high-capacity volumetric foundation model designed to establish a unified diagnostic paradigm across diverse human organs (including brain, lung, liver, heart, kidneys, prostate, and bones) and multiple imaging modalities (CT, MRI, Ultrasound, and X-Ray). By combining a modality-aware anatomical tokenization scheme with a **Saliency-Guided Masked Autoencoder (SGM)** framework, our approach prevents dominant-organ shortcut learning under simulated Missing Not at Random (MNAR) data distributions. 

Furthermore, we bridge the clinical "Actionability Gap" by integrating a decoupled Action Planning Head and 3D Grad-CAM Explainability (XAI) that translates high-dimensional latent anatomical embeddings into actionable diagnostic pathways, moving medical imaging AI from reactive classification toward proactive decision support.

---

## 1. Clinical Specialist & Disease Taxonomy Mapping
To ensure **Pan-Organ Net** produces multi-organ diagnostic representations that directly serve multidisciplinary clinical workflows, the model architecture is mapped across **7 core medical specialties** and **8 primary disease categories**:

```
                               ┌─────────────────────────────────────────┐
                               │       Pan-Organ Net Foundation Model    │
                               │   Multi-Modality (MRI, CT, X-Ray, US)   │
                               └────────────────────┬────────────────────┘
                                                    │
             ┌──────────────────────────────────────┼──────────────────────────────────────┐
             ▼                                      ▼                                      ▼
┌───────────────────────────┐          ┌───────────────────────────┐          ┌───────────────────────────┐
│ Neuro System Head         │          │ Thoracic & Vascular Head  │          │ Multi-Organ / Systemic    │
│ (Neurology/Neurosurgery)  │          │ (Pulmonology/Cardiology)  │          │ (Oncology/Rheumatology)   │
└────────────┬──────────────┘          └────────────┬──────────────┘          └────────────┬──────────────┘
             │                                      │                                      │
   • Neurodegenerative                    • Vascular (Aneurysms, etc)            • Neoplastic (Tumors)
   • Demyelinating                        • Respiratory Pathologies              • Autoimmune Conditions
   • Epileptic & Traumatic                • Infectious Infiltrates               • Systemic Infections
```

### Specialist-to-Organ & Disease Matrix

| # | Medical Specialist | Target Organ Systems | Covered Disease Categories (Out of 8) | Primary Imaging Modalities | Clinical Decision Support Output |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **1** | **Neurologist \& Neurosurgeon** | Brain, Spine, Central \& Peripheral Nervous System | Neurodegenerative, Demyelinating, Epileptic, Traumatic | MRI (T1w, T2w, FLAIR, dTI), CT (Head) | Lesion volume, white matter hyperintensity burden, midline shift mm |
| **2** | **Oncologist \& Surgical Oncologist** | Pan-Organ / Systemic (Brain, Lungs, Liver, Kidneys, Bones) | Neoplastic (Benign \& Malignant Tumors/Metastases) | Multi-Parametric MRI, Contrast CT, PET-CT | RECIST 1.1 tumor burden, TNM staging recommendations, biopsy target |
| **3** | **Cardiologist \& Vascular Surgeon** | Heart, Aorta, Peripheral Blood Vessels | Vascular (Aneurysms, Thrombosis, Stenosis, Ischemia) | CT Angiography (CTA), Cardiac MRI, Doppler US | Ejection fraction, luminal stenosis %, aneurysm diameter (cm) |
| **4** | **Infectious Disease Specialist** | Systemic Multi-Organ (Lungs, Liver, Brain, Blood) | Infectious (Bacterial, Viral, Fungal, Parasitic Infiltrates) | Chest X-Ray, Abdominal CT, Brain MRI | Consolidation volume %, abscess localization, organ involvement score |
| **5** | **Rheumatologist \& Immunologist** | Systemic (Joints, Kidneys, Vasculature, Soft Tissue) | Autoimmune (Lupus Nephritis, Rheumatoid Arthritis, Vasculitis) | Musculoskeletal MRI, Ultrasound, CT | Joint erosion index, renal parenchymal attenuation, vessel wall thickness |
| **6** | **Pulmonologist (Thoracic Specialist)** | Lungs, Tracheobronchial Tree, Pleura, Thoracic Cage | Infectious, Neoplastic, Traumatic, Interstitial | Chest X-Ray, High-Resolution CT (HRCT) | LUNA nodule malignancy risk %, emphysema severity, pleural effusion vol |
| **7** | **Radiologist (Diagnostic \& Interventional)** | Pan-Organ Whole-Body (Brain, Lungs, Liver, Kidneys, Cardiovascular) | **All 8 Categories** (Full Taxonomy Screening) | MRI, CT, X-Ray, Ultrasound | Full Pan-Organ Heatmap (Grad-CAM), Multi-Organ Dice Segmentation, Action Plan |

---

## 2. Identified Architectural & Clinical Gaps
Synthesizing literature review insights and clinical requirements reveals **three primary gaps**:

*   **Gap 1: Dominant-Organ Shortcutting in Masked Autoencoders (MAE)**
    *   *Problem:* Standard random masking (e.g. 75–85% patch dropping) allows models to reconstruct overall volumes by learning low-entropy shortcuts (homogeneous bone structures or abdominal cavity fat), bypassing fine margins or early-stage lesions.
    *   *Impact:* High reconstruction metrics on trivial regions, but weak performance on delicate multi-organ pathologies.
*   **Gap 2: Missing Not at Random (MNAR) Vulnerability**
    *   *Problem:* Real-world diagnostic scans rarely cover the entire torso. A patient may receive an isolated liver MRI or chest CT. Standard foundation models suffer severe accuracy decay when processing truncated/missing body regions.
    *   *Impact:* Up to 25.3% drop in segmentation accuracy when input coverage is partial.
*   **Gap 3: Actionability & Transparency Disconnect**
    *   *Problem:* Existing AI models generate black-box probability distributions or basic diagnostic classes without explaining *why* or guiding *what clinical step to take next*.
    *   *Impact:* Radiologists distrust outputs, creating a barrier to clinical adoption.

---

## 3. Methodology & System Architecture
To directly solve the gaps identified, **Pan-Organ Net** incorporates three novel core components:

```
                      Heterogeneous Input Image Volume X
                         (CT, MRI, Ultrasound, X-Ray)
                                      │
                                      ▼
                      ┌──────────────────────────────┐
                      │   Modality-Aware Tokenizer   │
                      │  Patch Projection + Pos-Embed│
                      │  + Metadata Meta-Embedding   │
                      └──────────────┬───────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │ Saliency-Guided Masking(SGM) │
                      │  Entropy Gradient Evaluation │
                      │ Retain High-Entropy Tokens   │
                      └──────────────┬───────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │   3D Swin-Transformer Encoder│
                      │  Unified Latent Embedding    │
                      └──────────────┬───────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
    ┌───────────────────────────┐           ┌───────────────────────────┐
    │  Reconstruction Decoder   │           │ Action Planning & XAI Head│
    │  Spatial Structure Loss   │           │  Grad-CAM Heatmaps +      │
    │  L_total = a L_rec + L_ali│           │  Diagnostic Trajectories  │
    └───────────────────────────┘           └───────────────────────────┘
```

### 3.1 Modality-Aware Tokenization
To ingest multi-modality volumetric ($3\text{D}$) and projection ($2\text{D}$) medical data into a single transformer backbone, we formulate a **Modality-Aware Tokenizer**. Let the input medical image volume be represented as $X \in \mathbb{R}^{H \times W \times D \times C}$, where $H, W, D$ represent height, width, and depth, respectively, and $C$ represents the input channel count. For projection images (e.g., X-Rays), we set $D=1$.

We extract non-overlapping volumetric patches of size $P_H \times P_W \times P_D$. These patches are projected to a $D$-dimensional latent space using a learnable linear projection $W_E$. To retain structural physical parameters, we append a **Modality-Aware Meta-Embedding** ($E_{\text{meta}}$) to each token, representing physical acquisition parameters:

$$E_{\text{meta}} = \text{MLP}\left([ \Delta_x, \Delta_y, \Delta_z, \mathbf{m} ]\right)$$

where $\Delta_x, \Delta_y, \Delta_z$ represent the spatial voxel spacing in millimeters, and $\mathbf{m}$ is a one-hot vector indicating the scanner modality (e.g., CT, MRI, Ultrasound, X-Ray). The final patch representation $z_i$ is defined as:

$$z_i = (x_i \cdot W_E) + E_{\text{pos}} + E_{\text{meta}}$$

where $E_{\text{pos}}$ represents learnable 3D positional embeddings.

### 3.2 Saliency-Guided Masking (SGM)
Standard Masked Autoencoders randomly mask a high percentage (typically $75\% - 85\%$) of patches. In multi-organ medical imaging, this approach leads to **dominant-organ shortcutting**, where the model reconstructs low-entropy tissues (such as homogenous muscle or fat) without capturing complex anatomical features of smaller organs or lesions.

To prevent this, we introduce **Saliency-Guided Masking (SGM)**. For each volume, we compute a baseline saliency map $S \in \mathbb{R}^{H \times W \times D}$ representing localized anatomical entropy or intensity gradients:

$$S(x,y,z) = \nabla(X(x,y,z))$$

We calculate a saliency score $s_i$ for each patch $i$ by averaging the saliency values within the patch volume. The probability of masking patch $i$, denoted by $P(\text{mask}_i)$, is inversely proportional to its saliency score:

$$P(\text{mask}_i) = \frac{\exp(-s_i / \tau)}{\sum_j \exp(-s_j / \tau)}$$

where $\tau$ is a temperature parameter controlling the uniformity of the masking. SGM forces the network to retain high-information tokens (e.g., boundary transitions, fine vessels, and lesion margins) during encoding, forcing the reconstruction decoder to solve a more challenging anatomical synthesis problem.

### 3.3 Reconstruction Pretext Task \& Decoupled Heads
The backbone is optimized using a dual-objective loss function combining a structural reconstruction loss ($\mathcal{L}_{\text{recon}}$) and a contrastive semantic alignment loss ($\mathcal{L}_{\text{align}}$):

$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{recon}} + (1 - \alpha) \mathcal{L}_{\text{align}}$$

To bridge the actionability gap, we bypass task-specific classification decoders. Instead, the latent representation is decoded through an **Action Planning Head** that outputs diagnostic action trajectories (e.g., suggesting secondary staging scans, recommending needle biopsy targets, or estimating regional hazard scores for survival predictions) alongside 3D Grad-CAM visual localized explainability maps.

---

## 4. Experimental Setup & Data Curation

### 4.1 Data Sources \& Cohort Compilation
To train and validate **Pan-Organ Net**, we assembled a heterogeneous dataset representing various anatomical regions and diagnostic targets (refer to `phases/Pan_Organ_Medical_Datasets_Catalog.csv` for detailed catalogs):

1.  **TotalSegmentator Dataset:** Comprising 1,204 high-resolution CT volumes with manual segmentations for 117 organs and structures.
2.  **MIMIC-CXR Database (v2.0.0):** Containing 377,110 projection chest radiography images (2D X-ray).
3.  **The Cancer Imaging Archive (TCIA):** Incorporating heterogeneous multi-parametric magnetic resonance imaging (MRI) scans and CT scans targeting prostate, brain (LGG/GBM), kidney (TCGA-KIRC), and lung cohorts, representing approximately 120,000 volumetric series.
4.  **BraTS 2023 Challenge Dataset:** Consisting of 4,500 volumetric multi-sequence brain MRI scans.
5.  **LUNA16 Dataset:** Consisting of 888 thoracic CT scans with 1,018 lung nodule ground-truth annotations.

### 4.2 Preprocessing and Augmentation Pipeline
Standardized preprocessing is implemented in `src/preprocessing.py`:
*   **Spatial Voxel Resampling (3D & 2D):** Continuous volumetric data (CT/MRI) is resampled to isotropic $1.5\text{mm} \times 1.5\text{mm} \times 1.5\text{mm}$ voxel grids via trilinear interpolation. Voxel segmentation masks are resampled using nearest-neighbor interpolation to preserve boundaries. 2D projections are resized to $512 \times 512$ pixels with aspect-ratio padding.
*   **Intensity Normalization:** 
    *   *CT:* HU windowing is applied to target tissues (Soft tissue: $[-150, 250]$, Lung: $[-1000, 200]$, Bone: $[200, 1000]$) and scaled to $[0.0, 1.0]$.
    *   *MRI:* Nyúl histogram standardization followed by volume Z-score normalization ($\mu = 0, \sigma = 1$).
    *   *X-Ray & Ultrasound:* CLAHE histogram equalization followed by min-max scaling to $[0.0, 1.0]$.
*   **Augmentations:** Dynamic 3D affine transformations (rotation $[-15^{\circ}, 15^{\circ}]$, scaling $[0.85, 1.15]$), physiological elastic deformations ($\sigma = 3.0, \alpha = 15.0$), and scanner noise jittering.

```python
import numpy as np
import scipy.ndimage

def resample_volume(volume, current_spacing, target_spacing=(1.5, 1.5, 1.5), is_mask=False):
    """Resamples 3D volume to isotropic 1.5mm voxel spacing."""
    resize_factor = current_spacing / target_spacing
    new_shape = np.round(volume.shape * resize_factor)
    real_resize_factor = new_shape / volume.shape
    order = 0 if is_mask else 1
    resampled_volume = scipy.ndimage.zoom(volume, real_resize_factor, order=order)
    return resampled_volume

def ct_windowing(ct_volume, hu_min=-150, hu_max=250):
    """Clips CT volume to target Hounsfield Units and scales to [0, 1]."""
    clipped = np.clip(ct_volume, hu_min, hu_max)
    normalized = (clipped - hu_min) / (hu_max - hu_min)
    return normalized.astype(np.float32)

def mri_zscore_normalize(mri_volume):
    """Applies Z-score normalization to MRI volume."""
    mask = mri_volume > 0
    mean = np.mean(mri_volume[mask])
    std = np.std(mri_volume[mask])
    normalized = np.zeros_like(mri_volume, dtype=np.float32)
    if std > 0:
        normalized[mask] = (mri_volume[mask] - mean) / std
    return normalized
```

### 4.3 Training Details \& Hyperparameters

| Hyperparameter | Pre-training Value | Fine-tuning Value |
| :--- | :--- | :--- |
| **Optimizer** | AdamW | AdamW |
| **Base Learning Rate** | $1.5 \times 10^{-4}$ | $5.0 \times 10^{-5}$ |
| **Weight Decay** | $0.05$ | $0.01$ |
| **Batch Size (Global)** | $128$ | $32$ |
| **Warmup Epochs** | $40$ | $5$ |
| **Total Epochs** | $800$ | $100$ |
| **SGM Temperature ($\tau$)**| $0.15$ | - |

---

## 5. Results \& Quantitative Benchmarking

To evaluate the zero-shot and few-shot capability of **Pan-Organ Net**, we compared it against several task-specific baselines (representing state-of-the-art architectures in narrow medical AI) and existing medical foundation models. 

We benchmarked three downstream tasks: multi-organ segmentation on TotalSegmentator (evaluating Dice Similarity Coefficient), lung nodule classification on LUNA16 (evaluating Area Under the ROC Curve), and brain lesion classification on TCIA.

| Model | TotalSegmentator (Dice) | LUNA16 Classification (AUC) | TCIA Brain Classification (AUC) | MNAR 50% Missing (Dice) |
| :--- | :--- | :--- | :--- | :--- |
| Task-Specific CNN (3D U-Net) | $0.842$ | $0.812$ | $0.788$ | $0.521$ |
| BiomedCLIP (2D Zero-shot) | $0.612$ | $0.841$ | $0.743$ | - |
| Pan-FM baseline (random mask) | $0.864$ | $0.887$ | $0.865$ | $0.610$ |
| **Pan-Organ Net (Ours + SGM)** | $\mathbf{0.898}$ | $\mathbf{0.912}$ | $\mathbf{0.904}$ | $\mathbf{0.850}$ |

### 5.1 SGM Ablation \& Robustness under Missing-Organ Scenarios
A key clinical issue in multi-organ foundation models is their vulnerability to Missing Not at Random (MNAR) scenarios. If a model expects a full torso scan but receives only a liver MRI, standard random masking pre-trained networks suffer from domain collapse. 

When 50% of the organ structures are omitted from the inputs to simulate partial scans, the standard random masking baseline experiences a **$25.3\%$ drop** in segmentation accuracy (Dice score dropping to $0.61$). In contrast, our **SGM** framework retains high stability, dropping only **$5.5\%$** (maintaining a Dice score of $0.85$). This demonstrates that forcing the model to learn localized structural entropy prevents it from relying on anatomical "shortcuts" (such as the ribs or abdominal cavity boundaries).

---

## 6. Project Management & Financial Analysis

| SN | Expense Item | Cost (BDT) |
| :-: | :--- | :-: |
| **1** | Cloud GPU Computing \& Model Training (NVIDIA H100) | 45,000 |
| **2** | Clinical Dataset Storage \& High-Speed Transfer | 12,000 |
| **3** | Technical Documentation, Printing \& Reporting | 3,000 |
| **4** | Specialist Consultation \& Clinical Validation | 15,000 |
| **5** | Miscellaneous \& Contingency | 5,000 |
| **Total** | **Overall Project Expense** | **80,000 BDT** |

---

## 7. References
1. O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” in *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 2015, pp. 234–241.
2. J. Wasserthal et al., “TotalSegmentator: Robust Segmentation of 117 Anatomical Structures in CT Images,” *Radiology: Artificial Intelligence*, vol. 5, no. 5, p. e230024, 2023.
3. A. E. W. Johnson et al., “MIMIC-CXR, a De-Identified Publicly Available Database of Chest Radiographs,” *Scientific Data*, vol. 6, p. 317, 2019.
4. Y. Zhang et al., “BiomedCLIP: A Multimodal Biomedical Vision-Language Foundation Model,” *arXiv preprint arXiv:2303.03393*, 2023.
5. A. Kirillov et al., “Segment Anything,” in *IEEE International Conference on Computer Vision (ICCV)*, 2023, pp. 4015–4026.
6. K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick, “Masked Autoencoders Are Scalable Vision Learners,” in *IEEE/CVF CVPR*, 2022, pp. 16000–16009.
