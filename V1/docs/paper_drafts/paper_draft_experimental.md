## IV. Experimental Setup & Data Curation

### 4.1 Data Sources & Cohort Compilation
To train and validate **Pan-Organ Net**, we assembled a heterogeneous dataset representing various anatomical regions and diagnostic targets:

1. **TotalSegmentator Dataset:** Comprising 1,204 high-resolution CT volumes with manual segmentations for 117 organs and structures. This serves as the benchmark dataset for multi-organ localization and structural spatial integrity.
2. **MIMIC-CXR Database:** Containing 377,110 projection chest radiography images (2D X-ray). This provides the baseline representation for high-throughput thoracic screening.
3. **The Cancer Imaging Archive (TCIA):** Incorporating heterogeneous multi-parametric magnetic resonance imaging (MRI) scans and CT scans targeting prostate, brain (LGG/GBM), kidney (TCGA-KIRC), and lung cohorts, representing approximately 120,000 volumetric series.

These sources aggregate to over 500,000 patient studies across 10 anatomical systems (thoracic, abdominal, pelvic, musculoskeletal, central nervous system, etc.).

### 4.2 Preprocessing and Augmentation Pipeline
To align datasets across hardware manufacturers and formats, we implemented a standardized preprocessing pipeline:
*   **Voxel Spacing Standardization:** All 3D volumes (CT and MRI) are resampled to an isotropic voxel resolution of $1.5\text{mm} \times 1.5\text{mm} \times 1.5\text{mm}$ using trilinear interpolation for images and nearest-neighbor interpolation for segmentation masks.
*   **Intensity Normalization:** Volumes are normalized using modality-specific z-score normalization. For CT, windowing is applied to target Hounsfield Unit (HU) ranges of interest (soft tissue: $[-150, 250]$, lung: $[-1000, 200]$, bone: $[200, 1000]$) prior to scaling.
*   **Data Augmentations:** Dynamic augmentations are applied on-the-fly during training, including random affine transformations, elastic deformations, random 3D rotations ($[-15^{\circ}, 15^{\circ}]$), and random intensity scaling to simulate variations in scanner physics.

### 4.3 Training Details & Hyperparameters
The network backbone was instantiated as a volumetric Vision Transformer (Swin Transformer 3D) containing approximately 86 million parameters. Pre-training was executed on a cluster of 8 NVIDIA H100 (80GB) GPUs using PyTorch and Hugging Face Accelerate. The training hyperparameters are detailed in the table below:

| Hyperparameter | Pre-training Value | Fine-tuning Value |
| :--- | :--- | :--- |
| **Optimizer** | AdamW | AdamW |
| **Base Learning Rate** | $1.5 \times 10^{-4}$ | $5.0 \times 10^{-5}$ |
| **Weight Decay** | $0.05$ | $0.01$ |
| **Batch Size (Global)** | $128$ | $32$ |
| **Warmup Epochs** | $40$ | $5$ |
| **Total Epochs** | $800$ | $100$ |
| **SGM Temperature ($\tau$)**| $0.15$ | - |

For evaluation, we employ **Linear Probing** (freezing the backbone weights and training a shallow classifier) and **Parameter-Efficient Fine-Tuning** using Low-Rank Adaptation (LoRA) to validate zero-shot and few-shot transfer capabilities.
