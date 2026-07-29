# Phase 3: Comprehensive Data Preprocessing & Augmentation Pipeline

---

## 1. Automated Multimodal Preprocessing Architecture

To digest heterogeneous datasets across 8 imaging modalities (CT, MRI, X-Ray, Ultrasound, Mammography, PET-CT, WSI) into a unified tensor space for **Pan-Organ Net**, Phase 3 defines a standardized preprocessing pipeline:

```
    Heterogeneous Input Volumes (DICOM / NIfTI / PNG)
                           │
                           ▼
     ┌───────────────────────────────────────────┐
     │ 1. Spatial Voxel Resampling (3D Isotropic)│
     │    Target: 1.5mm x 1.5mm x 1.5mm          │
     └─────────────────────┬─────────────────────┘
                           │
                           ▼
     ┌───────────────────────────────────────────┐
     │ 2. Modality Intensity Normalization       │
     │    CT: HU Windowing (Soft Tissue/Lung/Bone)│
     │    MRI: Nyul / Z-Score Intensity Scaling   │
     │    X-Ray/US: CLAHE Histogram Equalization │
     └─────────────────────┬─────────────────────┘
                           │
                           ▼
     ┌───────────────────────────────────────────┐
     │ 3. Dynamic Augmentations & Patching       │
     │    Elastic Deformation + Affine Rotation  │
     │    3D Non-Overlapping Volumetric Patching │
     └─────────────────────┬─────────────────────┘
                           │
                           ▼
       Unified Modality-Aware Input Tensors
```

---

## 2. Detailed Technical Specifications

### A. Spatial Voxel Resampling (3D & 2D)
- **3D Volumetric (CT, MRI, PET-CT):** Resampled to isotropic voxel spacing:
  $$\text{Target Resolution} = (1.5\,\text{mm}, 1.5\,\text{mm}, 1.5\,\text{mm})$$
  - **Trilinear Interpolation:** Applied to continuous voxel intensity signals.
  - **Nearest-Neighbor Interpolation:** Applied strictly to discrete anatomical segmentation ground-truth masks to preserve class boundaries.
- **2D Projections (X-Ray, Ultrasound, Mammography):** Resized to $512 \times 512$ pixel grid with aspect-ratio padding.

### B. Modality-Specific Intensity Normalization
- **Computed Tomography (CT):** Target Hounsfield Unit (HU) Windowing:
  - **Soft Tissue Window:** $[-150, 250]\,\text{HU}$ (Liver, Kidneys, Spleen)
  - **Lung Window:** $[-1000, 200]\,\text{HU}$ (Pulmonary parenchyma)
  - **Bone Window:** $[200, 1000]\,\text{HU}$ (Vertebrae, Ribs)
  Rescaled linearly to $[0.0, 1.0]$.
- **Magnetic Resonance Imaging (MRI):** Nyúl histogram standardization followed by volume Z-score normalization ($\mu = 0, \sigma = 1$).
- **X-Ray & Ultrasound:** Contrast Limited Adaptive Histogram Equalization (CLAHE) followed by min-max scaling to $[0.0, 1.0]$.

### C. Dynamic Data Augmentation Suite
1. **3D Affine Transformations:** Random rotations $\theta \in [-15^{\circ}, +15^{\circ}]$, random scaling factor $s \in [0.85, 1.15]$.
2. **Physiological Elastic Deformations:** Simulates organ morphometry changes during respiration/peristalsis ($\sigma = 3.0, \alpha = 15.0$).
3. **Scanner Noise Jittering:** Additive Gaussian noise ($\sigma = 0.02$) and random Gamma contrast adjustment ($\gamma \in [0.7, 1.5]$).

---

## 3. Python Implementation Script

The executable preprocessing module is implemented in `src/preprocessing.py`:

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

---

## 4. Phase 3 Verification Status

- **Completeness Check:** Covers spatial resampling, intensity windowing/normalization across CT, MRI, X-Ray, US, and 3D augmentation protocols.
- **Phase 4 Transition:** Formulated clean tensors ready for Phase 4 Gaps Finding and Phase 5 Method Development.
