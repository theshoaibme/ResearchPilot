# Phase 3: Data Preprocessing & Augmentation Pipeline

---

To enable unified feature extraction across diverse scanners, acquisition protocols, and dimensions, we define an automated pipeline:

## 1. Spatial Voxel Resampling
All 3D volumes (CT & MRI) are resampled to an isotropic voxel resolution:
$$\text{Target Resolution} = (1.5\,\text{mm}, 1.5\,\text{mm}, 1.5\,\text{mm})$$
- **Trilinear Interpolation:** Applied to continuous image intensity volumes.
- **Nearest-Neighbor Interpolation:** Applied to discrete anatomical segmentation masks to prevent label corruption.

---

## 2. Modality Intensity Normalization
- **Computed Tomography (CT):** Intensity windowing based on Target Hounsfield Units (HU):
  - Soft Tissue: $[-150, 250]\,\text{HU}$
  - Lung: $[-1000, 200]\,\text{HU}$
  - Bone: $[200, 1000]\,\text{HU}$
  Target intensities are rescaled to $[0, 1]$ or normalized using Z-score $\mu=0, \sigma=1$.
- **Magnetic Resonance Imaging (MRI):** Z-score normalization per volumetric sequence after brain/organ extraction.
- **X-Ray (2D Projections):** Histogram equalization and min-max scaling to $[0, 1]$.

---

## 3. Dynamic Augmentations
- **Affine Transformations:** Random rotation within $[-15^{\circ}, +15^{\circ}]$.
- **Elastic Deformations:** Simulates physiological variations in soft tissue elasticity.
- **Intensity Jittering:** Simulates scanner noise variations.
