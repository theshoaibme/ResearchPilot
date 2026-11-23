# V3 Roadmap — Post-Defense Full-3D Whole-Body Scaling

`V3/` will extend the validated `V2/` framework (`OmniStem`, `PhysicsFiLM`, `MA-UGM`, `MT-MoE`, `SACR`, and `$MC^2$-CP`) to full native 3D volumetric patches (`128×128×96`) for Q1 journal submission:

1. **Whole-Body 3D CT:** Add `TotalSegmentator v2` (Zenodo `10047292`, 1,204 CT scans, 117 anatomical structures).
2. **Whole-Body 3D MRI:** Add `TotalSegmentator-MRI` (80 anatomical structures across heterogeneous MRI pulse sequences).
3. **3D Renal Oncology:** Add `KiTS23` (489 contrast-enhanced CT volumes with kidney, renal tumor, and cyst segmentation masks).
4. **3D Multi-Parametric Prostate Oncology:** Pair full `PROSTATEx` DICOM volumes with official `ProstateX-Findings-Train.csv` (`ClinSig`) lesion coordinates.
5. **Compute Requirement:** Reuses the exact `V2` 256 MB Drive shard format and rolling local-SSD cache, targeting multi-session A100 40GB/80GB training (~40–60 GPU hours).
