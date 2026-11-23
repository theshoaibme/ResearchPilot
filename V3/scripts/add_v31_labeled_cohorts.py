#!/usr/bin/env python3
"""Add official-label CBIS-DDSM (Mammography) and NCI-GDC labeled TCGA-KIRC (Kidney CT) shards to V3."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pydicom

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
V2_PKG = REPO_ROOT / "V2"
if str(V2_PKG) not in sys.path:
    sys.path.insert(0, str(V2_PKG))

from panorgan.data import (
    build_patient_splits,
    harmonize_ct_slice,
    harmonize_projection_2d,
    verify_zero_patient_overlap,
    write_dataset_shards,
)


def _load_dicom(path: Path):
    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept
    return arr, str(getattr(ds, "PatientID", path.parent.name) or path.parent.name)


def build_cbis_labeled(raw_dir: Path, labels_dir: Path) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    cbis_csv = pd.concat(
        [
            pd.read_csv(labels_dir / f)
            for f in ("mass_case_description_train_set.csv", "mass_case_description_test_set.csv", "calc_case_description_train_set.csv", "calc_case_description_test_set.csv")
            if (labels_dir / f).exists()
        ],
        ignore_index=True,
    )
    path_map: Dict[str, int] = {}
    for _, r in cbis_csv.iterrows():
        pid = str(r["patient_id"])
        p_str = str(r["pathology"]).upper()
        path_map[pid] = 1 if "MALIGNANT" in p_str else 0

    prompt = "classify benign versus malignant breast lesion in mammogram"
    rows, items = [], []
    for idx, p in enumerate(sorted(raw_dir.rglob("*.dcm"))):
        try:
            arr, pid_raw = _load_dicom(p)
            if arr.ndim != 2 or min(arr.shape) < 32:
                continue
            m = re.search(r"P_\d+", pid_raw)
            if not m or m.group(0) not in path_map:
                continue
            p_key = m.group(0)
            lbl = path_map[p_key]
            img_r = harmonize_projection_2d(arr, target_hw=(256, 256))
        except Exception:
            continue

        sid = f"cbis_{idx:04d}"
        pid = f"cbis_{p_key}"
        items.append({
            "sample_id": sid,
            "image": img_r,
            "mask": None,
            "meta": {
                "sample_id": sid,
                "patient_id": pid,
                "dataset": "cbis_ddsm",
                "modality": "MG",
                "is_3d": False,
                "label": lbl,
                "multilabel": [-1.0] * 14,
                "has_mask": False,
                "body_coord": [0.5, 0.5, 0.5],
                "spacing_mm": [1.0, 1.0, 1.0],
                "prompt": prompt,
            },
        })
        rows.append({
            "sample_id": sid,
            "patient_id": pid,
            "dataset": "cbis_ddsm",
            "modality": "MG",
            "shard_subdir": "cbis_ddsm",
            "label": lbl,
            "has_mask": False,
            "prompt": prompt,
        })
    return pd.DataFrame(rows), items


def build_kirc_labeled(raw_dir: Path, labels_dir: Path, max_samples: int = 1200) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    kirc_df = pd.read_csv(labels_dir / "tcga_kirc_official_clinical.csv")
    grade_map: Dict[str, int] = {}
    for _, r in kirc_df.iterrows():
        g = str(r.get("grade", "")).upper()
        if g in ("G1", "G2"):
            grade_map[str(r["patient_id"])] = 0
        elif g in ("G3", "G4"):
            grade_map[str(r["patient_id"])] = 1

    dcm_files = sorted([p for p in raw_dir.rglob("*.dcm") if not p.name.startswith(".")])
    if len(dcm_files) > max_samples:
        step = max(1, len(dcm_files) // max_samples)
        dcm_files = dcm_files[::step][:max_samples]

    prompt = "classify low grade versus high grade renal cell carcinoma in CT"
    rows, items = [], []
    for idx, p in enumerate(dcm_files):
        try:
            arr, pid_raw = _load_dicom(p)
            if arr.ndim != 2 or min(arr.shape) < 32 or pid_raw not in grade_map:
                continue
            lbl = grade_map[pid_raw]
            img_r = harmonize_ct_slice(arr, target_hw=(256, 256))
            slab = np.concatenate([img_r, img_r[:2]], axis=0)
        except Exception:
            continue

        sid = f"tcga_kirc_ct_{idx:05d}"
        pid = f"tcga_kirc_ct_{pid_raw}"
        items.append({
            "sample_id": sid,
            "image": slab,
            "mask": None,
            "meta": {
                "sample_id": sid,
                "patient_id": pid,
                "dataset": "tcga_kirc_ct",
                "modality": "CT",
                "is_3d": True,
                "label": lbl,
                "multilabel": [-1.0] * 14,
                "has_mask": False,
                "body_coord": [0.5, 0.5, 0.5],
                "spacing_mm": [1.0, 1.0, 1.0],
                "prompt": prompt,
            },
        })
        rows.append({
            "sample_id": sid,
            "patient_id": pid,
            "dataset": "tcga_kirc_ct",
            "modality": "CT",
            "shard_subdir": "tcga_kirc_ct",
            "label": lbl,
            "has_mask": False,
            "prompt": prompt,
        })
    return pd.DataFrame(rows), items


def main() -> None:
    v3_shards = Path("/Volumes/Shoaib/PanOrgan_V3/preprocessed_shards")
    v1_raw = Path("/Volumes/Shoaib/PanOrgan_V2/V1_archive/dataset/raw")
    labels_dir = Path("/Volumes/Shoaib/PanOrgan_V3/official_labels")

    master_df = pd.read_parquet(v3_shards / "master_manifest.parquet")
    # Remove old unlabeled tcga_kirc_ct & any old cbis_ddsm rows
    master_df = master_df[~master_df["dataset"].isin(["cbis_ddsm", "tcga_kirc_ct"])].reset_index(drop=True)

    new_dfs = []
    for ds_name, fn in [
        ("cbis_ddsm", lambda: build_cbis_labeled(v1_raw / "CBIS_DDSM", labels_dir)),
        ("tcga_kirc_ct", lambda: build_kirc_labeled(v1_raw / "TCGA_KIRC", labels_dir, max_samples=1200)),
    ]:
        out_sub = v3_shards / ds_name
        if out_sub.exists():
            for f in out_sub.glob("*.tar"):
                f.unlink()
        df_c, items = fn()
        df_c = build_patient_splits(df_c, val_ratio=0.15, test_ratio=0.15, seed=42)
        shard_rows = write_dataset_shards(items, out_sub, shard_prefix=ds_name)
        df_c = df_c.merge(pd.DataFrame(shard_rows)[["sample_id", "shard_file"]], on="sample_id", how="left")
        df_c.to_parquet(out_sub / "cohort_manifest.parquet", index=False)
        new_dfs.append(df_c)
        print(f"Sharded {ds_name}: {len(df_c)} labeled samples | splits: {df_c['split'].value_counts().to_dict()}", flush=True)

    out_df = pd.concat([master_df] + new_dfs, ignore_index=True, sort=False)
    out_df["label"] = out_df["label"].fillna(-1).astype(int)
    out_df["has_mask"] = out_df["has_mask"].fillna(False).astype(bool)

    audit = verify_zero_patient_overlap(out_df)
    out_df.to_parquet(v3_shards / "master_manifest.parquet", index=False)
    (v3_shards / "split_audit_report.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Updated V3 Master Manifest -> Total: {len(out_df)} | Zero-Leakage Verified!")


if __name__ == "__main__":
    main()
