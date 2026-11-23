#!/usr/bin/env python3
"""Build PanOrganNet-V3 7-cohort multi-modal WebDataset shards & zero-leakage manifest on SSD."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
V2_PKG = REPO_ROOT / "V2"
if str(V2_PKG) not in sys.path:
    sys.path.insert(0, str(V2_PKG))

from panorgan.config import load_config
from panorgan.data import (
    build_patient_splits,
    harmonize_ct_slice,
    harmonize_mri_slice,
    harmonize_projection_2d,
    harmonize_ultrasound_2d,
    resize_with_aspect_pad,
    verify_zero_patient_overlap,
    write_dataset_shards,
)
from panorgan.data.datasets_registry import parse_ultrasound_nerve_dataset


def _load_rgb_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float32)


def _load_mask_binary(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"))
    return (arr > 127).astype(np.uint8)


def _load_dicom_array(path: Path) -> tuple[np.ndarray, Dict[str, Any]]:
    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept
    patient_id = str(getattr(ds, "PatientID", path.parent.name) or path.parent.name)
    px_sp = getattr(ds, "PixelSpacing", [1.0, 1.0])
    spacing = (
        float(getattr(ds, "SliceThickness", 1.0) or 1.0),
        float(px_sp[0]) if len(px_sp) > 0 else 1.0,
        float(px_sp[1]) if len(px_sp) > 1 else 1.0,
    )
    return arr, {"patient_id": patient_id, "spacing_mm": spacing}


def build_pulmonology_samples(raw_dir: Path, max_per_class: int, prompt: str) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    classes = {"Normal": 0, "COVID": 1, "Lung_Opacity": 2, "Viral Pneumonia": 3}
    rows, items = [], []
    rng = np.random.default_rng(42)

    for cname, label in classes.items():
        img_dir = raw_dir / "chest_xray" / cname / "images"
        msk_dir = raw_dir / "chest_xray" / cname / "masks"
        if not img_dir.exists():
            continue
        files = sorted([p for p in img_dir.iterdir() if p.is_file() and not p.name.startswith(".")])
        if len(files) > max_per_class:
            idx = rng.choice(len(files), size=max_per_class, replace=False)
            files = [files[i] for i in sorted(idx)]

        for p in files:
            sid = f"pulm_{cname.replace(' ', '_')}_{p.stem}"
            pid = f"pulm_{p.stem}"
            msk_p = msk_dir / p.name
            has_mask = msk_p.exists()
            try:
                arr = _load_rgb_gray(p)
                img_r = harmonize_projection_2d(arr, target_hw=(256, 256))
                msk_r = resize_with_aspect_pad(_load_mask_binary(msk_p), target_hw=(256, 256), is_mask=True, pad_value=0) if has_mask else None
            except Exception:
                continue

            items.append({
                "sample_id": sid,
                "image": img_r,
                "mask": msk_r,
                "meta": {
                    "sample_id": sid,
                    "patient_id": pid,
                    "dataset": "pulmonology_cxr",
                    "modality": "CXR",
                    "is_3d": False,
                    "label": label,
                    "multilabel": [-1.0] * 14,
                    "has_mask": bool(msk_r is not None),
                    "body_coord": [0.5, 0.5, 0.5],
                    "spacing_mm": [1.0, 1.0, 1.0],
                    "prompt": prompt,
                },
            })
            rows.append({
                "sample_id": sid,
                "patient_id": pid,
                "dataset": "pulmonology_cxr",
                "modality": "CXR",
                "shard_subdir": "pulmonology_cxr",
                "label": label,
                "has_mask": bool(msk_r is not None),
                "prompt": prompt,
            })
    return pd.DataFrame(rows), items


def build_us_nerve_samples(raw_dir: Path, max_samples: int, prompt: str) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    records = parse_ultrasound_nerve_dataset(raw_dir)
    if len(records) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(records), size=max_samples, replace=False)
        records = [records[i] for i in sorted(idx)]

    rows, items = [], []
    for rec in records:
        try:
            arr = _load_rgb_gray(Path(rec.image_path))
            img_r = harmonize_ultrasound_2d(arr, target_hw=(256, 256))
            msk_r = resize_with_aspect_pad(_load_mask_binary(Path(rec.mask_path)), target_hw=(256, 256), is_mask=True, pad_value=0) if rec.mask_path else None
        except Exception:
            continue

        items.append({
            "sample_id": rec.sample_id,
            "image": img_r,
            "mask": msk_r,
            "meta": {
                "sample_id": rec.sample_id,
                "patient_id": rec.patient_id,
                "dataset": "ultrasound_nerve",
                "modality": "US",
                "is_3d": False,
                "label": int(rec.label),
                "multilabel": [-1.0] * 14,
                "has_mask": bool(msk_r is not None),
                "body_coord": [0.5, 0.5, 0.5],
                "spacing_mm": [1.0, 1.0, 1.0],
                "prompt": prompt,
            },
        })
        rows.append({
            "sample_id": rec.sample_id,
            "patient_id": rec.patient_id,
            "dataset": "ultrasound_nerve",
            "modality": "US",
            "shard_subdir": "ultrasound_nerve",
            "label": int(rec.label),
            "has_mask": bool(msk_r is not None),
            "prompt": prompt,
        })
    return pd.DataFrame(rows), items


def build_dicom_ssl_samples(
    raw_dir: Path,
    ds_name: str,
    modality: str,
    max_samples: int,
    prompt: str,
) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    dcm_files = sorted([p for p in raw_dir.rglob("*.dcm") if not p.name.startswith(".")])
    if len(dcm_files) > max_samples:
        step = max(1, len(dcm_files) // max_samples)
        dcm_files = dcm_files[::step][:max_samples]

    rows, items = [], []
    for idx, p in enumerate(dcm_files):
        try:
            arr, dmeta = _load_dicom_array(p)
            if arr.ndim != 2 or min(arr.shape) < 32:
                continue
            img_r = harmonize_ct_slice(arr, target_hw=(256, 256)) if modality == "CT" else harmonize_mri_slice(arr, target_hw=(256, 256))
            slab = np.concatenate([img_r, img_r[:2]], axis=0)
        except Exception:
            continue

        sid = f"{ds_name}_{idx:05d}"
        pid = f"{ds_name}_{dmeta['patient_id']}"
        items.append({
            "sample_id": sid,
            "image": slab,
            "mask": None,
            "meta": {
                "sample_id": sid,
                "patient_id": pid,
                "dataset": ds_name,
                "modality": modality,
                "is_3d": True,
                "label": -1,
                "multilabel": [-1.0] * 14,
                "has_mask": False,
                "body_coord": [0.5, 0.5, 0.5],
                "spacing_mm": list(dmeta["spacing_mm"]),
                "prompt": prompt,
            },
        })
        rows.append({
            "sample_id": sid,
            "patient_id": pid,
            "dataset": ds_name,
            "modality": modality,
            "shard_subdir": ds_name,
            "label": -1,
            "has_mask": False,
            "prompt": prompt,
        })
    return pd.DataFrame(rows), items


def main() -> None:
    cfg = load_config(REPO_ROOT / "V3/configs/v3_panorgan.yaml")
    v3_shards = Path(cfg["storage"]["drive_root"]) / "preprocessed_shards"
    v2_shards = Path(cfg["storage"]["v2_shards_root"])
    v1_raw = Path(cfg["storage"]["v1_raw_root"])
    v3_shards.mkdir(parents=True, exist_ok=True)

    # 1. Reuse existing V2 cohorts via symlinks (zero duplicate disk space)
    v2_manifest = pd.read_parquet(v2_shards / "master_manifest.parquet")
    for sub in ("tcga_lgg", "busi", "nih_cxr14"):
        dst = v3_shards / sub
        if not dst.exists() and (v2_shards / sub).exists():
            os.symlink(v2_shards / sub, dst)

    new_dfs = []
    cohorts = [
        ("pulmonology_cxr", lambda: build_pulmonology_samples(
            v1_raw / "pulmonology",
            int(cfg["datasets"]["pulmonology_cxr"]["max_per_class"]),
            cfg["datasets"]["pulmonology_cxr"]["prompt_template"],
        )),
        ("ultrasound_nerve", lambda: build_us_nerve_samples(
            v1_raw / "Ultrasound_Nerve",
            int(cfg["datasets"]["ultrasound_nerve"]["max_samples"]),
            cfg["datasets"]["ultrasound_nerve"]["prompt_template"],
        )),
        ("prostatex_mri", lambda: build_dicom_ssl_samples(
            v1_raw / "PROSTATEx",
            "prostatex_mri",
            "MRI",
            int(cfg["datasets"]["prostatex_mri"]["max_samples"]),
            cfg["datasets"]["prostatex_mri"]["prompt_template"],
        )),
        ("tcga_kirc_ct", lambda: build_dicom_ssl_samples(
            v1_raw / "TCGA_KIRC",
            "tcga_kirc_ct",
            "CT",
            int(cfg["datasets"]["tcga_kirc_ct"]["max_samples"]),
            cfg["datasets"]["tcga_kirc_ct"]["prompt_template"],
        )),
    ]

    for ds_name, builder in cohorts:
        out_sub = v3_shards / ds_name
        manifest_cache = out_sub / "cohort_manifest.parquet"
        if manifest_cache.exists():
            print(f"[SKIP] {ds_name} already sharded.", flush=True)
            new_dfs.append(pd.read_parquet(manifest_cache))
            continue

        print(f"Building & sharding {ds_name}...", flush=True)
        df_c, items = builder()
        if df_c.empty:
            print(f"[WARN] {ds_name} produced 0 samples.", flush=True)
            continue
        df_c = build_patient_splits(df_c, val_ratio=0.15, test_ratio=0.15, seed=42)
        shard_rows = write_dataset_shards(items, out_sub, shard_prefix=ds_name)
        shard_df = pd.DataFrame(shard_rows)
        df_c = df_c.merge(shard_df[["sample_id", "shard_file"]], on="sample_id", how="left")
        df_c.to_parquet(manifest_cache, index=False)
        new_dfs.append(df_c)
        print(f"  -> {ds_name}: {len(df_c)} samples sharded cleanly.", flush=True)

    # Align columns with V2 manifest and combine
    all_dfs = [v2_manifest] + new_dfs
    master_df = pd.concat(all_dfs, ignore_index=True, sort=False)
    master_df["label"] = master_df["label"].fillna(-1).astype(int)
    master_df["has_mask"] = master_df["has_mask"].fillna(False).astype(bool)

    audit = verify_zero_patient_overlap(master_df)
    master_df.to_parquet(v3_shards / "master_manifest.parquet", index=False)
    (v3_shards / "split_audit_report.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("\n=== V3 MASTER MANIFEST SUMMARY ===", flush=True)
    print(f"Total samples: {len(master_df)}", flush=True)
    print(master_df.groupby(["dataset", "split"]).size().unstack(fill_value=0))
    print("\nZero-Leakage Audit Verified ->", v3_shards / "split_audit_report.json")


if __name__ == "__main__":
    main()
