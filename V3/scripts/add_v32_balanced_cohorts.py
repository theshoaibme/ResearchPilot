#!/usr/bin/env python3
"""Shard CMMD (MG), Kidney CT Abnormality (CT), and unused CXR Class 2/3 cohorts for PanOrgan 3.2."""
from __future__ import annotations

import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image

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


def build_cmmd_mg(cmmd_dir: Path, max_mal: int = 1500) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    df = pd.read_parquet(cmmd_dir / "data.parquet")
    ben = df[df["Classification"] == "Benign"]
    mal = df[df["Classification"] == "Malignant"].sample(n=min(max_mal, (df["Classification"] == "Malignant").sum()), random_state=42)
    sub = pd.concat([ben, mal], ignore_index=True)

    prompt = "classify benign versus malignant breast lesion in mammogram"
    rows, items = [], []
    zip_files = [zipfile.ZipFile(zp, "r") for zp in sorted(cmmd_dir.glob("images-*.zip"))]
    name_to_zf = {}
    for zf in zip_files:
        for nm in zf.namelist():
            name_to_zf[nm] = zf
    try:
        for idx, r in sub.iterrows():
            rel = str(r["Path"])
            zf = name_to_zf.get(rel)
            if zf is None:
                continue
            try:
                with zf.open(rel) as f:
                    arr = np.array(Image.open(f).convert("L"), dtype=np.float32)
                if arr.ndim != 2 or min(arr.shape) < 32:
                    continue
                img_r = harmonize_projection_2d(arr, target_hw=(256, 256))
            except Exception:
                continue

            m = re.search(r"D\d+-\d+", rel)
            pid_raw = m.group(0) if m else f"row_{idx}"
            lbl = 1 if str(r["Classification"]) == "Malignant" else 0
            sid = f"cmmd_mg_{idx:05d}"
            pid = f"cmmd_{pid_raw}"
            items.append({
                "sample_id": sid,
                "image": img_r,
                "mask": None,
                "meta": {
                    "sample_id": sid,
                    "patient_id": pid,
                    "dataset": "cmmd_mg",
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
                "dataset": "cmmd_mg",
                "modality": "MG",
                "shard_subdir": "cmmd_mg",
                "label": lbl,
                "has_mask": False,
                "prompt": prompt,
            })
    finally:
        for zf in zip_files:
            zf.close()
    return pd.DataFrame(rows), items


def build_kidney_ct(kct_dir: Path) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    records = [json.loads(line) for line in (kct_dir / "dataset_m.json").read_text(encoding="utf-8").splitlines() if line.strip()]
    lbl_map = {r["image"]: int(r["abnormality"]) for r in records}
    prompt = "classify normal versus abnormal renal lesion in CT"
    rows, items = [], []

    with zipfile.ZipFile(kct_dir / "kidney_ct" / "kidney_CT.zip", "r") as zf, tempfile.TemporaryDirectory() as tmpd:
        mha_entries = [n for n in zf.namelist() if n.endswith(".mha") and not n.startswith("__MACOSX")]
        for v_idx, entry in enumerate(sorted(mha_entries)):
            fname = Path(entry).name
            if fname not in lbl_map:
                continue
            lbl = lbl_map[fname]
            pid_raw = fname.replace("kidneyabnormality", "").replace("_0000.mha", "")
            pid = f"kct_{pid_raw}"
            tmp_path = Path(tmpd) / fname
            tmp_path.write_bytes(zf.read(entry))
            try:
                vol = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path))).astype(np.float32)
                tmp_path.unlink(missing_ok=True)
                if vol.ndim != 3 or vol.shape[0] < 3 or min(vol.shape[1:]) < 32:
                    continue
            except Exception:
                tmp_path.unlink(missing_ok=True)
                continue

            z_len = vol.shape[0]
            for s_idx, z_frac in enumerate((0.40, 0.50, 0.60)):
                z = min(z_len - 1, max(0, int(z_len * z_frac)))
                img_r = harmonize_ct_slice(vol[z], target_hw=(256, 256))
                slab = np.concatenate([img_r, img_r[:2]], axis=0)
                sid = f"kct_{v_idx:04d}_{s_idx}"
                items.append({
                    "sample_id": sid,
                    "image": slab,
                    "mask": None,
                    "meta": {
                        "sample_id": sid,
                        "patient_id": pid,
                        "dataset": "kidney_ct_abn",
                        "modality": "CT",
                        "is_3d": True,
                        "label": lbl,
                        "multilabel": [-1.0] * 14,
                        "has_mask": False,
                        "body_coord": [0.5, 0.5, float(z_frac)],
                        "spacing_mm": [1.0, 1.0, 1.0],
                        "prompt": prompt,
                    },
                })
                rows.append({
                    "sample_id": sid,
                    "patient_id": pid,
                    "dataset": "kidney_ct_abn",
                    "modality": "CT",
                    "shard_subdir": "kidney_ct_abn",
                    "label": lbl,
                    "has_mask": False,
                    "prompt": prompt,
                })
    return pd.DataFrame(rows), items


def build_cxr_class23_ext(cxr_root: Path) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    prompt = "segment lung fields and classify normal covid lung opacity or viral pneumonia in chest radiograph"
    specs = [
        ("Lung_Opacity", 2, 500, 2500),
        ("Viral Pneumonia", 3, 500, 845),
    ]
    rows, items = [], []
    counter = 0
    for folder, lbl, skip_n, take_n in specs:
        img_dir = cxr_root / folder / "images"
        mask_dir = cxr_root / folder / "masks"
        imgs = sorted([p for p in img_dir.glob("*.png") if not p.name.startswith(".")])[skip_n:skip_n + take_n]
        for p in imgs:
            try:
                arr = np.array(Image.open(p).convert("L"), dtype=np.float32)
                img_r = harmonize_projection_2d(arr, target_hw=(256, 256))
                m_path = mask_dir / p.name
                has_mask = m_path.exists()
                mask_arr = None
                if has_mask:
                    m_img = Image.open(m_path).convert("L").resize((256, 256), Image.NEAREST)
                    mask_arr = (np.array(m_img, dtype=np.float32) > 127.0).astype(np.float32)
            except Exception:
                continue

            sid = f"cxr_ext_{counter:05d}"
            pid = f"cxr_ext_{p.stem}"
            counter += 1
            items.append({
                "sample_id": sid,
                "image": img_r,
                "mask": mask_arr,
                "meta": {
                    "sample_id": sid,
                    "patient_id": pid,
                    "dataset": "pulmonology_cxr_ext",
                    "modality": "CXR",
                    "is_3d": False,
                    "label": lbl,
                    "multilabel": [-1.0] * 14,
                    "has_mask": bool(has_mask),
                    "body_coord": [0.5, 0.5, 0.5],
                    "spacing_mm": [1.0, 1.0, 1.0],
                    "prompt": prompt,
                },
            })
            rows.append({
                "sample_id": sid,
                "patient_id": pid,
                "dataset": "pulmonology_cxr_ext",
                "modality": "CXR",
                "shard_subdir": "pulmonology_cxr_ext",
                "label": lbl,
                "has_mask": bool(has_mask),
                "prompt": prompt,
            })
    return pd.DataFrame(rows), items


def main() -> None:
    v3_shards = Path("/Volumes/Shoaib/PanOrgan_V3/preprocessed_shards")
    raw_v32 = Path("/Volumes/Shoaib/PanOrgan_V3/raw_v32")
    cxr_root = Path("/Volumes/Shoaib/PanOrgan_V2/V1_archive/dataset/raw/pulmonology/chest_xray")

    master_df = pd.read_parquet(v3_shards / "master_manifest.parquet")
    master_df = master_df[~master_df["dataset"].isin(["cmmd_mg", "kidney_ct_abn", "pulmonology_cxr_ext"])].reset_index(drop=True)

    new_dfs = []
    for ds_name, fn in [
        ("cmmd_mg", lambda: build_cmmd_mg(raw_v32 / "cmmd")),
        ("kidney_ct_abn", lambda: build_kidney_ct(raw_v32 / "kidney_ct")),
        ("pulmonology_cxr_ext", lambda: build_cxr_class23_ext(cxr_root)),
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
        print(f"Sharded {ds_name}: {len(df_c)} samples | pts={df_c['patient_id'].nunique()} | splits={df_c['split'].value_counts().to_dict()}", flush=True)

    out_df = pd.concat([master_df] + new_dfs, ignore_index=True, sort=False)
    out_df["label"] = out_df["label"].fillna(-1).astype(int)
    out_df["has_mask"] = out_df["has_mask"].fillna(False).astype(bool)

    audit = verify_zero_patient_overlap(out_df)
    out_df.to_parquet(v3_shards / "master_manifest.parquet", index=False)
    (v3_shards / "split_audit_report.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Updated Master Manifest -> Total: {len(out_df)} | Zero-Leakage Verified!", flush=True)
    print("Label counts:\n", out_df.groupby(["label", "split"]).size().unstack(fill_value=0), flush=True)


if __name__ == "__main__":
    main()
