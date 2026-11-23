"""Tests for modality physics harmonization, zero-leakage gate, sharding, and LRU SSD cache."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panorgan.data import (
    BalancedModalityBatchSampler,
    PanOrganShardDataset,
    PatientLeakageError,
    ShardCacheManager,
    build_25d_slab,
    build_meta_vector,
    build_patient_splits,
    compute_modality_saliency,
    harmonize_ct_slice,
    harmonize_mri_slice,
    harmonize_projection_2d,
    harmonize_ultrasound_2d,
    verify_zero_patient_overlap,
    write_dataset_shards,
)


def test_modality_harmonizers_and_saliency_edge_cases():
    # CT multi-window HU clipping
    ct_raw = np.linspace(-1200, 1200, 64 * 80, dtype=np.float32).reshape(64, 80)
    ct_out = harmonize_ct_slice(ct_raw, target_hw=(64, 64))
    assert ct_out.shape == (3, 64, 64)
    assert float(ct_out.min()) >= 0.0 and float(ct_out.max()) <= 1.0

    # MRI constant/zero-variance safety
    mri_zero = np.zeros((48, 48), dtype=np.float32)
    mri_out = harmonize_mri_slice(mri_zero, target_hw=(64, 64))
    assert mri_out.shape == (3, 64, 64)
    assert np.isfinite(mri_out).all()

    # 16-bit safe projection
    xr_16 = np.random.randint(0, 65535, size=(90, 70), dtype=np.uint16)
    xr_out = harmonize_projection_2d(xr_16, target_hw=(64, 64))
    assert xr_out.shape == (3, 64, 64)
    assert float(xr_out.min()) >= 0.0 and float(xr_out.max()) <= 1.0

    # Ultrasound despeckled saliency
    rng = np.random.default_rng(0)
    us = rng.uniform(0, 255, size=(64, 64)).astype(np.float32)
    us_out = harmonize_ultrasound_2d(us, target_hw=(64, 64))
    sal = compute_modality_saliency(us_out, modality="US")
    assert sal.shape == (64, 64)
    assert float(sal.min()) >= 0.0 and float(sal.max()) <= 1.0

    # 2.5D volumetric slab builder
    vol = rng.normal(0.0, 200.0, size=(7, 48, 48)).astype(np.float32)
    slab = build_25d_slab(vol, center_idx=0, slab_depth=5, modality="CT", target_hw=(64, 64))
    assert slab.shape == (5, 64, 64)

    meta = build_meta_vector(modality_id=0, spacing_mm=(1.5, 1.5, 3.0), is_3d=True, has_contrast=True)
    assert meta.shape == (10,)


def test_patient_leakage_gate_catches_overlap_and_validates_clean_splits():
    # Intentionally leaky dataframe MUST raise PatientLeakageError
    leaky_df = pd.DataFrame(
        {
            "dataset": ["busi"] * 4,
            "patient_id": ["P1", "P2", "P1", "P3"],
            "split": ["train", "train", "test", "val"],
        }
    )
    with pytest.raises(PatientLeakageError):
        verify_zero_patient_overlap(leaky_df)

    # Clean patient-level split generation across 30 patients (multiple slices per patient)
    rows = []
    for p in range(30):
        lbl = p % 3
        for s in range(4):
            rows.append(
                {
                    "sample_id": f"p{p}_s{s}",
                    "dataset": "busi",
                    "patient_id": f"patient_{p:03d}",
                    "label": lbl,
                }
            )
    df = pd.DataFrame(rows)
    split_df = build_patient_splits(df, val_ratio=0.15, test_ratio=0.15, seed=42)
    audit = verify_zero_patient_overlap(split_df)
    assert audit["busi"]["leakage_count"] == 0
    assert set(split_df["split"].unique()) == {"train", "val", "test"}


def test_shard_roundtrip_lru_eviction_and_balanced_sampler(tmp_path: Path):
    drive_shards = tmp_path / "drive_shards"
    local_cache = tmp_path / "local_cache"

    samples = []
    for i in range(6):
        samples.append(
            {
                "sample_id": f"s_{i:03d}",
                "image": np.random.rand(3, 32, 32).astype(np.float32),
                "mask": np.random.randint(0, 2, size=(32, 32), dtype=np.uint8),
                "meta": {
                    "dataset": "busi" if i < 4 else "amos22_ct",
                    "patient_id": f"pt_{i}",
                    "modality": "US" if i < 4 else "CT",
                    "is_3d": i >= 4,
                    "label": i % 2,
                    "prompt": "segment lesion",
                },
            }
        )

    # Force small shard size so multiple shards are created
    manifest_rows = write_dataset_shards(
        samples,
        output_dir=drive_shards / "mixed",
        shard_prefix="train",
        max_shard_bytes=4096,
    )
    assert len(manifest_rows) == 6
    manifest_df = pd.DataFrame(manifest_rows)
    assert manifest_df["shard_file"].nunique() >= 2

    # Very tight cache cap forces LRU eviction cleanly
    cache_mgr = ShardCacheManager(drive_shards, local_cache, max_cache_gb=0.000015)
    ds = PanOrganShardDataset(
        manifest_df,
        cache_manager=cache_mgr,
        modality_map={"CT": 0, "MRI": 1, "CXR": 2, "US": 3, "MG": 4},
        target_hw=(64, 64),
        slab_depth=5,
        augment=True,
        mnar_truncation=0.25,
    )
    item = ds[0]
    assert item["image"].shape == (5, 64, 64)
    assert item["mask"].shape == (64, 64)
    assert item["meta_vec"].shape == (10,)

    # Access remaining items to exercise LRU eviction without raising
    for idx in range(len(ds)):
        _ = ds[idx]
    assert cache_mgr.cached_bytes <= cache_mgr.max_bytes * 2

    sampler = BalancedModalityBatchSampler(
        dataset_names=manifest_df["dataset"].tolist(),
        batch_size=4,
        temperature_alpha=0.5,
        steps_per_epoch=3,
    )
    batches = list(sampler)
    assert len(batches) == 3 and all(len(b) == 4 for b in batches)
