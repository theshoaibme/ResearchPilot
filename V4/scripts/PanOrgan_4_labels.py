#!/usr/bin/env python3
"""Create V4 master manifest with official TCIA PROSTATEx ClinSig labels attached (zero leakage)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "V2"))
from panorgan.data import verify_zero_patient_overlap


def main() -> None:
    shards_dir = Path("/Volumes/Shoaib/PanOrgan_V3/preprocessed_shards")
    px_dir = Path("/Volumes/Shoaib/PanOrgan_V3/official_labels/prostatex")

    tr = pd.read_csv(px_dir / "ProstateX-TrainingLesionInformationv2" / "ProstateX-Findings-Train.csv")
    te = pd.read_csv(px_dir / "ProstateX-Findings-Test.csv")
    findings = pd.concat([tr, te], ignore_index=True)

    # Patient-level clinically significant prostate cancer (1 if any lesion ClinSig == True, else 0)
    px_map = findings.groupby("ProxID")["ClinSig"].apply(lambda s: int(any(bool(x) for x in s))).to_dict()

    df = pd.read_parquet(shards_dir / "master_manifest.parquet")
    mask = df["dataset"] == "prostatex_mri"
    matched = 0
    for idx in df[mask].index:
        raw_pid = str(df.at[idx, "patient_id"]).replace("prostatex_mri_", "")
        if raw_pid in px_map:
            df.at[idx, "label"] = px_map[raw_pid]
            df.at[idx, "prompt"] = "classify clinically significant prostate cancer in multiparametric MRI"
            matched += 1

    audit = verify_zero_patient_overlap(df)
    out_manifest = shards_dir / "master_manifest_v4.parquet"
    df.to_parquet(out_manifest, index=False)
    (shards_dir / "split_audit_v4.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(f"V4 Master Manifest -> {out_manifest}")
    print(f"Total samples: {len(df)} | Unique patients: {df['patient_id'].nunique()}")
    print(f"PROSTATEx labeled samples: {matched} / {int(mask.sum())}")
    print(f"Unlabeled remaining: {int((df['label'] < 0).sum())}")


if __name__ == "__main__":
    main()
