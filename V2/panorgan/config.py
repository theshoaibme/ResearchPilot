"""YAML-driven configuration and GPU-tier auto-scaling for Pan-Organ Net V2."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


@dataclass
class RuntimeProfile:
    gpu_name: str
    tier: str
    batch_size: int
    grad_accum: int
    amp_dtype: str
    device: str


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration with optional environment-variable overrides."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        cfg: Dict[str, Any] = yaml.safe_load(handle)

    drive_override = os.environ.get("PANORGAN_DRIVE_ROOT")
    if drive_override:
        cfg["storage"]["drive_root"] = drive_override

    cache_override = os.environ.get("PANORGAN_CACHE_DIR")
    if cache_override:
        cfg["storage"]["local_cache_dir"] = cache_override

    return cfg


def detect_runtime_profile(cfg: Dict[str, Any]) -> RuntimeProfile:
    """Detect GPU hardware (A100/L4/T4/MPS/CPU) and select optimal batch/precision profile."""
    profiles = cfg["training"]["gpu_profiles"]
    if not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and "MPS" in profiles:
            p = profiles["MPS"]
            return RuntimeProfile(
                gpu_name="Apple Silicon MPS",
                tier="MPS",
                batch_size=int(p["batch_size"]),
                grad_accum=int(p["grad_accum"]),
                amp_dtype=str(p["amp_dtype"]),
                device="mps",
            )
        p = profiles["CPU"]
        return RuntimeProfile(
            gpu_name="CPU",
            tier="CPU",
            batch_size=int(p["batch_size"]),
            grad_accum=int(p["grad_accum"]),
            amp_dtype=str(p["amp_dtype"]),
            device="cpu",
        )

    gpu_name = torch.cuda.get_device_name(0)
    upper = gpu_name.upper()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    if "A100" in upper or "H100" in upper or vram_gb >= 35.0:
        tier = "A100"
    elif "L4" in upper or "V100" in upper or vram_gb >= 20.0:
        tier = "L4"
    else:
        tier = "T4"

    p = profiles[tier]
    return RuntimeProfile(
        gpu_name=gpu_name,
        tier=tier,
        batch_size=int(p["batch_size"]),
        grad_accum=int(p["grad_accum"]),
        amp_dtype=str(p["amp_dtype"]),
        device="cuda",
    )


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Enforce deterministic reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
