#!/usr/bin/env python3
"""PanOrgan 4 Interactive Clinical Inference Web Server (FastAPI)."""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

_V3_SSD = Path("/Volumes/Shoaib/PanOrgan_V3")
_SSD_ROOT = Path(os.environ.get("PANORGAN_DRIVE_ROOT", "/Volumes/Shoaib/PanOrgan_4"))
os.environ.setdefault("HF_HOME", str(_V3_SSD / "model_cache" / "hf"))
os.environ.setdefault("TORCH_HOME", str(_V3_SSD / "model_cache" / "torch"))
os.environ.setdefault("TIMM_CACHE_DIR", str(_V3_SSD / "model_cache" / "timm"))

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "V2"))
sys.path.insert(0, str(_REPO_ROOT / "V4" / "scripts"))

from panorgan.config import load_config
from panorgan.models import PanOrganNetV2
from PanOrgan_4_fix_heads import COHORT_HEAD_MAP, MultiScaleCohortHead, pool_multiscale
from PanOrgan_4_optimize import PanOrgan4MultiHeadOpt
from PanOrgan_4_train import TASK_GROUP_MAP

WEB_DIR = Path(__file__).resolve().parent
DEMO_DIR = WEB_DIR / "static" / "demos"

CLASS_LABELS: Dict[str, list[str]] = {
    "cxr_4class": ["Normal", "Bacterial Pneumonia", "Viral / COVID-19", "Lung Opacity / Other"],
    "busi_3class": ["Normal", "Benign Lesion", "Malignant Lesion"],
    "cmmd_mg": ["Benign / Normal Breast", "Malignant Breast Lesion"],
    "cbis_ddsm": ["Benign / Normal Breast", "Malignant Breast Lesion"],
    "kidney_ct_abn": ["Normal Kidney", "Abnormal Renal Lesion"],
    "tcga_kirc_ct": ["Low-Grade RCC", "High-Grade RCC"],
    "tcga_lgg": ["Low-Grade Glioma", "High-Grade Glioma"],
    "prostatex_mri": ["Clinically Insignificant", "Clinically Significant Cancer"],
    "ultrasound_nerve": ["Nerve Absent", "Brachial Plexus Nerve Present"],
    "mg_binary": ["Benign / Normal", "Malignant"],
    "ct_binary": ["Normal / Low-Grade", "Abnormal / High-Grade"],
    "mri_us_binary": ["Negative / Low-Grade", "Positive / High-Grade"],
}

MULTILABEL_FINDINGS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural Thickening",
    "Pneumonia", "Pneumothorax",
]

COHORT_TO_DEFAULT_PROMPT: Dict[str, str] = {
    "pulmonology_cxr": "segment lung and classify thoracic disease in chest radiograph",
    "busi": "classify normal benign or malignant breast lesion in ultrasound",
    "cmmd_mg": "classify benign versus malignant breast lesion in mammogram",
    "kidney_ct_abn": "classify normal versus abnormal renal lesion in CT",
    "tcga_lgg": "classify low grade versus high grade glioma in brain MRI",
    "ultrasound_nerve": "segment brachial plexus nerve and classify presence in ultrasound",
}

app = FastAPI(title="PanOrgan 4 Clinical Inference Studio")
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

_STATE: Dict[str, Any] = {}


def get_model_bundle() -> Dict[str, Any]:
    if "model" in _STATE:
        return _STATE
    cfg = load_config(_REPO_ROOT / "V3" / "configs" / "v3_panorgan.yaml")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mod_map = {k: int(v["id"]) for k, v in cfg["modalities"].items()}

    base_model = PanOrganNetV2.from_config(cfg)
    model = PanOrgan4MultiHeadOpt(base_model, embed_dim=int(cfg["model"].get("embed_dim", 384)), dropout=0.30).to(device)
    ckpt_dir = _SSD_ROOT / "checkpoints" / "panorgan_4"
    st = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(st["model_state_dict"], strict=False)
    model.eval()

    ms_heads: Dict[str, MultiScaleCohortHead] = {}
    temps: Dict[str, float] = {}
    thrs: Dict[str, float] = {}
    ms_path = ckpt_dir / "multiscale_heads.pt"
    if ms_path.exists():
        ms_ck = torch.load(ms_path, map_location="cpu", weights_only=False)
        temps = ms_ck.get("temperatures", {})
        thrs = ms_ck.get("thresholds", {})
        for hname, sd in ms_ck.get("multiscale_heads", {}).items():
            n_cls = sd["net.9.weight"].shape[0]
            h = MultiScaleCohortHead(in_dim=2688, hidden_dim=512, num_classes=n_cls, dropout=0.35)
            h.load_state_dict(sd)
            h.eval()
            ms_heads[hname] = h.to(device)

    _STATE.update({
        "model": model,
        "ms_heads": ms_heads,
        "temps": temps,
        "thrs": thrs,
        "device": device,
        "mod_map": mod_map,
        "cfg": cfg,
    })
    return _STATE


def _encode_png(arr_rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr_rgb.astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((WEB_DIR / "static" / "index.html").read_text(encoding="utf-8"))


@app.get("/api/demos")
def list_demos() -> JSONResponse:
    demos_path = DEMO_DIR / "demos.json"
    if not demos_path.exists():
        return JSONResponse([])
    return JSONResponse(json.loads(demos_path.read_text(encoding="utf-8")))


@app.get("/api/report")
def get_report() -> JSONResponse:
    rep_path = _SSD_ROOT / "experiments" / "thesis_reports" / "panorgan_4_complete_thesis_report.json"
    if not rep_path.exists():
        return JSONResponse({"status": "pending"})
    return JSONResponse(json.loads(rep_path.read_text(encoding="utf-8")))


@app.post("/api/predict")
async def predict(
    demo_id: Optional[str] = Form(None),
    cohort: str = Form("pulmonology_cxr"),
    modality: str = Form("CXR"),
    prompt: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
) -> JSONResponse:
    bundle = get_model_bundle()
    model: PanOrgan4MultiHeadOpt = bundle["model"]
    ms_heads: Dict[str, MultiScaleCohortHead] = bundle["ms_heads"]
    temps: Dict[str, float] = bundle["temps"]
    thrs: Dict[str, float] = bundle["thrs"]
    device: torch.device = bundle["device"]
    mod_map: Dict[str, int] = bundle["mod_map"]

    true_label_idx: Optional[int] = None
    if demo_id:
        npy_path = DEMO_DIR / f"{demo_id}.npy"
        if not npy_path.exists():
            return JSONResponse({"error": f"Demo '{demo_id}' not found"}, status_code=404)
        arr = np.load(npy_path).astype(np.float32)
        cohort = demo_id
        demos = json.loads((DEMO_DIR / "demos.json").read_text())
        meta = next((d for d in demos if d["id"] == demo_id), {})
        modality = meta.get("modality", modality)
        true_label_idx = meta.get("true_label")
        prompt_str = prompt or meta.get("prompt") or COHORT_TO_DEFAULT_PROMPT.get(cohort, "medical scan analysis")
    elif file is not None:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("L").resize((256, 256), Image.Resampling.BILINEAR)
        g = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.stack([g] * 5, axis=0)
        prompt_str = prompt or COHORT_TO_DEFAULT_PROMPT.get(cohort, "medical scan analysis")
    else:
        return JSONResponse({"error": "Provide either demo_id or file"}, status_code=400)

    mod_id = mod_map.get(modality, 0)
    meta_vec = torch.zeros((1, 10), dtype=torch.float32, device=device)
    if mod_id < 5:
        meta_vec[0, mod_id] = 1.0
    is_3d = torch.tensor([modality in ("CT", "MRI")], dtype=torch.bool, device=device)
    x = torch.from_numpy(arr).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out1 = model.base(x, meta_vec, is_3d, prompts=[prompt_str], return_features=True)
        hname, _ = COHORT_HEAD_MAP.get(cohort, ("cxr_4class", 4))
        if hname in ms_heads:
            out2 = model.base(torch.flip(x, dims=[-1]), meta_vec, is_3d, prompts=[prompt_str], return_features=True)
            ms_feat = 0.5 * (pool_multiscale(out1) + pool_multiscale(out2))
            logits = ms_heads[hname](ms_feat)[0].cpu()
            temp = float(temps.get(hname, 1.0))
            thr = float(thrs.get(hname, 0.50))
            head_used = f"{hname} (Multi-Scale + TTA)"
        else:
            tg = TASK_GROUP_MAP.get(cohort, "cxr_4class")
            logits = model.task_heads[tg](out1["embedding"])[0].cpu()
            temp = 1.0
            thr = 0.50
            head_used = tg
            hname = tg
    if device.type == "mps":
        torch.mps.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    probs = F.softmax(logits / max(temp, 1e-3), dim=-1).numpy()
    pred_idx = int((probs[1] >= thr) if len(probs) == 2 else probs.argmax())
    labels = CLASS_LABELS.get(hname, [f"Class {i}" for i in range(len(probs))])

    ml_probs = torch.sigmoid(out1["multilabel_logits"][0]).cpu().numpy()
    findings = sorted(
        [{"name": MULTILABEL_FINDINGS[i], "prob": round(float(p), 4)} for i, p in enumerate(ml_probs) if i < len(MULTILABEL_FINDINGS)],
        key=lambda r: r["prob"],
        reverse=True,
    )[:6]

    coords = out1["pred_coords"][0].cpu().numpy().tolist() if out1.get("pred_coords") is not None else []

    seg_mask = (out1["seg_logits"][0].argmax(dim=0) > 0).cpu().numpy()
    mid = arr[2]
    base_u8 = np.clip((mid - mid.min()) / (mid.max() - mid.min() + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    orig_rgb = np.stack([base_u8, base_u8, base_u8], axis=-1)
    overlay_rgb = orig_rgb.copy()
    overlay_rgb[seg_mask, 0] = (0.35 * overlay_rgb[seg_mask, 0]).astype(np.uint8)
    overlay_rgb[seg_mask, 1] = np.clip(0.55 * overlay_rgb[seg_mask, 1] + 115, 0, 255).astype(np.uint8)
    overlay_rgb[seg_mask, 2] = np.clip(0.55 * overlay_rgb[seg_mask, 2] + 115, 0, 255).astype(np.uint8)

    return JSONResponse({
        "head_used": head_used,
        "cohort": cohort,
        "modality": modality,
        "prompt": prompt_str,
        "device": str(device),
        "latency_ms": round(latency_ms, 1),
        "temperature": round(temp, 3),
        "threshold": round(thr, 3),
        "predicted_index": pred_idx,
        "predicted_label": labels[pred_idx] if pred_idx < len(labels) else f"Class {pred_idx}",
        "true_label": labels[true_label_idx] if (true_label_idx is not None and true_label_idx < len(labels)) else None,
        "disease_detected": pred_idx > 0,
        "probabilities": [
            {"label": labels[i] if i < len(labels) else f"Class {i}", "prob": round(float(p), 4)}
            for i, p in enumerate(probs)
        ],
        "multilabel_findings": findings,
        "sacr_coords": [round(float(c), 3) for c in coords],
        "seg_area_pct": round(float(seg_mask.mean()) * 100.0, 2),
        "original_png": _encode_png(orig_rgb),
        "overlay_png": _encode_png(overlay_rgb),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8010)
