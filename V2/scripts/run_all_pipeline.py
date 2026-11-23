#!/usr/bin/env python3
"""
Master End-to-End Orchestrator for Pan-Organ Net V2 (Stages 2-5).
Executes SSL Pretraining, Multi-Task Training, Baselines/Ablations, and Q1 Evaluation.
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path
import subprocess
import sys
import time

LOG_DIR = Path("/content")
DRIVE_LOG_DIR = Path("/content/drive/MyDrive/ResearchPilot_PanOrgan/V2/experiments/pipeline_logs")
DRIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)

STAGES = [
    ("Stage-1 SSL Pretraining (MA-UGM + SACR)", "/content/run_notebook_02.py", "notebook_02.log"),
    ("Stage-2 Multi-Task Training", "/content/run_notebook_03.py", "notebook_03.log"),
    ("Baselines & Controlled Ablations", "/content/run_notebook_04.py", "notebook_04.log"),
    ("Q1 Evaluation, Calibration & Paper Tables", "/content/run_notebook_05.py", "notebook_05.log"),
]

def log(msg: str) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] [PIPELINE] {msg}"
    print(formatted)
    sys.stdout.flush()
    with open("/content/pipeline_master.log", "a", encoding="utf-8") as f:
        f.write(formatted + "\n")
    with open(DRIVE_LOG_DIR / "pipeline_master.log", "a", encoding="utf-8") as f:
        f.write(formatted + "\n")

log("=== Starting Pan-Organ Net V2 Full Automated Pipeline Execution ===")

for stage_name, script_path, log_file in STAGES:
    log(f"--- Launching {stage_name} ---")
    log(f"Script: {script_path} | Log: {log_file}")
    
    local_log = LOG_DIR / log_file
    drive_log = DRIVE_LOG_DIR / log_file
    
    t0 = time.time()
    with open(local_log, "w", encoding="utf-8") as out_f:
        proc = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            out_f.write(line)
            out_f.flush()
            if "Starting" in line or "Completed" in line or "Epoch" in line or "Loss" in line or "Best" in line:
                log(f"[{log_file}] {line.strip()}")
        proc.wait()
    
    elapsed = time.time() - t0
    # Copy log to Drive
    if local_log.exists():
        drive_log.write_text(local_log.read_text(encoding="utf-8"), encoding="utf-8")
    
    if proc.returncode == 0:
        log(f"SUCCESS: {stage_name} completed in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    else:
        log(f"ERROR: {stage_name} exited with returncode {proc.returncode} after {elapsed:.1f}s. Review {local_log}")
        break

log("=== Pan-Organ Net V2 Full Automated Pipeline Execution Finished ===")
