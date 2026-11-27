"""Persist smaller value-set closures (Beam-LNS + SuperWalk) and crack their 48-reg schedules via Warm-Splice Rescue."""

from __future__ import annotations

import csv
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

csv.field_size_limit(1 << 28)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from validator import Puzzle, load_puzzles, validate_program  # noqa: E402
from cpp_engines.run_hyperwalk import to_pfmt_block  # noqa: E402
from santa_engine.run_superwalk import parse_superwalk_stdout, save_best  # noqa: E402


def main() -> None:
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 55
    puzzles = load_puzzles(ROOT / "data" / "puzzles.csv")
    best_csv = ROOT / "submissions" / "santa_best.csv"
    vset_bank = ROOT / "santa_engine" / "discovered_smaller_vsets.txt"

    beam_bin = ROOT / "santa_engine" / "beam_lns_bin"
    sw_bin = ROOT / "santa_engine" / "superwalk_bin"
    ws_bin = ROOT / "santa_engine" / "warm_splice_rescue_bin"
    var_pfmt = ROOT / "cpp_engines" / "variants.pfmt"

    best: dict[str, tuple[int, str, list[list]]] = {}
    for row in csv.DictReader(best_csv.open("r", encoding="utf-8")):
        pid = row["id"]
        vr = validate_program(puzzles[pid], row["program"])
        assert vr.valid
        best[pid] = (vr.length, row["program"], json.loads(row["program"]))

    start_total = sum(v[0] for v in best.values())
    ranked = sorted(puzzles.values(), key=lambda p: best[p.id][0], reverse=True)[:top_n]
    ncores = os.cpu_count() or 10
    lock = threading.Lock()
    improved = 0
    saved = 0
    t0 = time.time()

    print(f"[HarvestRescue] Start = {start_total:,} | Harvesting top {len(ranked)} puzzles on {ncores} cores...", flush=True)

    # 1. Recover scored_0729 (seed 561054) + run Beam-LNS v4 on top_n persisting ALL V-lines!
    def harvest_beam(idx_p: tuple[int, Puzzle]) -> None:
        nonlocal improved, saved
        idx, p = idx_p
        with lock:
            cur_prog = best[p.id][2]
        inp = to_pfmt_block(p, cur_prog)
        with tempfile.TemporaryDirectory() as td:
            s_out = pathlib.Path(td) / "seeds.txt"
            proc = subprocess.run(
                [
                    str(beam_bin),
                    "--width",
                    "120",
                    "--prodk",
                    "6",
                    "--pairs",
                    "56",
                    "--triples",
                    "24",
                    "--seed",
                    str(202677703 + idx * 10007),
                    "--seed-out",
                    str(s_out),
                ],
                input=inp,
                text=True,
                capture_output=True,
                check=True,
            )
            cand = parse_superwalk_stdout(proc.stdout)
            lines = s_out.read_text().splitlines() if s_out.exists() else []
        with lock:
            if cand is not None:
                js = json.dumps(cand, separators=(",", ":"))
                vr = validate_program(p, js)
                if vr.valid and vr.length < best[p.id][0]:
                    diff = best[p.id][0] - vr.length
                    best[p.id] = (vr.length, js, cand)
                    improved += 1
                    saved += diff
                    save_best(best_csv, puzzles, best)
                    print(f"  *** [DIRECT WIN] {p.id}: -{diff} -> {sum(v[0] for v in best.values()):,} ***", flush=True)
            if lines:
                with vset_bank.open("a", encoding="utf-8") as bf:
                    for ln in lines:
                        pts = ln.split()
                        if len(pts) >= 6 and pts[0] == "V":
                            print(f"  [HARVESTED BEAM V-SET] {p.id}: {pts[2]} -> {pts[3]} unique values!", flush=True)
                            bf.write(ln + "\n")

    # Also run SuperWalk minV harvest on scored_0729 (seed 561054) + top 10 peak-48 puzzles in parallel!
    def harvest_sw(seed_pid: tuple[int, str]) -> None:
        nonlocal improved, saved
        s, pid = seed_pid
        p = puzzles[pid]
        with lock:
            cur_prog = best[pid][2]
        inp = to_pfmt_block(p, cur_prog)
        with tempfile.TemporaryDirectory() as td:
            dm = pathlib.Path(td) / "dm.txt"
            cmd = [
                str(sw_bin),
                "--time",
                "20",
                "--seed",
                str(s),
                "--T",
                "0.28",
                "--cap",
                "4",
                "--rb",
                "0.58",
                "--pair",
                "0.26",
                "--dump-minv",
                str(dm),
            ]
            if var_pfmt.exists():
                cmd.extend(["--variants", str(var_pfmt)])
            proc = subprocess.run(cmd, input=inp, text=True, capture_output=True, check=True)
            cand = parse_superwalk_stdout(proc.stdout)
            lines = dm.read_text().splitlines() if dm.exists() else []
        with lock:
            if cand is not None:
                js = json.dumps(cand, separators=(",", ":"))
                vr = validate_program(p, js)
                if vr.valid and vr.length < best[pid][0]:
                    diff = best[pid][0] - vr.length
                    best[pid] = (vr.length, js, cand)
                    improved += 1
                    saved += diff
                    save_best(best_csv, puzzles, best)
                    print(f"  *** [SW WIN] {pid}: -{diff} -> {sum(v[0] for v in best.values()):,} ***", flush=True)
            if lines:
                with vset_bank.open("a", encoding="utf-8") as bf:
                    for ln in lines:
                        pts = ln.split()
                        if len(pts) >= 6 and pts[0] == "V":
                            print(f"  [HARVESTED SW V-SET] {pid}: {pts[2]} -> {pts[3]} unique values!", flush=True)
                            bf.write(ln + "\n")

    sw_targets = [
        (561054, "scored_0729"),
        (777001, "scored_0729"),
        (777002, "scored_0051"),
        (777003, "scored_0218"),
        (777004, "scored_0183"),
        (777005, "scored_0744"),
        (777006, "scored_0269"),
        (777007, "scored_0069"),
        (777008, "scored_0982"),
        (777009, "scored_0258"),
    ]
    with ThreadPoolExecutor(ncores) as ex:
        list(ex.map(harvest_sw, sw_targets))
        list(ex.map(harvest_beam, enumerate(ranked)))

    # 2. Deduplicate harvested V-sets with len < current best and run 10-thread WarmSpliceRescue!
    cand_pids: set[str] = {"scored_0774"}
    if vset_bank.exists():
        for ln in vset_bank.read_text().splitlines():
            pts = ln.split()
            if len(pts) >= 6 and pts[0] == "V" and pts[1] in best:
                if int(pts[3]) < best[pts[1]][0]:
                    cand_pids.add(pts[1])

    print(f"\n[WarmSplice] Running 10-thread 250M-iter Warm-Spliced Schedule Rescue on: {sorted(cand_pids)}", flush=True)
    for pid in sorted(cand_pids):
        p = puzzles[pid]
        inp = to_pfmt_block(p, best[pid][2])
        cmd = [str(ws_bin), "18.0", str(ncores)]
        if vset_bank.exists():
            cmd.append(str(vset_bank))
        proc = subprocess.run(cmd, input=inp, text=True, capture_output=True, check=True)
        for eln in proc.stderr.splitlines():
            print("  " + eln, flush=True)
        cand = parse_superwalk_stdout(proc.stdout)
        if cand is not None:
            js = json.dumps(cand, separators=(",", ":"))
            vr = validate_program(p, js)
            if vr.valid and vr.length < best[pid][0]:
                diff = best[pid][0] - vr.length
                old_l = best[pid][0]
                best[pid] = (vr.length, js, cand)
                improved += 1
                saved += diff
                save_best(best_csv, puzzles, best)
                print(
                    f"  *** [WARM-SPLICE BREAKTHROUGH] {pid}: {old_l} -> {vr.length} (-{diff}) "
                    f"| NEW SCORE = {sum(v[0] for v in best.values()):,} ***",
                    flush=True,
                )

    end_total = sum(v[0] for v in best.values())
    print(
        f"\n[HarvestRescue Done in {time.time() - t0:.1f}s] {start_total:,} -> {end_total:,} "
        f"(improved {improved} puzzles, -{saved} steps)",
        flush=True,
    )


if __name__ == "__main__":
    main()
