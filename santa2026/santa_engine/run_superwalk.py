"""10-Core SuperWalk Sweep & Atomic Validator for Santa 2026."""

from __future__ import annotations

import csv
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

csv.field_size_limit(1 << 28)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from validator import Puzzle, load_puzzles, validate_program  # noqa: E402
from cpp_engines.run_hyperwalk import to_pfmt_block  # noqa: E402


def parse_superwalk_stdout(stdout: str) -> list[list] | None:
    lines = stdout.splitlines()
    i = 0
    best_cand: list[list] | None = None
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("R "):
            parts = ln.split()
            new_len = int(parts[3])
            cand = []
            for k in range(1, new_len + 1):
                op_s, a_s, b_s, d_s = lines[i + k].split()
                cand.append([op_s, int(a_s), int(b_s), int(d_s)])
            best_cand = cand
            i += new_len + 1
        else:
            i += 1
    return best_cand


def save_best(best_csv: pathlib.Path, puzzles: dict[str, Puzzle], best: dict[str, tuple[int, str, list[list]]]) -> None:
    tmp_csv = best_csv.with_suffix(".csv.tmp")
    with tmp_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "program"])
        for pid in puzzles:
            w.writerow([pid, best[pid][1]])
    tmp_csv.replace(best_csv)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "top"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    base_seed = int(sys.argv[4]) if len(sys.argv) > 4 else 2026112801

    sw_bin = ROOT / "santa_engine" / "superwalk_bin"
    var_pfmt = ROOT / "cpp_engines" / "variants.pfmt"
    puzzles = load_puzzles(ROOT / "data" / "puzzles.csv")
    best_csv = ROOT / "submissions" / "santa_best.csv"

    best: dict[str, tuple[int, str, list[list]]] = {}
    for row in csv.DictReader(best_csv.open("r", encoding="utf-8")):
        pid = row["id"]
        vr = validate_program(puzzles[pid], row["program"])
        assert vr.valid, f"Invalid baseline {pid}"
        best[pid] = (vr.length, row["program"], json.loads(row["program"]))

    start_total = sum(v[0] for v in best.values())
    print(f"[SuperWalk] Loaded verified santa_best.csv = {start_total:,}", flush=True)

    ranked = sorted(puzzles.values(), key=lambda p: best[p.id][0], reverse=True)
    if mode == "top":
        targets = ranked[:limit]
    elif mode == "sweet":
        # True structured (starts <= 32) 48..155 puzzles where 73% of frontier gains reside!
        from collections import defaultdict
        bmap: dict[int, list[tuple[float, Puzzle]]] = defaultdict(list)
        for p in puzzles.values():
            L = best[p.id][0]
            if max(p.starting_values) <= 32 and 48 <= L <= 155:
                nt = max(1, len([t for t in p.targets if t not in p.starting_values]))
                mb = max(p.targets).bit_length() // 5
                bmap[mb].append((L / nt, p))
        cand_list: list[tuple[float, Puzzle]] = []
        for mb, items in bmap.items():
            s_ipt = sorted(x[0] for x in items)
            fl = s_ipt[max(0, len(s_ipt) // 5)]
            for ipt, p in items:
                cand_list.append((ipt - fl, p))
        cand_list.sort(key=lambda x: x[0], reverse=True)
        targets = [p for _, p in cand_list[:limit]]
    elif mode == "mid":
        targets = [p for p in ranked if 110 <= best[p.id][0] < 210][:limit]
    elif mode == "low":
        targets = [p for p in ranked if 45 <= best[p.id][0] < 110][:limit]
    elif mode == "all":
        targets = [p for p in ranked if best[p.id][0] >= 35][:limit]
    else:
        # Specific comma-separated puzzle IDs, run across all cores with distinct seeds!
        pids = [x.strip() for x in mode.split(",") if x.strip() in puzzles]
        ncores = os.cpu_count() or 10
        targets = [puzzles[pids[i % len(pids)]] for i in range(max(len(pids), ncores))]

    ncores = os.cpu_count() or 10
    print(f"[SuperWalk] Mode={mode}, jobs={len(targets)}, secs={secs}s/job on {ncores} cores...", flush=True)

    lock = threading.Lock()
    improved_puzzles = 0
    saved_steps = 0
    done_jobs = 0
    t0 = time.time()

    def run_job(idx_p: tuple[int, Puzzle]) -> None:
        nonlocal improved_puzzles, saved_steps, done_jobs
        idx, p = idx_p
        with lock:
            cur_prog = best[p.id][2]
        inp = to_pfmt_block(p, cur_prog)
        if mode == "sweet":
            temp = "0.48" if idx % 2 == 0 else "0.35"
            cap = "9" if idx % 2 == 0 else "6"
        else:
            temp = "0.32" if idx % 2 == 0 else "0.22"
            cap = "5" if idx % 3 == 0 else "4"
        cmd = [
            str(sw_bin),
            "--time",
            str(secs),
            "--seed",
            str(base_seed + idx * 104729),
            "--T",
            temp,
            "--cap",
            cap,
            "--rb",
            "0.58",
            "--pair",
            "0.28",
            "--cross",
            "50000",
            "--verbose",
        ]
        if var_pfmt.exists():
            cmd.extend(["--variants", str(var_pfmt)])
        proc = subprocess.run(cmd, input=inp, text=True, capture_output=True, check=True)
        cand = parse_superwalk_stdout(proc.stdout)
        with lock:
            done_jobs += 1
            if cand is not None:
                js = json.dumps(cand, separators=(",", ":"))
                vr = validate_program(p, js)
                if vr.valid and vr.length < best[p.id][0]:
                    diff = best[p.id][0] - vr.length
                    old_len = best[p.id][0]
                    best[p.id] = (vr.length, js, cand)
                    saved_steps += diff
                    improved_puzzles += 1
                    cur_tot = sum(v[0] for v in best.values())
                    save_best(best_csv, puzzles, best)
                    print(
                        f"  *** [BREAKTHROUGH] {p.id}: {old_len} -> {vr.length} (-{diff}) "
                        f"| NEW TOTAL = {cur_tot:,} ***",
                        flush=True,
                    )
            if done_jobs % 10 == 0 or done_jobs == len(targets):
                cur_tot = sum(v[0] for v in best.values())
                print(
                    f"  [Progress {done_jobs}/{len(targets)} | {time.time() - t0:.1f}s] "
                    f"improved={improved_puzzles} (-{saved_steps}) | current_best={cur_tot:,}",
                    flush=True,
                )

    with ThreadPoolExecutor(ncores) as ex:
        list(ex.map(run_job, enumerate(targets)))

    end_total = sum(v[0] for v in best.values())
    print(
        f"[SuperWalk Done in {time.time() - t0:.1f}s] {start_total:,} -> {end_total:,} "
        f"(improved {improved_puzzles} puzzles, -{saved_steps} steps)",
        flush=True,
    )


if __name__ == "__main__":
    main()
