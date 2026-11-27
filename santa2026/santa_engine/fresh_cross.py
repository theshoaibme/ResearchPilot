"""Generate fresh high-width Rudolph lineages on top bottleneck puzzles, then MILP + HyperWalk."""

from __future__ import annotations

import csv
import json
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

csv.field_size_limit(1 << 28)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from validator import Puzzle, load_puzzles, validate_program  # noqa: E402
from cpp_engines.milp_fast import _worker as milp_worker  # noqa: E402
from cpp_engines.milp_recombine import extract_values  # noqa: E402
from cpp_engines.run_hyperwalk import to_pfmt_block  # noqa: E402


def main() -> None:
    top_k = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    hw_secs = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0

    rudolph_bin = ROOT / "cpp_engines" / "rudolph_bin"
    hw_bin = ROOT / "cpp_engines" / "hyperwalk_bin"
    var_pfmt = ROOT / "cpp_engines" / "variants.pfmt"
    puzzles = load_puzzles(ROOT / "data" / "puzzles.csv")

    best_csv = ROOT / "submissions" / "santa_best.csv"
    best: dict[str, tuple[int, str, list[list]]] = {}
    for row in csv.DictReader(best_csv.open("r", encoding="utf-8")):
        pid = row["id"]
        vr = validate_program(puzzles[pid], row["program"])
        assert vr.valid
        best[pid] = (vr.length, row["program"], json.loads(row["program"]))

    ranked = sorted(puzzles.values(), key=lambda p: best[p.id][0], reverse=True)[:top_k]
    print(f"Step 1: Running high-width Rudolph (W=32, PROD_K=4) on top {len(ranked)} puzzles...", flush=True)

    settings = [
        (32, 0.18, 60, 80, 16, 40, 1, 3, 3, 4, 5, 2),
        (32, 0.18, 72, 88, 16, 40, 3, 3, 3, 4, 5, 2),
    ]

    fresh_progs: dict[str, list[str]] = {p.id: [] for p in ranked}
    t0 = time.time()

    def run_rudolph(args_p: tuple[tuple, Puzzle]) -> tuple[str, str]:
        st_args, p = args_p
        line = " ".join(
            map(
                str,
                (
                    p.id,
                    p.value_limit,
                    len(p.starting_values),
                    len(p.targets),
                    *p.starting_values,
                    *sorted(p.targets),
                ),
            )
        )
        out = subprocess.run(
            [str(rudolph_bin), *map(str, st_args)],
            input=line + "\n",
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        pid, js = out.split(" ", 1)
        return pid, js

    jobs = [(st, p) for st in settings for p in ranked]
    improved = 0
    saved = 0
    with ThreadPoolExecutor(os.cpu_count() or 8) as ex:
        for pid, js in ex.map(run_rudolph, jobs):
            vr = validate_program(puzzles[pid], js)
            if vr.valid:
                fresh_progs[pid].append(js)
                if vr.length < best[pid][0]:
                    diff = best[pid][0] - vr.length
                    saved += diff
                    improved += 1
                    print(f"  [RUDOLPH IMP] {pid}: {best[pid][0]} -> {vr.length} (-{diff})", flush=True)
                    best[pid] = (vr.length, js, json.loads(js))

    print(f"Rudolph finished in {time.time() - t0:.1f}s (direct improvements: {improved}, -{saved} steps).", flush=True)

    # Step 2: MILP Recombine best + fresh Rudolph lineages
    print("Step 2: Running HiGHS MILP recombination across fresh + best lineages...", flush=True)
    for p in ranked:
        vsets = [extract_values(p, best[p.id][1])]
        for js in fresh_progs[p.id]:
            vsets.append(extract_values(p, js))
        _, m_js = milp_worker((p, vsets, best[p.id][0]))
        if m_js is not None:
            vr = validate_program(p, m_js)
            if vr.valid and vr.length < best[p.id][0]:
                diff = best[p.id][0] - vr.length
                saved += diff
                improved += 1
                print(f"  [FRESH-MILP IMP] {p.id}: {best[p.id][0]} -> {vr.length} (-{diff})", flush=True)
                best[p.id] = (vr.length, m_js, json.loads(m_js))

    # Step 3: HyperWalk annealing starting from fresh Rudolph lineages (cross with variants)
    print(f"Step 3: HyperWalk ({hw_secs}s/puzzle) annealing fresh Rudolph basins...", flush=True)
    seed_path = ROOT / "santa_engine" / "fresh_seeds.txt"
    with seed_path.open("w", encoding="utf-8") as sf:
        for p in ranked:
            for js in fresh_progs[p.id]:
                vals = extract_values(p, js)
                sf.write(
                    f"V {p.id} {best[p.id][0]} {len(vals)} {len(vals) - best[p.id][0]} {len(vals)} "
                    + " ".join(map(str, vals))
                    + "\n"
                )

    def run_hw(idx_p: tuple[int, Puzzle]) -> tuple[str, list[list] | None]:
        idx, p = idx_p
        inp = to_pfmt_block(p, best[p.id][2])
        proc = subprocess.run(
            [
                str(hw_bin),
                "--time",
                str(hw_secs),
                "--seed",
                str(3141592 + idx * 1009),
                "--T",
                "0.35",
                "--cap",
                "5",
                "--rb",
                "0.65",
                "--pair",
                "0.25",
                "--variants",
                str(var_pfmt),
                "--seed-sets",
                str(seed_path),
                "--cross",
                "30000",
            ],
            input=inp,
            text=True,
            capture_output=True,
            check=True,
        )
        lines = proc.stdout.splitlines()
        i = 0
        best_cand = None
        while i < len(lines):
            ln = lines[i].strip()
            if ln.startswith("R "):
                new_len = int(ln.split()[3])
                cand = []
                for k in range(1, new_len + 1):
                    op_s, a_s, b_s, d_s = lines[i + k].split()
                    cand.append([op_s, int(a_s), int(b_s), int(d_s)])
                best_cand = cand
                i += new_len + 1
            else:
                i += 1
        return p.id, best_cand

    with ThreadPoolExecutor(os.cpu_count() or 8) as ex:
        for pid, cand in ex.map(run_hw, enumerate(ranked)):
            if cand is not None:
                js = json.dumps(cand, separators=(",", ":"))
                vr = validate_program(puzzles[pid], js)
                if vr.valid and vr.length < best[pid][0]:
                    diff = best[pid][0] - vr.length
                    saved += diff
                    improved += 1
                    print(f"  [FRESH-HW IMP] {pid}: {best[pid][0]} -> {vr.length} (-{diff})", flush=True)
                    best[pid] = (vr.length, js, cand)

    if improved > 0:
        with best_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "program"])
            for pid in puzzles:
                w.writerow([pid, best[pid][1]])

    print(
        f"Fresh-Cross Pipeline done in {time.time() - t0:.1f}s: improved {improved} (-{saved} steps) "
        f"-> Verified score = {sum(v[0] for v in best.values()):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
