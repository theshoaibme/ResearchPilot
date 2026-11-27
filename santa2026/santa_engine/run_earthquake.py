"""10-Core Earthquake-MILP-SuperWalk Pipeline: Destroys frozen hubs to create fresh +0..+5 competitive lineages,
recombines them via Exact HiGHS Steiner MILP, and collapses them via SuperWalk."""

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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

csv.field_size_limit(1 << 28)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from validator import Puzzle, load_puzzles, validate_program  # noqa: E402
from cpp_engines.milp_recombine import extract_values  # noqa: E402
from cpp_engines.run_hyperwalk import to_pfmt_block  # noqa: E402
from santa_engine.run_superwalk import parse_superwalk_stdout, save_best  # noqa: E402
from santa_engine.steiner_pipeline import solve_steiner_milp  # noqa: E402


def main() -> None:
    lo_len = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    hi_len = int(sys.argv[2]) if len(sys.argv) > 2 else 235
    max_puzzles = int(sys.argv[3]) if len(sys.argv) > 3 else 180
    n_mutants = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    sw_secs = float(sys.argv[5]) if len(sys.argv) > 5 else 18.0
    base_seed = int(sys.argv[6]) if len(sys.argv) > 6 else 202660606

    gen_bin = ROOT / "santa_engine" / "genesis_backbone_bin"
    sw_bin = ROOT / "santa_engine" / "superwalk_bin"
    var_pfmt = ROOT / "cpp_engines" / "variants.pfmt"
    puzzles = load_puzzles(ROOT / "data" / "puzzles.csv")
    best_csv = ROOT / "submissions" / "santa_best.csv"

    best: dict[str, tuple[int, str, list[list]]] = {}
    for row in csv.DictReader(best_csv.open("r", encoding="utf-8")):
        pid = row["id"]
        vr = validate_program(puzzles[pid], row["program"])
        assert vr.valid
        best[pid] = (vr.length, row["program"], json.loads(row["program"]))

    start_total = sum(v[0] for v in best.values())
    targets = [p for p in puzzles.values() if lo_len <= best[p.id][0] <= hi_len]
    targets.sort(key=lambda p: best[p.id][0], reverse=True)
    targets = targets[:max_puzzles]
    ncores = os.cpu_count() or 10

    print(
        f"[Earthquake] Start = {start_total:,} | Generating {n_mutants} forbidden-hub mutant lineages/puzzle "
        f"across {len(targets)} puzzles ([{lo_len}, {hi_len}]) on {ncores} cores...",
        flush=True,
    )

    lock = threading.Lock()
    improved = 0
    saved = 0
    mutants_by_pid: dict[str, list[list[int]]] = {}
    t0 = time.time()

    def gen_mutants(idx_p: tuple[int, Puzzle]) -> None:
        nonlocal improved, saved
        idx, p = idx_p
        with lock:
            cur_prog = best[p.id][2]
        inp = to_pfmt_block(p, cur_prog)
        with tempfile.TemporaryDirectory() as td:
            s_out = pathlib.Path(td) / "muts.txt"
            proc = subprocess.run(
                [
                    str(gen_bin),
                    "--time",
                    "0.01",
                    "--scratch",
                    "0",
                    "--width",
                    "96",
                    "--prodk",
                    "6",
                    "--earthquake",
                    str(n_mutants),
                    "--seed",
                    str(base_seed + idx * 10007),
                    "--seed-out",
                    str(s_out),
                ],
                input=inp,
                text=True,
                capture_output=True,
                check=True,
            )
            cand = parse_superwalk_stdout(proc.stdout)
            vsets: list[list[int]] = []
            if s_out.exists():
                for ln in s_out.read_text().splitlines():
                    pts = ln.split()
                    if len(pts) >= 6 and pts[0] == "V":
                        vsets.append([int(x) for x in pts[6:]])
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
                    print(f"  *** [EQ DIRECT WIN] {p.id}: -{diff} -> {sum(v[0] for v in best.values()):,} ***", flush=True)
            if vsets:
                mutants_by_pid[p.id] = vsets

    with ThreadPoolExecutor(ncores) as ex:
        list(ex.map(gen_mutants, enumerate(targets)))

    total_muts = sum(len(v) for v in mutants_by_pid.values())
    print(
        f"  [Stage 1 Done | {time.time() - t0:.1f}s] Generated {total_muts} fresh forbidden-hub lineages across {len(mutants_by_pid)} puzzles!",
        flush=True,
    )

    # Stage 2: Exact HiGHS Steiner MILP recombination across original + all fresh Earthquake mutants!
    milp_jobs: list[tuple[Puzzle, list[int], int, float]] = []
    for pid, vsets in mutants_by_pid.items():
        p = puzzles[pid]
        pool = set(extract_values(p, best[pid][1]))
        for vs in vsets:
            pool.update(vs)
        milp_jobs.append((p, sorted(pool), best[pid][0], 2.5))

    print(f"  [Stage 2] Running 10-core HiGHS Steiner MILP recombination across {len(milp_jobs)} multi-mutant pools...", flush=True)
    milp_hits = 0
    with ProcessPoolExecutor(ncores) as ex:
        for pid, ordered_vals in ex.map(solve_steiner_milp, milp_jobs, chunksize=3):
            if ordered_vals is not None and len(ordered_vals) < best[pid][0]:
                milp_hits += 1
                print(f"    *** [EQ-MILP HIT] {pid}: {best[pid][0]} -> {len(ordered_vals)} unique values! ***", flush=True)
                mutants_by_pid[pid].insert(0, ordered_vals)

    print(f"  [Stage 2 Done | {time.time() - t0:.1f}s] MILP Steiner hits = {milp_hits}", flush=True)

    # Stage 3: Anneal & collapse fresh Earthquake lineages via SuperWalk!
    print(f"  [Stage 3] Collapsing {len(mutants_by_pid)} multi-mutant puzzle basins via SuperWalk ({sw_secs}s/puzzle)...", flush=True)
    done3 = 0

    def collapse_one(idx_pid: tuple[int, str]) -> None:
        nonlocal improved, saved, done3
        idx, pid = idx_pid
        p = puzzles[pid]
        with lock:
            cur_prog = best[pid][2]
            vss = mutants_by_pid[pid]
        inp = to_pfmt_block(p, cur_prog)
        with tempfile.TemporaryDirectory() as td:
            sf = pathlib.Path(td) / "seeds.txt"
            with sf.open("w", encoding="utf-8") as f:
                for vs in vss:
                    f.write(
                        f"V {pid} {best[pid][0]} {len(vs)} {len(vs) - best[pid][0]} {len(vs)} "
                        + " ".join(map(str, vs))
                        + "\n"
                    )
            cmd = [
                str(sw_bin),
                "--time",
                str(sw_secs),
                "--seed",
                str(base_seed + 900000 + idx * 104729),
                "--T",
                "0.38",
                "--cap",
                "6",
                "--rb",
                "0.55",
                "--pair",
                "0.26",
                "--cross",
                "30000",
                "--seed-sets",
                str(sf),
            ]
            if var_pfmt.exists():
                cmd.extend(["--variants", str(var_pfmt)])
            proc = subprocess.run(cmd, input=inp, text=True, capture_output=True, check=True)
            cand = parse_superwalk_stdout(proc.stdout)

        with lock:
            done3 += 1
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
                        f"  *** [EARTHQUAKE+SW BREAKTHROUGH] {pid}: {old_l} -> {vr.length} (-{diff}) "
                        f"| NEW SCORE = {sum(v[0] for v in best.values()):,} ***",
                        flush=True,
                    )
            if done3 % 25 == 0 or done3 == len(mutants_by_pid):
                print(
                    f"  [Stage 3 {done3}/{len(mutants_by_pid)} | {time.time() - t0:.1f}s] "
                    f"wins={improved} (-{saved}) score={sum(v[0] for v in best.values()):,}",
                    flush=True,
                )

    with ThreadPoolExecutor(ncores) as ex:
        list(ex.map(collapse_one, enumerate(sorted(mutants_by_pid.keys()))))

    end_total = sum(v[0] for v in best.values())
    print(
        f"[Earthquake Done in {time.time() - t0:.1f}s] {start_total:,} -> {end_total:,} "
        f"(improved {improved} puzzles, -{saved} steps)",
        flush=True,
    )


if __name__ == "__main__":
    main()
