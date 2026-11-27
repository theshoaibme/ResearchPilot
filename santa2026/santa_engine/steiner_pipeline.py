"""10-Core Steiner Beam-MILP-SuperWalk Pipeline for Santa 2026.

Stage 1: C++20 Deep Backward-Beam Cone Ruin + Coupled 2-Target Top-K Cone Search + Multi-Cone Harvester
Stage 2: 10-Core HiGHS Exact Steiner Hypergraph MILP over Harvested Multi-Cone Pools + Public Variants
Stage 3: C++20 SuperWalk 48-Register Scheduler (with Peak-Buster + Dynamic Remat + Register-Relaxation Burst)
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

csv.field_size_limit(1 << 28)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from validator import Puzzle, load_puzzles, validate_program  # noqa: E402
from cpp_engines.milp_recombine import extract_values, find_cycles  # noqa: E402
from cpp_engines.run_hyperwalk import to_pfmt_block  # noqa: E402
from santa_engine.run_superwalk import parse_superwalk_stdout, save_best  # noqa: E402


def solve_steiner_milp(args: tuple[Puzzle, list[int], int, float]) -> tuple[str, list[int] | None]:
    """Find an acyclic Steiner value subset strictly smaller than best_len."""
    puzzle, candidate_pool, best_len, time_lim = args
    starts = set(puzzle.starting_values)
    pool_set = set(starts) | set(candidate_pool)
    non_starts = sorted(pool_set - starts)
    if len(non_starts) <= best_len or len(non_starts) > 1350:
        return puzzle.id, None

    pool = sorted(pool_set)
    lim = puzzle.value_limit
    recipes_by_val: dict[int, list[tuple[str, int, int]]] = {v: [] for v in non_starts}
    n_p = len(pool)
    for i in range(n_p):
        a = pool[i]
        lim_add = lim - a
        lim_mul = lim // a if a > 1 else lim
        for j in range(i, n_p):
            b = pool[j]
            if b <= lim_add:
                s = a + b
                if s in recipes_by_val:
                    recipes_by_val[s].append(("+", a, b))
            if a > 1 and b <= lim_mul:
                m = a * b
                if m in recipes_by_val:
                    recipes_by_val[m].append(("*", a, b))
            if b > a:
                d = b - a
                if d in recipes_by_val:
                    recipes_by_val[d].append(("-", b, a))
            if a > 1 and b % a == 0:
                q = b // a
                if q in recipes_by_val:
                    recipes_by_val[q].append(("/", b, a))

    # Also allow 1 = s / s from starting value
    if 1 in recipes_by_val and puzzle.starting_values:
        s0 = puzzle.starting_values[0]
        recipes_by_val[1].insert(0, ("/", s0, s0))

    non_starts = [v for v in non_starts if recipes_by_val[v]]
    v_idx = {v: idx for idx, v in enumerate(non_starts)}
    nv = len(non_starts)

    all_recipes: list[tuple[int, str, int, int]] = []
    val_recipe_ids: dict[int, list[int]] = {v: [] for v in non_starts}
    for v in non_starts:
        for op, a, b in recipes_by_val[v][:48]:
            if a == v or b == v:
                continue
            if (a not in starts and a not in v_idx) or (b not in starts and b not in v_idx):
                continue
            rid = len(all_recipes)
            all_recipes.append((v, op, a, b))
            val_recipe_ids[v].append(rid)

    nr = len(all_recipes)
    n_vars = nv + nr
    c = np.zeros(n_vars, dtype=float)
    c[:nv] = 1.0

    lb_vars = np.zeros(n_vars, dtype=float)
    ub_vars = np.ones(n_vars, dtype=float)
    for t in puzzle.targets:
        if t not in starts:
            if t not in v_idx or not val_recipe_ids[t]:
                return puzzle.id, None
            lb_vars[v_idx[t]] = 1.0

    rows_data: list[tuple[list[int], list[float], float]] = []
    for v in non_starts:
        rids = val_recipe_ids[v]
        if not rids:
            ub_vars[v_idx[v]] = 0.0
            continue
        rows_data.append(([v_idx[v]] + [nv + r for r in rids], [1.0] + [-1.0] * len(rids), 0.0))

    for rid, (_, _, a, b) in enumerate(all_recipes):
        if a not in starts:
            rows_data.append(([nv + rid, v_idx[a]], [1.0, -1.0], 0.0))
        if b != a and b not in starts:
            rows_data.append(([nv + rid, v_idx[b]], [1.0, -1.0], 0.0))

    # Strictly beat best_len!
    rows_data.append((list(range(nv)), [1.0] * nv, float(best_len - 1)))
    integrality = np.ones(n_vars, dtype=int)

    for _ in range(10):
        m_rows = len(rows_data)
        mat = lil_matrix((m_rows, n_vars), dtype=float)
        ub_con = np.empty(m_rows, dtype=float)
        for r_i, (cols, vals, rhs) in enumerate(rows_data):
            mat.rows[r_i] = cols
            mat.data[r_i] = vals
            ub_con[r_i] = rhs

        res = milp(
            c=c,
            integrality=integrality,
            bounds=Bounds(lb_vars, ub_vars),
            constraints=LinearConstraint(mat.tocsc(), -np.inf, ub_con),
            options={"time_limit": time_lim},
        )
        if not res.success or res.x is None:
            return puzzle.id, None

        sol = res.x
        active_vals = [non_starts[i] for i in range(nv) if sol[i] > 0.5]
        if len(active_vals) >= best_len:
            return puzzle.id, None

        chosen: dict[int, tuple[str, int, int]] = {}
        chosen_rid: dict[int, int] = {}
        for v in active_vals:
            for rid in val_recipe_ids[v]:
                if sol[nv + rid] > 0.5:
                    _, op, a, b = all_recipes[rid]
                    chosen[v] = (op, a, b)
                    chosen_rid[v] = rid
                    break

        cycles = find_cycles(chosen, starts)
        if not cycles:
            # Topological order of active_vals
            done = set(starts)
            ordered: list[int] = []
            rem = set(active_vals)
            while rem:
                progressed = False
                for v in list(rem):
                    _, a, b = chosen[v]
                    if a in done and b in done:
                        ordered.append(v)
                        done.add(v)
                        rem.remove(v)
                        progressed = True
                if not progressed:
                    return puzzle.id, None
            return puzzle.id, ordered

        for cyc in cycles:
            rids = [nv + chosen_rid[u] for u in cyc if u in chosen_rid]
            if rids:
                rows_data.append((rids, [1.0] * len(rids), float(len(rids) - 1)))

    return puzzle.id, None


def main() -> None:
    lo_len = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    hi_len = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    max_puzzles = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    milp_secs = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0

    beam_bin = ROOT / "santa_engine" / "beam_lns_bin"
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
        f"[SteinerPipeline] Loaded {start_total:,} | Running on {len(targets)} puzzles "
        f"(len in [{lo_len}, {hi_len}]) across {ncores} cores...",
        flush=True,
    )

    # Also gather alternative public CSV value sets to enrich Steiner pools
    pub_csvs = sorted((ROOT / "public_frontier").rglob("*.csv"))
    pub_maps = [{r["id"]: r["program"] for r in csv.DictReader(c.open("r", encoding="utf-8"))} for c in pub_csvs]

    t0 = time.time()
    improved = 0
    saved = 0
    harvested_pools: dict[str, list[int]] = {}
    unsched_seeds: dict[str, list[list[int]]] = {}

    def run_beam_one(idx_p: tuple[int, Puzzle]) -> tuple[str, list[list] | None, list[int], list[list[int]]]:
        idx, p = idx_p
        inp = to_pfmt_block(p, best[p.id][2])
        with tempfile.TemporaryDirectory() as td:
            td_path = pathlib.Path(td)
            s_out = td_path / "seeds.txt"
            h_out = td_path / "harv.txt"
            proc = subprocess.run(
                [
                    str(beam_bin),
                    "--width",
                    "96",
                    "--prodk",
                    "5",
                    "--pairs",
                    "1",
                    "--seed",
                    str(20269000 + idx * 7919),
                    "--seed-out",
                    str(s_out),
                    "--harvest-out",
                    str(h_out),
                ],
                input=inp,
                text=True,
                capture_output=True,
                check=True,
            )
            cand = parse_superwalk_stdout(proc.stdout)
            pool: list[int] = []
            if h_out.exists():
                for ln in h_out.read_text().splitlines():
                    pts = ln.split()
                    if len(pts) >= 3 and pts[0] == "H":
                        pool.extend(int(x) for x in pts[3:])
            vsets: list[list[int]] = []
            if s_out.exists():
                for ln in s_out.read_text().splitlines():
                    pts = ln.split()
                    if len(pts) >= 6 and pts[0] == "V":
                        vsets.append([int(x) for x in pts[6:]])
            return p.id, cand, pool, vsets

    print("  [Stage 1] Running C++20 Deep Backward-Beam Ruin + Multi-Cone Harvester...", flush=True)
    with ThreadPoolExecutor(ncores) as ex:
        for done_i, (pid, cand, pool, vsets) in enumerate(ex.map(run_beam_one, enumerate(targets)), 1):
            if cand is not None:
                js = json.dumps(cand, separators=(",", ":"))
                vr = validate_program(puzzles[pid], js)
                if vr.valid and vr.length < best[pid][0]:
                    diff = best[pid][0] - vr.length
                    print(f"    *** [BEAM DIRECT WIN] {pid}: {best[pid][0]} -> {vr.length} (-{diff}) ***", flush=True)
                    best[pid] = (vr.length, js, cand)
                    improved += 1
                    saved += diff
                    save_best(best_csv, puzzles, best)
            if pool:
                harvested_pools[pid] = pool
            if vsets:
                unsched_seeds.setdefault(pid, []).extend(vsets)
            if done_i % 25 == 0 or done_i == len(targets):
                print(
                    f"    [Stage 1 {done_i}/{len(targets)} | {time.time() - t0:.1f}s] "
                    f"direct_imp={improved} (-{saved}) unsched_hits={len(unsched_seeds)}",
                    flush=True,
                )

    # Enrich harvested pools with public frontier variants and run Stage 2 (10-core HiGHS Steiner MILP)
    milp_jobs: list[tuple[Puzzle, list[int], int, float]] = []
    for p in targets:
        pid = p.id
        pset = set(harvested_pools.get(pid, []))
        pset.update(extract_values(p, best[pid][1]))
        for m in pub_maps:
            if pid in m:
                vr = validate_program(p, m[pid])
                if vr.valid and vr.length <= best[pid][0] + 6:
                    pset.update(extract_values(p, m[pid]))
        milp_jobs.append((p, sorted(pset), best[pid][0], milp_secs))

    print(f"  [Stage 2] Running 10-core HiGHS Exact Steiner MILP on {len(milp_jobs)} enriched cone pools...", flush=True)
    milp_hits = 0
    with ProcessPoolExecutor(ncores) as ex:
        for pid, ordered_vals in ex.map(solve_steiner_milp, milp_jobs, chunksize=2):
            if ordered_vals is not None and len(ordered_vals) < best[pid][0]:
                milp_hits += 1
                print(
                    f"    [MILP STEINER HIT] {pid}: value-set {best[pid][0]} -> {len(ordered_vals)}!",
                    flush=True,
                )
                unsched_seeds.setdefault(pid, []).append(ordered_vals)

    print(
        f"  [Stage 2 Done | {time.time() - t0:.1f}s] MILP Steiner value-set hits={milp_hits}, "
        f"total candidate seed puzzles={len(unsched_seeds)}",
        flush=True,
    )

    # Stage 3: Schedule & collapse all candidate value sets via SuperWalk!
    if unsched_seeds:
        print(f"  [Stage 3] Scheduling & collapsing {len(unsched_seeds)} puzzles via SuperWalk...", flush=True)

        def run_sw_seed(pid_vsets: tuple[str, list[list[int]]]) -> tuple[str, list[list] | None]:
            pid, vsets = pid_vsets
            p = puzzles[pid]
            inp = to_pfmt_block(p, best[pid][2])
            with tempfile.TemporaryDirectory() as td:
                sf = pathlib.Path(td) / "seeds.txt"
                with sf.open("w", encoding="utf-8") as f:
                    for vs in vsets:
                        f.write(
                            f"V {pid} {best[pid][0]} {len(vs)} {len(vs) - best[pid][0]} {len(vs)} "
                            + " ".join(map(str, vs))
                            + "\n"
                        )
                cmd = [
                    str(sw_bin),
                    "--time",
                    "18.0",
                    "--seed",
                    "20267777",
                    "--T",
                    "0.25",
                    "--cap",
                    "4",
                    "--seed-sets",
                    str(sf),
                ]
                if var_pfmt.exists():
                    cmd.extend(["--variants", str(var_pfmt)])
                proc = subprocess.run(cmd, input=inp, text=True, capture_output=True, check=True)
                return pid, parse_superwalk_stdout(proc.stdout)

        with ThreadPoolExecutor(ncores) as ex:
            for pid, cand in ex.map(run_sw_seed, list(unsched_seeds.items())):
                if cand is not None:
                    js = json.dumps(cand, separators=(",", ":"))
                    vr = validate_program(puzzles[pid], js)
                    if vr.valid and vr.length < best[pid][0]:
                        diff = best[pid][0] - vr.length
                        print(f"    *** [STEINER+SW WIN] {pid}: {best[pid][0]} -> {vr.length} (-{diff}) ***", flush=True)
                        best[pid] = (vr.length, js, cand)
                        improved += 1
                        saved += diff
                        save_best(best_csv, puzzles, best)

    end_total = sum(v[0] for v in best.values())
    print(
        f"[SteinerPipeline Done in {time.time() - t0:.1f}s] {start_total:,} -> {end_total:,} "
        f"(improved {improved} puzzles, -{saved} steps)",
        flush=True,
    )


if __name__ == "__main__":
    main()
