"""10-Core Exact Hub-Pool Steiner MILP Optimizer for Santa 2026.

Constructs a dense candidate hub pool around each frontier DAG V0:
  1. All current values V0 + any distinct public/mutant variant values
  2. Top ~500 S1(U0) high-multiplicity hub candidates (cnt >= 2 relations with U0, plus a/2 and isqrt(a)),
     weighted toward hubs touching targets and removable helpers
  3. Top ~150 2-step bridge candidates (connecting top S1 hubs to targets/starts)
Then solves the Global Exact Cycle-Free Minimum Steiner Hypergraph via HiGHS MILP (sum(x_v) <= L0 - 1)
to simultaneously discover arbitrary k-for-(k-1) multi-hub replacements (3-for-2, 5-for-4, 8-for-6, etc.)!
"""

from __future__ import annotations

import csv
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

csv.field_size_limit(1 << 28)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from validator import Puzzle, load_puzzles, validate_program  # noqa: E402
from cpp_engines.milp_recombine import compile_dag_to_json, extract_values, find_cycles  # noqa: E402
from cpp_engines.run_hyperwalk import to_pfmt_block  # noqa: E402
from santa_engine.run_superwalk import parse_superwalk_stdout, save_best  # noqa: E402


def build_hub_pool(puzzle: Puzzle, base_vals: list[int], extra_vals: list[int], max_new: int = 580) -> list[int]:
    starts = set(puzzle.starting_values)
    tset = set(puzzle.targets)
    u0_set = starts | set(base_vals)
    u0 = sorted(u0_set)
    lim = puzzle.value_limit

    score: Counter[int] = Counter()
    n = len(u0)

    for i in range(n):
        a = u0[i]
        a_t = 2 if a in tset else 1
        # Self-halving & exact square root (x+x=a, x*x=a)
        if (a & 1) == 0:
            h = a >> 1
            if h > 0 and h not in u0_set:
                score[h] += 2 + a_t
        if a >= 4:
            rt = math.isqrt(a)
            if rt * rt == a and rt not in u0_set:
                score[rt] += 3 + a_t

        lim_add = lim - a
        lim_mul = lim // a if a > 1 else 0
        for j in range(i + 1, n):
            b = u0[j]
            w = a_t + (2 if b in tset else 1)
            if b <= lim_add:
                s = a + b
                if s not in u0_set:
                    score[s] += w
            d = b - a
            if d > 0 and d not in u0_set:
                score[d] += w
            if a > 1:
                if b <= lim_mul:
                    m = a * b
                    if m not in u0_set:
                        score[m] += w
                if b % a == 0:
                    q = b // a
                    if q not in u0_set:
                        score[q] += w + 1

    # Keep only candidates with >= 2 independent relations (score >= 4)
    s1_ranked = [v for v, sc in score.most_common(max_new * 2) if sc >= 4][: (max_new * 3) // 4]
    top_s1 = s1_ranked[:90]

    # 2-step bridge candidates: values connecting top_s1 with starts/targets that ALSO touch U0 U top_s1!
    anchors = sorted(starts | (tset - starts))
    u1_set = u0_set | set(s1_ranked)
    bridge_sc: Counter[int] = Counter()
    for h in top_s1:
        lim_add = lim - h
        lim_mul = lim // h if h > 1 else 0
        for a in anchors:
            if a <= lim_add:
                s = h + a
                if s not in u1_set:
                    bridge_sc[s] += 1
            d = h - a if h > a else a - h
            if d > 0 and d not in u1_set:
                bridge_sc[d] += 1
            if h > 1 and a > 1:
                if a <= lim_mul:
                    m = h * a
                    if m not in u1_set:
                        bridge_sc[m] += 1
                if h > a and h % a == 0:
                    q = h // a
                    if q not in u1_set:
                        bridge_sc[q] += 1
                elif a > h and a % h == 0:
                    q = a // h
                    if q not in u1_set:
                        bridge_sc[q] += 1

    bridges = [v for v, c in bridge_sc.most_common(300) if c >= 2][: max_new - len(s1_ranked)]
    pool = set(base_vals) | set(extra_vals) | set(s1_ranked) | set(bridges)
    return sorted(pool - starts)


def solve_one_hub_milp(args: tuple[Puzzle, list[int], list[int], int, float]) -> tuple[str, str | None, list[int] | None]:
    puzzle, base_vals, extra_vals, best_len, time_lim = args
    starts = set(puzzle.starting_values)
    non_starts = build_hub_pool(puzzle, base_vals, extra_vals, max_new=560)
    if len(non_starts) <= best_len or len(non_starts) > 1100:
        return puzzle.id, None, None

    pool = sorted(starts | set(non_starts))
    lim = puzzle.value_limit
    recipes_by_val: dict[int, list[tuple[str, int, int]]] = {v: [] for v in non_starts}
    if 1 in recipes_by_val and puzzle.starting_values:
        s0 = puzzle.starting_values[0]
        recipes_by_val[1].append(("/", s0, s0))

    n_p = len(pool)
    for i in range(n_p):
        a = pool[i]
        lim_add = lim - a
        lim_mul = lim // a if a > 1 else 0
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

    non_starts = [v for v in non_starts if recipes_by_val[v]]
    v_idx = {v: idx for idx, v in enumerate(non_starts)}
    nv = len(non_starts)

    # Prioritize recipes whose operands are in starts or base_vals!
    base_set = starts | set(base_vals)
    all_recipes: list[tuple[int, str, int, int]] = []
    val_recipe_ids: dict[int, list[int]] = {v: [] for v in non_starts}
    for v in non_starts:
        recs = recipes_by_val[v]
        if len(recs) > 44:
            recs.sort(key=lambda r: (0 if r[1] in base_set else 1) + (0 if r[2] in base_set else 1))
            recs = recs[:44]
        for op, a, b in recs:
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
                return puzzle.id, None, None
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

    rows_data.append((list(range(nv)), [1.0] * nv, float(best_len - 1)))
    integrality = np.ones(n_vars, dtype=int)

    for _ in range(12):
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
            return puzzle.id, None, None

        sol = res.x
        active_vals = [non_starts[i] for i in range(nv) if sol[i] > 0.5]
        if len(active_vals) >= best_len:
            return puzzle.id, None, None

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
            js = compile_dag_to_json(puzzle, chosen)
            # Also return topological value order for SuperWalk 48-reg scheduling if greedy compile spilled!
            done = set(starts)
            ordered: list[int] = []
            rem = set(active_vals)
            while rem:
                prog = False
                for v in list(rem):
                    _, a, b = chosen[v]
                    if a in done and b in done:
                        ordered.append(v)
                        done.add(v)
                        rem.remove(v)
                        prog = True
                if not progressed_ok(prog):
                    break
            return puzzle.id, js, ordered

        for cyc in cycles:
            rids = [nv + chosen_rid[u] for u in cyc if u in chosen_rid]
            if rids:
                rows_data.append((rids, [1.0] * len(rids), float(len(rids) - 1)))

    return puzzle.id, None, None


def progressed_ok(flag: bool) -> bool:
    return flag


def main() -> None:
    lo_len = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    hi_len = int(sys.argv[2]) if len(sys.argv) > 2 else 175
    time_lim = float(sys.argv[3]) if len(sys.argv) > 3 else 3.5

    puzzles = load_puzzles(ROOT / "data" / "puzzles.csv")
    best_csv = ROOT / "submissions" / "santa_best.csv"
    sw_bin = ROOT / "santa_engine" / "superwalk_bin"
    var_pfmt = ROOT / "cpp_engines" / "variants.pfmt"

    best: dict[str, tuple[int, str, list[list]]] = {}
    for row in csv.DictReader(best_csv.open("r", encoding="utf-8")):
        pid = row["id"]
        vr = validate_program(puzzles[pid], row["program"])
        assert vr.valid
        best[pid] = (vr.length, row["program"], json.loads(row["program"]))

    pub_csvs = sorted((ROOT / "public_frontier").rglob("*.csv"))
    pub_maps = [{r["id"]: r["program"] for r in csv.DictReader(c.open("r", encoding="utf-8"))} for c in pub_csvs]

    start_total = sum(v[0] for v in best.values())
    targets = [p for p in puzzles.values() if lo_len <= best[p.id][0] <= hi_len]
    # Sort ascending so fastest small/medium structured puzzles solve first!
    targets.sort(key=lambda p: best[p.id][0])

    jobs: list[tuple[Puzzle, list[int], list[int], int, float]] = []
    for p in targets:
        pid = p.id
        b_vals = extract_values(p, best[pid][1])
        ex_set: set[int] = set()
        for m in pub_maps:
            if pid in m:
                vr = validate_program(p, m[pid])
                if vr.valid and vr.length <= best[pid][0] + 6:
                    ex_set.update(extract_values(p, m[pid]))
        jobs.append((p, b_vals, sorted(ex_set), best[pid][0], time_lim))

    ncores = os.cpu_count() or 10
    print(
        f"[HubSteinerMILP] Start = {start_total:,} | Solving {len(jobs)} puzzles "
        f"(len in [{lo_len}, {hi_len}], limit={time_lim}s) across {ncores} cores...",
        flush=True,
    )

    t0 = time.time()
    improved = 0
    saved = 0
    unsched: dict[str, list[int]] = {}
    done_cnt = 0

    with ProcessPoolExecutor(ncores) as ex:
        for pid, js, ordered in ex.map(solve_one_hub_milp, jobs, chunksize=2):
            done_cnt += 1
            if js is not None:
                vr = validate_program(puzzles[pid], js)
                if vr.valid and vr.length < best[pid][0]:
                    diff = best[pid][0] - vr.length
                    old_l = best[pid][0]
                    best[pid] = (vr.length, js, json.loads(js))
                    improved += 1
                    saved += diff
                    save_best(best_csv, puzzles, best)
                    print(
                        f"  *** [HUB-MILP DIRECT WIN] {pid}: {old_l} -> {vr.length} (-{diff}) "
                        f"| NEW SCORE = {sum(v[0] for v in best.values()):,} ***",
                        flush=True,
                    )
                    continue
            if ordered is not None and len(ordered) < best[pid][0]:
                print(f"  [HUB-MILP UNSCHED HIT] {pid}: {best[pid][0]} -> {len(ordered)} values!", flush=True)
                unsched[pid] = ordered

            if done_cnt % 50 == 0 or done_cnt == len(jobs):
                print(
                    f"  [Progress {done_cnt}/{len(jobs)} | {time.time() - t0:.1f}s] "
                    f"wins={improved} (-{saved}) unsched={len(unsched)} score={sum(v[0] for v in best.values()):,}",
                    flush=True,
                )

    if unsched:
        print(f"[HubSteinerMILP] Scheduling {len(unsched)} MILP hits via SuperWalk...", flush=True)

        def sched_sw(pid_ord: tuple[str, list[int]]) -> tuple[str, list[list] | None]:
            pid, ord_v = pid_ord
            p = puzzles[pid]
            inp = to_pfmt_block(p, best[pid][2])
            with tempfile.TemporaryDirectory() as td:
                sf = pathlib.Path(td) / "s.txt"
                sf.write_text(
                    f"V {pid} {best[pid][0]} {len(ord_v)} {len(ord_v) - best[pid][0]} {len(ord_v)} "
                    + " ".join(map(str, ord_v))
                    + "\n"
                )
                cmd = [str(sw_bin), "--time", "15.0", "--seed", "2026999", "--seed-sets", str(sf)]
                if var_pfmt.exists():
                    cmd.extend(["--variants", str(var_pfmt)])
                proc = subprocess.run(cmd, input=inp, text=True, capture_output=True, check=True)
                return pid, parse_superwalk_stdout(proc.stdout)

        with ThreadPoolExecutor(ncores) as ex:
            for pid, cand in ex.map(sched_sw, list(unsched.items())):
                if cand is not None:
                    js = json.dumps(cand, separators=(",", ":"))
                    vr = validate_program(puzzles[pid], js)
                    if vr.valid and vr.length < best[pid][0]:
                        diff = best[pid][0] - vr.length
                        old_l = best[pid][0]
                        best[pid] = (vr.length, js, cand)
                        improved += 1
                        saved += diff
                        save_best(best_csv, puzzles, best)
                        print(
                            f"  *** [HUB-MILP+SW WIN] {pid}: {old_l} -> {vr.length} (-{diff}) "
                            f"| NEW SCORE = {sum(v[0] for v in best.values()):,} ***",
                            flush=True,
                        )

    end_total = sum(v[0] for v in best.values())
    print(
        f"[HubSteinerMILP Done in {time.time() - t0:.1f}s] {start_total:,} -> {end_total:,} "
        f"(improved {improved} puzzles, -{saved} steps)",
        flush=True,
    )


if __name__ == "__main__":
    main()
