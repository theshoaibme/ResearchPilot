"""10-core parallel HiGHS MILP Free-Order Sub-DAG Recombiner across all 9 banks."""

from __future__ import annotations

import csv
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

csv.field_size_limit(1 << 28)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from validator import Puzzle, load_puzzles, validate_program  # noqa: E402
from cpp_engines.milp_recombine import compile_dag_to_json, extract_values, find_cycles  # noqa: E402


def _worker(args: tuple[Puzzle, list[list[int]], int]) -> tuple[str, str | None]:
    puzzle, val_sets, best_len = args
    starts = set(puzzle.starting_values)
    pool_set = set(starts)
    for vs in val_sets:
        pool_set.update(vs)
    non_starts = sorted(pool_set - starts)
    if len(non_starts) <= best_len or len(non_starts) > 550:
        return puzzle.id, None

    pool = sorted(pool_set)
    lim = puzzle.value_limit
    recipes_by_val: dict[int, list[tuple[str, int, int]]] = {v: [] for v in non_starts}
    n_p = len(pool)
    for i in range(n_p):
        a = pool[i]
        for j in range(i, n_p):
            b = pool[j]
            s = a + b
            if s <= lim and s in recipes_by_val:
                recipes_by_val[s].append(("+", a, b))
            m = a * b
            if m <= lim and m in recipes_by_val:
                recipes_by_val[m].append(("*", a, b))
            if b > a:
                d = b - a
                if d in recipes_by_val:
                    recipes_by_val[d].append(("-", b, a))
            if a != 0 and b % a == 0:
                q = b // a
                if q in recipes_by_val:
                    recipes_by_val[q].append(("/", b, a))

    non_starts = [v for v in non_starts if recipes_by_val[v]]
    v_idx = {v: idx for idx, v in enumerate(non_starts)}
    nv = len(non_starts)

    all_recipes: list[tuple[int, str, int, int]] = []
    val_recipe_ids: dict[int, list[int]] = {v: [] for v in non_starts}
    for v in non_starts:
        for op, a, b in recipes_by_val[v][:30]:
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

    rows_data.append((list(range(nv)), [1.0] * nv, float(best_len - 1)))
    integrality = np.ones(n_vars, dtype=int)

    for _ in range(6):
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
            options={"time_limit": 0.8},
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
            return puzzle.id, compile_dag_to_json(puzzle, chosen)

        for cyc in cycles:
            rids = [nv + chosen_rid[u] for u in cyc if u in chosen_rid]
            if rids:
                rows_data.append((rids, [1.0] * len(rids), float(len(rids) - 1)))

    return puzzle.id, None


def main() -> None:
    puzzles = load_puzzles(ROOT / "data" / "puzzles.csv")
    csv_paths = [
        p
        for p in sorted((ROOT / "public_frontier").rglob("*.csv"))
        + [ROOT / "submissions" / "hyperwalk_best.csv", ROOT / "submissions" / "rust_best_v4.csv"]
        if p.exists()
    ]
    maps = [{r["id"]: r["program"] for r in csv.DictReader(p.open("r", encoding="utf-8"))} for p in csv_paths]

    base_map = {
        r["id"]: r["program"]
        for r in csv.DictReader((ROOT / "submissions" / "hyperwalk_best.csv").open("r", encoding="utf-8"))
    }
    best: dict[str, tuple[int, str]] = {}
    for pid, p in puzzles.items():
        vr = validate_program(p, base_map[pid])
        assert vr.valid
        best[pid] = (vr.length, base_map[pid])

    tasks: list[tuple[Puzzle, list[list[int]], int]] = []
    for pid, p in puzzles.items():
        distinct_vsets: list[list[int]] = []
        seen: set[frozenset[int]] = set()
        for m in maps:
            if pid not in m:
                continue
            vr = validate_program(p, m[pid])
            if not vr.valid or vr.length > best[pid][0] + 12:
                continue
            vs = extract_values(p, m[pid])
            fs = frozenset(vs)
            if fs not in seen:
                seen.add(fs)
                distinct_vsets.append(vs)
        if len(distinct_vsets) >= 2:
            tasks.append((p, distinct_vsets, best[pid][0]))

    print(f"Running 10-core HiGHS MILP across {len(tasks)} multi-lineage puzzles...", flush=True)
    t0 = time.time()
    improved = 0
    saved = 0
    with ProcessPoolExecutor(os.cpu_count() or 8) as ex:
        for pid, js in ex.map(_worker, tasks, chunksize=4):
            if js is not None:
                vr = validate_program(puzzles[pid], js)
                if vr.valid and vr.length < best[pid][0]:
                    diff = best[pid][0] - vr.length
                    saved += diff
                    improved += 1
                    print(f"  [MILP IMP] {pid}: {best[pid][0]} -> {vr.length} (-{diff})", flush=True)
                    best[pid] = (vr.length, js)

    out_csv = ROOT / "submissions" / "frontier_milp_fast.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "program"])
        for pid in puzzles:
            w.writerow([pid, best[pid][1]])

    print(
        f"Parallel MILP done in {time.time() - t0:.1f}s: improved {improved} (-{saved} steps) "
        f"-> Score = {sum(v[0] for v in best.values()):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
