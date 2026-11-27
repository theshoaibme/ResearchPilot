"""Exact HiGHS MILP Free-Order Cycle-Free Sub-DAG Recombiner across distinct lineages."""

from __future__ import annotations

import csv
import json
import pathlib
import subprocess
import sys
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

csv.field_size_limit(1 << 28)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from validator import Puzzle, load_puzzles, validate_program  # noqa: E402


def extract_values(puzzle: Puzzle, prog_str: str) -> list[int]:
    prog = json.loads(prog_str)
    regs: list[int | None] = [None] * puzzle.registers
    for i, v in enumerate(puzzle.starting_values):
        regs[i] = v
    out: list[int] = []
    for op, i, j, d in prog:
        a, b = regs[i], regs[j]
        assert a is not None and b is not None
        r = a + b if op == "+" else a - b if op == "-" else a * b if op == "*" else a // b
        regs[d] = r
        out.append(r)
    return out


def compile_dag_to_json(puzzle: Puzzle, chosen_recipes: dict[int, tuple[str, int, int]]) -> str | None:
    """Schedule acyclic chosen_recipes {val: (op, a, b)} into <= 48 registers."""
    starts = set(puzzle.starting_values)
    # Prune to reachable from targets
    keep: dict[int, tuple[str, int, int]] = {}
    stack = [t for t in puzzle.targets if t not in starts]
    while stack:
        v = stack.pop()
        if v in starts or v in keep:
            continue
        rec = chosen_recipes[v]
        keep[v] = rec
        if rec[1] not in starts:
            stack.append(rec[1])
        if rec[2] not in starts:
            stack.append(rec[2])

    # Greedy min-live-register topological schedule (with randomized tie-break retries)
    rng = np.random.default_rng(42)
    nodes = list(keep.keys())
    for attempt in range(20):
        use_cnt: dict[int, int] = {}
        for v, (_, a, b) in keep.items():
            use_cnt[a] = use_cnt.get(a, 0) + 1
            if b != a:
                use_cnt[b] = use_cnt.get(b, 0) + 1

        done = set(starts)
        rem = set(nodes)
        order: list[int] = []
        noise = {v: float(rng.random()) for v in nodes}
        ok = True
        while rem:
            ready = [v for v in rem if keep[v][1] in done and keep[v][2] in done]
            if not ready:
                ok = False
                break

            def score_ready(v: int) -> tuple[int, float]:
                _, a, b = keep[v]
                delta = 1 if use_cnt.get(v, 0) > 0 else 0
                if a not in starts and use_cnt.get(a, 0) == 1:
                    delta -= 1
                if b != a and b not in starts and use_cnt.get(b, 0) == 1:
                    delta -= 1
                return (delta, float(v) if attempt == 0 else noise[v])

            best_v = min(ready, key=score_ready)
            rem.remove(best_v)
            done.add(best_v)
            order.append(best_v)
            _, a, b = keep[best_v]
            use_cnt[a] = use_cnt.get(a, 1) - 1
            if b != a:
                use_cnt[b] = use_cnt.get(b, 1) - 1

        if not ok:
            return None

        # Allocate registers linearly
        all_vals = list(puzzle.starting_values) + order
        pos = {v: idx for idx, v in enumerate(all_vals)}
        ns = len(puzzle.starting_values)
        n = len(all_vals)
        last = [-1] * n
        for k in range(ns, n):
            v = all_vals[k]
            _, a, b = keep[v]
            last[pos[a]] = max(last[pos[a]], k)
            last[pos[b]] = max(last[pos[b]], k)

        reg_of = [-1] * n
        holder = [-1] * puzzle.registers
        for i in range(ns):
            reg_of[i] = i
            if last[i] >= ns:
                holder[i] = i

        prog = []
        spill = False
        for k in range(ns, n):
            v = all_vals[k]
            op, a, b = keep[v]
            ia, ib = pos[a], pos[b]
            ra, rb = reg_of[ia], reg_of[ib]
            if last[ia] <= k and ra >= 0:
                holder[ra] = -1
            if last[ib] <= k and rb >= 0:
                holder[rb] = -1
            rd = -1
            for r in range(puzzle.registers):
                if holder[r] == -1:
                    rd = r
                    break
            if rd < 0:
                spill = True
                break
            reg_of[k] = rd
            if last[k] > k:
                holder[rd] = k
            prog.append([op, ra, rb, rd])

        if not spill:
            js = json.dumps(prog, separators=(",", ":"))
            if validate_program(puzzle, js).valid:
                return js
    return None


def find_cycles(chosen: dict[int, tuple[str, int, int]], starts: set[int]) -> list[list[int]]:
    state: dict[int, int] = {}
    parent: dict[int, int] = {}
    cycles: list[list[int]] = []

    for root in chosen:
        if state.get(root, 0) != 0:
            continue
        stack: list[tuple[int, int]] = [(root, 0)]
        while stack:
            u, stage = stack.pop()
            if stage == 0:
                state[u] = 1
                stack.append((u, 1))
                _, a, b = chosen[u]
                for w in (a, b):
                    if w in starts or w not in chosen:
                        continue
                    st = state.get(w, 0)
                    if st == 0:
                        parent[w] = u
                        stack.append((w, 0))
                    elif st == 1:
                        cyc = [u]
                        cur = u
                        while cur != w and cur in parent:
                            cur = parent[cur]
                            cyc.append(cur)
                        if cur == w:
                            cycles.append(cyc)
            else:
                state[u] = 2
    return cycles


def solve_milp_puzzle(puzzle: Puzzle, val_sets: list[list[int]], best_len: int) -> str | None:
    starts = set(puzzle.starting_values)
    pool_set = set(starts)
    for vs in val_sets:
        pool_set.update(vs)
    non_starts = sorted(pool_set - starts)
    if len(non_starts) <= best_len or len(non_starts) > 650:
        return None

    pool = sorted(pool_set)
    val_in_pool = set(pool)
    lim = puzzle.value_limit

    # Enumerate all 1-step recipes among pool values that produce a non-start pool value
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

    # Filter out any non-start with 0 recipes
    non_starts = [v for v in non_starts if recipes_by_val[v]]
    v_idx = {v: idx for idx, v in enumerate(non_starts)}
    nv = len(non_starts)

    # Flatten recipes (skip self-loops)
    all_recipes: list[tuple[int, str, int, int]] = []
    val_recipe_ids: dict[int, list[int]] = {v: [] for v in non_starts}
    for v in non_starts:
        # Cap at 25 recipes per value to keep MILP crisp
        recs = recipes_by_val[v][:25]
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
    c[:nv] = 1.0  # Minimize count of active values!

    lb_vars = np.zeros(n_vars, dtype=float)
    ub_vars = np.ones(n_vars, dtype=float)
    for t in puzzle.targets:
        if t not in starts:
            if t not in v_idx or not val_recipe_ids[t]:
                return None
            lb_vars[v_idx[t]] = 1.0

    # Build sparse constraint matrix:
    # 1) sum_{r in R(v)} y_r - x_v >= 0  ==>  x_v - sum y_r <= 0
    # 2) y_r - x_a <= 0 (if a not in starts), y_r - x_b <= 0 (if b not in starts)
    # 3) sum_{v} x_v <= best_len - 1  (strictly beat best_len!)
    rows_data: list[tuple[list[int], list[float], float]] = []
    for v in non_starts:
        rids = val_recipe_ids[v]
        if not rids:
            ub_vars[v_idx[v]] = 0.0
            continue
        cols = [v_idx[v]] + [nv + r for r in rids]
        vals = [1.0] + [-1.0] * len(rids)
        rows_data.append((cols, vals, 0.0))

    for rid, (v, op, a, b) in enumerate(all_recipes):
        if a not in starts:
            rows_data.append(([nv + rid, v_idx[a]], [1.0, -1.0], 0.0))
        if b != a and b not in starts:
            rows_data.append(([nv + rid, v_idx[b]], [1.0, -1.0], 0.0))

    # Strict improvement cut: sum x_v <= best_len - 1
    rows_data.append((list(range(nv)), [1.0] * nv, float(best_len - 1)))

    integrality = np.ones(n_vars, dtype=int)

    for cut_round in range(12):
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
            options={"time_limit": 2.5},
        )
        if not res.success or res.x is None:
            return None

        sol = res.x
        active_vals = [non_starts[i] for i in range(nv) if sol[i] > 0.5]
        if len(active_vals) >= best_len:
            return None

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
            if js is not None:
                return js
            return None

        # Add lazy cycle-elimination cuts
        for cyc in cycles:
            rids = [nv + chosen_rid[u] for u in cyc if u in chosen_rid]
            if rids:
                rows_data.append((rids, [1.0] * len(rids), float(len(rids) - 1)))

    return None


def main() -> None:
    puzzles = load_puzzles(ROOT / "data" / "puzzles.csv")
    csv_paths = [
        p
        for p in sorted((ROOT / "public_frontier").rglob("*.csv"))
        + [ROOT / "submissions" / "frontier_merged_all.csv", ROOT / "submissions" / "rust_best_v4.csv"]
        if p.exists()
    ]
    maps = [{r["id"]: r["program"] for r in csv.DictReader(p.open("r", encoding="utf-8"))} for p in csv_paths]

    best: dict[str, tuple[int, str]] = {}
    base_map = {
        r["id"]: r["program"]
        for r in csv.DictReader((ROOT / "submissions" / "frontier_merged_all.csv").open("r", encoding="utf-8"))
    }
    for pid, p in puzzles.items():
        vr = validate_program(p, base_map[pid])
        assert vr.valid
        best[pid] = (vr.length, base_map[pid])

    t0 = time.time()
    checked = 0
    improved = 0
    saved = 0
    for pid, p in puzzles.items():
        distinct_vsets: list[list[int]] = []
        seen_sets: list[frozenset[int]] = []
        for m in maps:
            if pid not in m:
                continue
            vs = extract_values(p, m[pid])
            fs = frozenset(vs)
            if fs not in seen_sets and len(vs) <= best[pid][0] * 1.15 + 4:
                seen_sets.append(fs)
                distinct_vsets.append(vs)
        if len(distinct_vsets) < 2:
            continue
        checked += 1
        js = solve_milp_puzzle(p, distinct_vsets, best[pid][0])
        if js is not None:
            vr = validate_program(p, js)
            if vr.valid and vr.length < best[pid][0]:
                diff = best[pid][0] - vr.length
                saved += diff
                improved += 1
                print(f"  [MILP IMP] {pid}: {best[pid][0]} -> {vr.length} (-{diff})", flush=True)
                best[pid] = (vr.length, js)

    out_csv = ROOT / "submissions" / "frontier_milp.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "program"])
        for pid in puzzles:
            w.writerow([pid, best[pid][1]])

    print(
        f"MILP checked {checked} multi-lineage puzzles in {time.time() - t0:.1f}s: "
        f"improved {improved} (-{saved} steps) -> Score = {sum(v[0] for v in best.values()):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
