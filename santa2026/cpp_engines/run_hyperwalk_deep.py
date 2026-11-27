"""Deep Alternative-Basin HyperWalk runner (--seed-sets + high slack/temp annealing)."""

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
from cpp_engines.run_hyperwalk import extract_vset, to_pfmt_block  # noqa: E402


def main() -> None:
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 999888

    eng_dir = pathlib.Path(__file__).resolve().parent
    hw_bin = eng_dir / "hyperwalk_bin"
    puzzles = load_puzzles(ROOT / "data" / "puzzles.csv")

    base_csv = ROOT / "submissions" / "frontier_milp_fast.csv"
    if not base_csv.exists():
        base_csv = ROOT / "submissions" / "hyperwalk_best.csv"

    best: dict[str, tuple[int, str, list[list]]] = {}
    for row in csv.DictReader(base_csv.open("r", encoding="utf-8")):
        pid = row["id"]
        vr = validate_program(puzzles[pid], row["program"])
        assert vr.valid
        best[pid] = (vr.length, row["program"], json.loads(row["program"]))

    all_csvs = sorted((ROOT / "public_frontier").rglob("*.csv"))
    maps = [{r["id"]: r["program"] for r in csv.DictReader(p.open("r", encoding="utf-8"))} for p in all_csvs]

    # Write --seed-sets file: V id orig_ops new_ops delta chg v1 v2 ...
    # Only include genuinely distinct alternative lineages (differing by >= 15% of values and within +4 steps!)
    seed_sets_path = eng_dir / "alt_seeds.txt"
    var_pfmt_path = eng_dir / "variants.pfmt"
    target_pids: list[str] = []

    with seed_sets_path.open("w", encoding="utf-8") as sf:
        for pid, p in puzzles.items():
            b_len, _, b_prog = best[pid]
            b_vs = extract_vset(p, b_prog)
            seen = {b_vs}
            added_for_pid = 0
            for m in maps:
                if pid not in m:
                    continue
                vr = validate_program(p, m[pid])
                if not vr.valid or vr.length > b_len + 4:
                    continue
                prog = json.loads(m[pid])
                vs = extract_vset(p, prog)
                if vs in seen:
                    continue
                overlap = len(vs & b_vs) / max(1, len(b_vs))
                if overlap < 0.88:
                    seen.add(vs)
                    # Replay in order to preserve topological value order
                    regs: list[int | None] = [None] * p.registers
                    for idx, sv in enumerate(p.starting_values):
                        regs[idx] = sv
                    ordered_vs: list[int] = []
                    seen_v = set(p.starting_values)
                    for op, i, j, d in prog:
                        a, b = regs[i], regs[j]
                        assert a is not None and b is not None
                        r = a + b if op == "+" else a - b if op == "-" else a * b if op == "*" else a // b
                        regs[d] = r
                        if r not in seen_v:
                            seen_v.add(r)
                            ordered_vs.append(r)
                    sf.write(
                        f"V {pid} {b_len} {len(ordered_vs)} {len(ordered_vs) - b_len} {len(vs - b_vs)} "
                        + " ".join(map(str, ordered_vs))
                        + "\n"
                    )
                    added_for_pid += 1
            if added_for_pid > 0:
                target_pids.append(pid)

    print(f"Prepared distinct alternative-basin seed sets for {len(target_pids)} puzzles ({secs}s each)", flush=True)
    active = [puzzles[pid] for pid in target_pids]
    chunks = [active[i : i + 2] for i in range(0, len(active), 2)]

    def run_chunk(c_info: tuple[int, list[Puzzle]]) -> list[tuple[str, list[list]]]:
        c_idx, chunk = c_info
        inp = "".join(to_pfmt_block(p, best[p.id][2]) for p in chunk)
        proc = subprocess.run(
            [
                str(hw_bin),
                "--time",
                str(secs),
                "--seed",
                str(seed + c_idx * 1009),
                "--T",
                "0.38",
                "--cap",
                "5",
                "--rb",
                "0.55",
                "--pair",
                "0.30",
                "--variants",
                str(var_pfmt_path),
                "--seed-sets",
                str(seed_sets_path),
                "--cross",
                "30000",
                "--verbose",
            ],
            input=inp,
            text=True,
            capture_output=True,
            check=True,
        )
        out_progs: list[tuple[str, list[list]]] = []
        lines = proc.stdout.splitlines()
        i = 0
        while i < len(lines):
            ln = lines[i].strip()
            if ln.startswith("R "):
                parts = ln.split()
                pid = parts[1]
                new_len = int(parts[3])
                cand: list[list] = []
                for k in range(1, new_len + 1):
                    op_s, a_s, b_s, d_s = lines[i + k].split()
                    cand.append([op_s, int(a_s), int(b_s), int(d_s)])
                out_progs.append((pid, cand))
                i += new_len + 1
            else:
                i += 1
        return out_progs

    t0 = time.time()
    improved = 0
    saved = 0
    with ThreadPoolExecutor(os.cpu_count() or 8) as ex:
        for res_list in ex.map(run_chunk, enumerate(chunks)):
            for pid, cand in res_list:
                js = json.dumps(cand, separators=(",", ":"))
                vr = validate_program(puzzles[pid], js)
                if vr.valid and vr.length < best[pid][0]:
                    diff = best[pid][0] - vr.length
                    saved += diff
                    improved += 1
                    print(f"  [DEEP-HW IMP] {pid}: {best[pid][0]} -> {vr.length} (-{diff})", flush=True)
                    best[pid] = (vr.length, js, cand)

    out_csv = ROOT / "submissions" / "frontier_deep_best.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "program"])
        for pid in puzzles:
            w.writerow([pid, best[pid][1]])

    print(
        f"Deep Alt-Basin HyperWalk finished in {time.time() - t0:.1f}s: improved {improved} (-{saved} steps) "
        f"-> Final verified score = {sum(v[0] for v in best.values()):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
