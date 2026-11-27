"""Anneal distinct close-alternative lineages with C++ Value-Set MITM LNS to tunnel below 127,240."""

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


def main() -> None:
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 202699

    eng_dir = pathlib.Path(__file__).resolve().parent
    bin_path = eng_dir / "lns_frontier_bin"
    if not bin_path.exists():
        subprocess.run(
            ["clang++", "-std=c++17", "-O3", str(eng_dir / "lns_frontier.cpp"), "-o", str(bin_path)],
            check=True,
        )

    puzzles = load_puzzles(ROOT / "data" / "puzzles.csv")
    base_path = ROOT / "submissions" / "frontier_milp.csv"
    if not base_path.exists():
        base_path = ROOT / "submissions" / "frontier_merged_all.csv"

    best: dict[str, tuple[int, str]] = {}
    for row in csv.DictReader(base_path.open("r", encoding="utf-8")):
        pid = row["id"]
        vr = validate_program(puzzles[pid], row["program"])
        assert vr.valid
        best[pid] = (vr.length, row["program"])

    alt_csvs = sorted((ROOT / "public_frontier").rglob("*.csv"))
    alt_maps = [{r["id"]: r["program"] for r in csv.DictReader(p.open("r", encoding="utf-8"))} for p in alt_csvs]

    jobs: list[tuple[Puzzle, list[int]]] = []
    for pid, p in puzzles.items():
        best_vals = frozenset(extract_values(p, best[pid][1]))
        seen = {best_vals}
        for m in alt_maps:
            if pid not in m:
                continue
            vals = extract_values(p, m[pid])
            fs = frozenset(vals)
            # Close alternative lineage within +4 instructions of current best!
            if fs not in seen and len(vals) <= best[pid][0] + 4:
                seen.add(fs)
                jobs.append((p, vals))

    print(f"Found {len(jobs)} close alternative lineages to anneal ({secs}s each across {os.cpu_count()} threads)", flush=True)
    chunks = [jobs[i : i + 3] for i in range(0, len(jobs), 3)]

    def run_chunk(c_info: tuple[int, list[tuple[Puzzle, list[int]]]]) -> list[tuple[str, str]]:
        c_idx, chunk = c_info
        lines = [
            " ".join(
                map(
                    str,
                    (
                        p.id,
                        len(p.starting_values),
                        *p.starting_values,
                        len(p.targets),
                        *p.targets,
                        p.value_limit,
                        len(vals),
                        *vals,
                    ),
                )
            )
            for p, vals in chunk
        ]
        proc = subprocess.run(
            [str(bin_path), str(secs), str(seed + c_idx * 131)],
            input="\n".join(lines) + "\n",
            text=True,
            capture_output=True,
            check=True,
        )
        out: list[tuple[str, str]] = []
        for ln in proc.stdout.splitlines():
            if " " in ln:
                pid, js = ln.split(" ", 1)
                if js != "NONE":
                    out.append((pid, js))
        return out

    t0 = time.time()
    improved = 0
    saved = 0
    with ThreadPoolExecutor(os.cpu_count() or 8) as ex:
        for res_list in ex.map(run_chunk, enumerate(chunks)):
            for pid, js in res_list:
                vr = validate_program(puzzles[pid], js)
                if vr.valid and vr.length < best[pid][0]:
                    diff = best[pid][0] - vr.length
                    saved += diff
                    improved += 1
                    print(f"  [ALT-LNS IMP] {pid}: {best[pid][0]} -> {vr.length} (-{diff})", flush=True)
                    best[pid] = (vr.length, js)

    out_csv = ROOT / "submissions" / "frontier_best.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "program"])
        for pid in puzzles:
            w.writerow([pid, best[pid][1]])

    print(
        f"Alt-Lineage LNS finished in {time.time() - t0:.1f}s: improved {improved} (-{saved} steps) "
        f"-> Final verified score = {sum(v[0] for v in best.values()):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
