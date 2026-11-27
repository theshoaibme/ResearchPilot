"""Merge best verified programs and run multi-core Value-Set MITM LNS to beat 127,240."""

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
from solver import merge_best_submissions  # noqa: E402
from validator import Puzzle, load_puzzles, validate_program  # noqa: E402


def extract_program_values(puzzle: Puzzle, prog_str: str) -> list[int]:
    prog = json.loads(prog_str)
    regs: list[int | None] = [None] * puzzle.registers
    for idx, val in enumerate(puzzle.starting_values):
        regs[idx] = val
    vals: list[int] = []
    for op, i, j, d in prog:
        a = regs[i]
        b = regs[j]
        assert a is not None and b is not None
        if op == "+":
            res = a + b
        elif op == "-":
            res = a - b
        elif op == "*":
            res = a * b
        else:
            res = a // b
        regs[d] = res
        vals.append(res)
    return vals


def main() -> None:
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 777

    eng_dir = pathlib.Path(__file__).resolve().parent
    cpp_path = eng_dir / "lns_frontier.cpp"
    bin_path = eng_dir / "lns_frontier_bin"

    subprocess.run(
        ["clang++", "-std=c++17", "-O3", str(cpp_path), "-o", str(bin_path)],
        check=True,
    )

    puzzles = load_puzzles(ROOT / "data" / "puzzles.csv")
    base_csv = ROOT / "submissions" / "frontier_base.csv"
    sources = [
        p
        for p in [
            ROOT / "public_frontier" / "submission.csv",
            ROOT / "public_frontier" / "submission_live_merge.csv",
            ROOT / "submissions" / "rust_best_v4.csv",
            ROOT / "submissions" / "frontier_polished.csv",
        ]
        if p.exists()
    ]
    base_score = merge_best_submissions(ROOT / "data" / "puzzles.csv", sources, base_csv)
    print(f"Verified frontier base score: {base_score:,} across {len(puzzles)} puzzles", flush=True)

    best: dict[str, tuple[int, str]] = {}
    with base_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pid = row["id"]
            res = validate_program(puzzles[pid], row["program"])
            assert res.valid, f"Invalid base {pid}: {res.error}"
            best[pid] = (res.length, row["program"])

    # Prioritize puzzles with length >= 50 (where LNS finds improvements)
    active_puzzles = [p for p in puzzles.values() if best[p.id][0] >= 35]
    chunks = [active_puzzles[i : i + 5] for i in range(0, len(active_puzzles), 5)]

    def run_chunk(chunk_info: tuple[int, list[Puzzle]]) -> list[tuple[str, str]]:
        c_idx, chunk = chunk_info
        lines: list[str] = []
        for p in chunk:
            vals = extract_program_values(p, best[p.id][1])
            lines.append(
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
            )
        proc = subprocess.run(
            [str(bin_path), str(secs), str(seed + c_idx * 997)],
            input="\n".join(lines) + "\n",
            text=True,
            capture_output=True,
            check=True,
        )
        if proc.stderr.strip():
            for ln in proc.stderr.strip().splitlines():
                print(" ", ln, flush=True)
        out_pairs: list[tuple[str, str]] = []
        for ln in proc.stdout.splitlines():
            if " " in ln:
                pid, js = ln.split(" ", 1)
                if js != "NONE":
                    out_pairs.append((pid, js))
        return out_pairs

    t0 = time.time()
    n_workers = os.cpu_count() or 8
    improved = 0
    saved_steps = 0
    with ThreadPoolExecutor(n_workers) as ex:
        for res_list in ex.map(run_chunk, enumerate(chunks)):
            for pid, js in res_list:
                vr = validate_program(puzzles[pid], js)
                if vr.valid and vr.length < best[pid][0]:
                    saved_steps += best[pid][0] - vr.length
                    improved += 1
                    best[pid] = (vr.length, js)

    out_csv = ROOT / "submissions" / "frontier_polished.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "program"])
        for pid in puzzles:
            w.writerow([pid, best[pid][1]])

    final_score = sum(v[0] for v in best.values())
    print(
        f"Finished LNS in {time.time() - t0:.1f}s: improved {improved} puzzles (-{saved_steps} steps) "
        f"-> Final verified score = {final_score:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
