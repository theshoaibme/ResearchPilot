"""Multi-threaded runner for the C++ Backward-Beam + Belady Register Allocator portfolio."""

from __future__ import annotations

import csv
import json
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from solver import merge_best_submissions  # noqa: E402
from validator import load_puzzles, validate_program  # noqa: E402

SETTINGS = [
    (20, 0.18, 60, 80, 16, 40, 1),
    (20, 0.18, 60, 80, 16, 40, 4),
    (20, 0.18, 60, 80, 16, 40, 3),
    (20, 0.18, 60, 80, 16, 40, 2),
    (20, 0.18, 60, 80, 16, 40, 0),
    (20, 0.18, 60, 80, 16, 40, 6),
    (20, 0.18, 72, 88, 16, 40, 0),
]


def main() -> None:
    eng_dir = pathlib.Path(__file__).resolve().parent
    bin_path = eng_dir / "rudolph_bin"
    cpp_path = eng_dir / "rudolph.cpp"

    subprocess.run(
        ["clang++", "-std=c++17", "-O3", str(cpp_path), "-o", str(bin_path)],
        check=True,
    )

    puzzles = load_puzzles(ROOT / "data" / "puzzles.csv")
    seed_csv = ROOT / "submissions" / "rust_best_v4.csv"
    best: dict[str, tuple[int, str]] = {}
    if seed_csv.exists():
        with seed_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                pid = row["id"]
                res = validate_program(puzzles[pid], row["program"])
                if res.valid:
                    best[pid] = (res.length, row["program"])

    print(f"Initial seed score: {sum(v[0] for v in best.values())} ({len(best)} puzzles)", flush=True)

    puz_list = list(puzzles.values())
    chunks = [puz_list[i : i + 8] for i in range(0, len(puz_list), 8)]

    def solve_chunk(args: tuple, chunk: list) -> list[tuple[str, str]]:
        lines = [
            " ".join(
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
            for p in chunk
        ]
        out = subprocess.run(
            [str(bin_path), *map(str, args)],
            input="\n".join(lines) + "\n",
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        return [tuple(ln.split(" ", 1)) for ln in out.splitlines() if " " in ln]

    n_workers = os.cpu_count() or 8
    for s_idx, args in enumerate(SETTINGS, 1):
        t0 = time.time()
        arm_tot = 0
        improved = 0
        with ThreadPoolExecutor(n_workers) as ex:
            for chunk_res in ex.map(lambda c: solve_chunk(args, c), chunks):
                for pid, prog_str in chunk_res:
                    res = validate_program(puzzles[pid], prog_str)
                    if not res.valid:
                        raise RuntimeError(f"Invalid program from rudolph on {pid}: {res.error}")
                    arm_tot += res.length
                    if pid not in best or res.length < best[pid][0]:
                        best[pid] = (res.length, prog_str)
                        improved += 1
        cur_tot = sum(v[0] for v in best.values())
        print(
            f"[{s_idx}/{len(SETTINGS)}] {args} -> arm={arm_tot:,} | portfolio={cur_tot:,} "
            f"(improved {improved} puzzles, {time.time() - t0:.1f}s)",
            flush=True,
        )

    out_csv = ROOT / "submissions" / "portfolio_rudolph.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "program"])
        for pid in puzzles:
            w.writerow([pid, best[pid][1]])

    print(f"Saved validated portfolio to {out_csv} (score = {sum(v[0] for v in best.values()):,})", flush=True)


if __name__ == "__main__":
    main()
