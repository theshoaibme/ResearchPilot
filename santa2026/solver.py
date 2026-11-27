"""Baseline multi-target solver and portfolio combiner for Santa 2026."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from validator import Puzzle, load_puzzles, validate_program


def _binary_chain_for_target(
    target: int,
    one_reg: int,
    dst: int,
) -> list[list]:
    """Build a short exact instruction sequence for `target > 1` using doubling + addition."""
    bits = bin(target)[2:]
    if len(bits) <= 1:
        return []
    insts: list[list] = [["+", one_reg, one_reg, dst]]
    if bits[1] == "1":
        insts.append(["+", dst, one_reg, dst])
    for b in bits[2:]:
        insts.append(["+", dst, dst, dst])
        if b == "1":
            insts.append(["+", dst, one_reg, dst])
    return insts


def solve_puzzle_baseline(puzzle: Puzzle) -> list[list]:
    """Generate a valid baseline program for a puzzle using available registers."""
    regs: list[int | None] = [None] * puzzle.registers
    produced: dict[int, int] = {}
    for idx, val in enumerate(puzzle.starting_values):
        regs[idx] = val
        produced[val] = idx

    program: list[list] = []
    next_free = len(puzzle.starting_values)
    scratch = puzzle.registers - 1

    # Ensure we have `1` available if needed (any x / x = 1)
    if 1 not in produced:
        src_reg = 0
        one_reg = next_free if next_free < scratch else scratch
        program.append(["/", src_reg, src_reg, one_reg])
        regs[one_reg] = 1
        produced[1] = one_reg
        if one_reg == next_free:
            next_free += 1
    one_reg = produced[1]

    # Sort remaining targets ascending so smaller targets can be reused for larger ones
    remaining = [t for t in puzzle.targets if t not in produced]
    for target in remaining:
        if target in produced:
            continue

        # 1. Check 1-step exact operation from already-held registers
        best_one_step: list | None = None
        held = [(r, v) for r, v in enumerate(regs) if v is not None and r != scratch]
        for r1, v1 in held:
            for r2, v2 in held:
                if v1 + v2 == target:
                    best_one_step = ["+", r1, r2, None]
                    break
                if v1 - v2 == target:
                    best_one_step = ["-", r1, r2, None]
                    break
                if v1 * v2 == target:
                    best_one_step = ["*", r1, r2, None]
                    break
                if v2 != 0 and v1 % v2 == 0 and v1 // v2 == target:
                    best_one_step = ["/", r1, r2, None]
                    break
            if best_one_step is not None:
                break

        dst = next_free if next_free < scratch else scratch
        if best_one_step is not None:
            best_one_step[3] = dst
            program.append(best_one_step)
        else:
            program.extend(_binary_chain_for_target(target=target, one_reg=one_reg, dst=dst))

        regs[dst] = target
        produced[target] = dst
        if dst == next_free and next_free < scratch:
            next_free += 1

    return program


def run_baseline(puzzles_csv: str | Path, out_csv: str | Path) -> int:
    """Solve all puzzles in puzzles_csv and write submission CSV; returns total score."""
    puzzles = load_puzzles(puzzles_csv)
    total = 0
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "program"])
        for pid, puzzle in puzzles.items():
            prog = solve_puzzle_baseline(puzzle)
            res = validate_program(puzzle, prog)
            if not res.valid:
                raise RuntimeError(f"Baseline failed on {pid}: {res.error}")
            total += res.length
            writer.writerow([pid, json.dumps(prog, separators=(",", ":"))])
    return total


def merge_best_submissions(puzzles_csv: str | Path, submission_paths: list[str | Path], out_csv: str | Path) -> int:
    """Combine multiple valid submissions by keeping the shortest valid program per puzzle."""
    puzzles = load_puzzles(puzzles_csv)
    best: dict[str, tuple[int, str]] = {}

    for sub_path in submission_paths:
        with Path(sub_path).open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                pid = row["id"]
                if pid not in puzzles:
                    continue
                res = validate_program(puzzles[pid], row["program"])
                if not res.valid:
                    continue
                if pid not in best or res.length < best[pid][0]:
                    best[pid] = (res.length, row["program"])

    if len(best) != len(puzzles):
        missing = sorted(set(puzzles.keys()) - set(best.keys()))
        raise RuntimeError(f"Missing valid programs for {len(missing)} puzzles, e.g. {missing[0]}")

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "program"])
        for pid in puzzles:
            length, prog_str = best[pid]
            total += length
            writer.writerow([pid, prog_str])
    return total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Santa 2026 baseline solver.")
    parser.add_argument("puzzles_csv", help="Path to puzzles.csv or practice_puzzles.csv")
    parser.add_argument("out_csv", help="Output submission CSV path")
    args = parser.parse_args()

    score = run_baseline(args.puzzles_csv, args.out_csv)
    print(f"Generated baseline submission at {args.out_csv} (total score = {score}).")
