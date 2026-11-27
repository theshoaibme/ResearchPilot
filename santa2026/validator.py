"""Simulator and validator for Santa 2026 - The Elf Calculator."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

MAX_INSTRUCTIONS = 4096
MAX_PROGRAM_BYTES = 131072
VALID_OPS = frozenset({"+", "-", "*", "/"})


@dataclass(frozen=True)
class Puzzle:
    """Single Elf Calculator puzzle specification."""

    id: str
    starting_values: tuple[int, ...]
    targets: tuple[int, ...]
    value_limit: int
    registers: int = 48


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of simulating and validating one puzzle program."""

    valid: bool
    length: int
    error: str | None = None


def load_puzzles(csv_path: str | Path) -> dict[str, Puzzle]:
    """Load puzzles from puzzles.csv or practice_puzzles.csv keyed by id."""
    puzzles: dict[str, Puzzle] = {}
    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["id"]
            puzzles[pid] = Puzzle(
                id=pid,
                starting_values=tuple(json.loads(row["starting_values"])),
                targets=tuple(json.loads(row["targets"])),
                value_limit=int(row["value_limit"]),
                registers=int(row["registers"]),
            )
    return puzzles


def validate_program(puzzle: Puzzle, program_raw: str | list[list]) -> ValidationResult:
    """Simulate a program against a puzzle and verify all competition constraints."""
    if isinstance(program_raw, str):
        if len(program_raw.encode("utf-8")) > MAX_PROGRAM_BYTES:
            return ValidationResult(False, 0, "Program exceeds 131,072 UTF-8 bytes")
        try:
            program = json.loads(program_raw)
        except json.JSONDecodeError as exc:
            return ValidationResult(False, 0, f"Invalid JSON: {exc}")
    else:
        program = program_raw
        encoded = json.dumps(program, separators=(",", ":")).encode("utf-8")
        if len(encoded) > MAX_PROGRAM_BYTES:
            return ValidationResult(False, 0, "Program exceeds 131,072 UTF-8 bytes")

    if not isinstance(program, list):
        return ValidationResult(False, 0, "Program must be a JSON list")

    n_inst = len(program)
    if n_inst > MAX_INSTRUCTIONS:
        return ValidationResult(False, n_inst, f"Too many instructions: {n_inst} > {MAX_INSTRUCTIONS}")

    regs: list[int | None] = [None] * puzzle.registers
    produced: set[int] = set()
    for idx, val in enumerate(puzzle.starting_values):
        if idx >= puzzle.registers:
            return ValidationResult(False, n_inst, "Starting values exceed register count")
        regs[idx] = val
        produced.add(val)

    limit = puzzle.value_limit
    num_regs = puzzle.registers

    for step, inst in enumerate(program):
        if not isinstance(inst, list) or len(inst) != 4:
            return ValidationResult(False, n_inst, f"Step {step}: instruction must be 4-element list")
        op, i, j, d = inst
        if op not in VALID_OPS:
            return ValidationResult(False, n_inst, f"Step {step}: invalid op {op!r}")
        if not (isinstance(i, int) and isinstance(j, int) and isinstance(d, int)):
            return ValidationResult(False, n_inst, f"Step {step}: register indices must be ints")
        if not (0 <= i < num_regs and 0 <= j < num_regs and 0 <= d < num_regs):
            return ValidationResult(False, n_inst, f"Step {step}: register index out of range 0..{num_regs - 1}")

        a = regs[i]
        b = regs[j]
        if a is None or b is None:
            return ValidationResult(False, n_inst, f"Step {step}: read from undefined register")

        if op == "+":
            res = a + b
        elif op == "-":
            res = a - b
        elif op == "*":
            res = a * b
        else:
            if b == 0 or a % b != 0:
                return ValidationResult(False, n_inst, f"Step {step}: non-exact or zero division ({a}/{b})")
            res = a // b

        if res <= 0 or res > limit:
            return ValidationResult(False, n_inst, f"Step {step}: value {res} outside 1..{limit}")

        regs[d] = res
        produced.add(res)

    missing = [t for t in puzzle.targets if t not in produced]
    if missing:
        return ValidationResult(False, n_inst, f"Missing {len(missing)} target(s), e.g. {missing[0]}")

    return ValidationResult(True, n_inst)


def score_submission(puzzles: dict[str, Puzzle], submission_csv: str | Path) -> tuple[bool, int, list[str]]:
    """Validate an entire submission.csv against puzzles and return (all_valid, total_score, errors)."""
    errors: list[str] = []
    seen: set[str] = set()
    total_score = 0

    with Path(submission_csv).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "program"]:
            return False, 0, [f"Invalid header: {reader.fieldnames}"]
        for row in reader:
            pid = row["id"]
            if pid not in puzzles:
                errors.append(f"Unknown puzzle id: {pid}")
                continue
            if pid in seen:
                errors.append(f"Duplicate puzzle id: {pid}")
                continue
            seen.add(pid)
            res = validate_program(puzzles[pid], row["program"])
            if not res.valid:
                errors.append(f"{pid}: {res.error}")
            else:
                total_score += res.length

    missing_ids = set(puzzles.keys()) - seen
    if missing_ids:
        errors.append(f"Missing {len(missing_ids)} puzzle(s), e.g. {sorted(missing_ids)[0]}")

    return len(errors) == 0, total_score, errors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate a Santa 2026 submission CSV.")
    parser.add_argument("puzzles_csv", help="Path to puzzles.csv or practice_puzzles.csv")
    parser.add_argument("submission_csv", help="Path to submission.csv")
    args = parser.parse_args()

    loaded = load_puzzles(args.puzzles_csv)
    ok, score, errs = score_submission(loaded, args.submission_csv)
    if ok:
        print(f"VALID submission! Total score = {score} across {len(loaded)} puzzles.")
    else:
        print(f"INVALID submission ({len(errs)} error(s)):")
        for err in errs[:20]:
            print(f"  - {err}")
        raise SystemExit(1)
