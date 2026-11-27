"""Focused 45s/puzzle full-cooling HyperWalk run on the top bottleneck puzzles."""

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
from cpp_engines.run_hyperwalk import to_pfmt_block  # noqa: E402


def main() -> None:
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 45.0
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 2026777

    eng_dir = pathlib.Path(__file__).resolve().parent
    hw_bin = eng_dir / "hyperwalk_bin"
    var_pfmt_path = eng_dir / "variants.pfmt"
    puzzles = load_puzzles(ROOT / "data" / "puzzles.csv")

    base_csv = ROOT / "submissions" / "frontier_deep_best.csv"
    best: dict[str, tuple[int, str, list[list]]] = {}
    for row in csv.DictReader(base_csv.open("r", encoding="utf-8")):
        pid = row["id"]
        vr = validate_program(puzzles[pid], row["program"])
        assert vr.valid
        best[pid] = (vr.length, row["program"], json.loads(row["program"]))

    # Pick top_k highest-overhead puzzles (by steps per target)
    ranked = sorted(
        puzzles.values(),
        key=lambda p: best[p.id][0] / max(1, len(set(p.targets) - set(p.starting_values))),
        reverse=True,
    )[:top_k]

    print(
        f"Running full-cooling HyperWalk ({secs}s/puzzle) on top {len(ranked)} bottleneck puzzles across {os.cpu_count()} cores...",
        flush=True,
    )

    def run_one(idx_p: tuple[int, Puzzle]) -> tuple[str, list[list] | None]:
        idx, p = idx_p
        inp = to_pfmt_block(p, best[p.id][2])
        proc = subprocess.run(
            [
                str(hw_bin),
                "--time",
                str(secs),
                "--seed",
                str(seed + idx * 7919),
                "--T",
                "0.28",
                "--cap",
                "4",
                "--rb",
                "0.55",
                "--pair",
                "0.25",
                "--variants",
                str(var_pfmt_path),
                "--cross",
                "60000",
                "--verbose",
            ],
            input=inp,
            text=True,
            capture_output=True,
            check=True,
        )
        lines = proc.stdout.splitlines()
        i = 0
        best_cand: list[list] | None = None
        while i < len(lines):
            ln = lines[i].strip()
            if ln.startswith("R "):
                parts = ln.split()
                new_len = int(parts[3])
                cand = []
                for k in range(1, new_len + 1):
                    op_s, a_s, b_s, d_s = lines[i + k].split()
                    cand.append([op_s, int(a_s), int(b_s), int(d_s)])
                best_cand = cand
                i += new_len + 1
            else:
                i += 1
        return p.id, best_cand

    t0 = time.time()
    improved = 0
    saved = 0
    with ThreadPoolExecutor(os.cpu_count() or 8) as ex:
        for pid, cand in ex.map(run_one, enumerate(ranked)):
            if cand is not None:
                js = json.dumps(cand, separators=(",", ":"))
                vr = validate_program(puzzles[pid], js)
                if vr.valid and vr.length < best[pid][0]:
                    diff = best[pid][0] - vr.length
                    saved += diff
                    improved += 1
                    print(f"  [BOTTLENECK IMP] {pid}: {best[pid][0]} -> {vr.length} (-{diff})", flush=True)
                    best[pid] = (vr.length, js, cand)

    out_csv = ROOT / "submissions" / "frontier_best_latest.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "program"])
        for pid in puzzles:
            w.writerow([pid, best[pid][1]])

    print(
        f"Bottleneck HyperWalk finished in {time.time() - t0:.1f}s: improved {improved} (-{saved} steps) "
        f"-> Verified score = {sum(v[0] for v in best.values()):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
