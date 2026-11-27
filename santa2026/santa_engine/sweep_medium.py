"""High-temperature deep HyperWalk sweep on [80, 210) puzzles to break 127,240."""

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
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    lo_len = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    hi_len = int(sys.argv[3]) if len(sys.argv) > 3 else 210
    max_n = int(sys.argv[4]) if len(sys.argv) > 4 else 150
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 2026888

    hw_bin = ROOT / "cpp_engines" / "hyperwalk_bin"
    var_pfmt = ROOT / "cpp_engines" / "variants.pfmt"
    puzzles = load_puzzles(ROOT / "data" / "puzzles.csv")

    best_csv = ROOT / "submissions" / "santa_best.csv"
    best: dict[str, tuple[int, str, list[list]]] = {}
    for row in csv.DictReader(best_csv.open("r", encoding="utf-8")):
        pid = row["id"]
        vr = validate_program(puzzles[pid], row["program"])
        assert vr.valid
        best[pid] = (vr.length, row["program"], json.loads(row["program"]))

    targets = [
        p for p in puzzles.values() if lo_len <= best[p.id][0] < hi_len
    ]
    targets.sort(key=lambda p: best[p.id][0], reverse=True)
    targets = targets[:max_n]
    print(f"Sweeping {len(targets)} puzzles in [{lo_len}, {hi_len}) at {secs}s/puzzle...", flush=True)

    def run_one(idx_p: tuple[int, Puzzle]) -> tuple[str, list[list] | None]:
        idx, p = idx_p
        inp = to_pfmt_block(p, best[p.id][2])
        proc = subprocess.run(
            [
                str(hw_bin),
                "--time",
                str(secs),
                "--seed",
                str(seed + idx * 104729),
                "--T",
                "0.45",
                "--cap",
                "6",
                "--rb",
                "0.80",
                "--pair",
                "0.30",
                "--variants",
                str(var_pfmt),
                "--cross",
                "25000",
            ],
            input=inp,
            text=True,
            capture_output=True,
            check=True,
        )
        lines = proc.stdout.splitlines()
        i = 0
        best_cand = None
        while i < len(lines):
            ln = lines[i].strip()
            if ln.startswith("R "):
                new_len = int(ln.split()[3])
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
        for pid, cand in ex.map(run_one, enumerate(targets)):
            if cand is not None:
                js = json.dumps(cand, separators=(",", ":"))
                vr = validate_program(puzzles[pid], js)
                if vr.valid and vr.length < best[pid][0]:
                    diff = best[pid][0] - vr.length
                    saved += diff
                    improved += 1
                    print(f"  [SWEEP IMP] {pid}: {best[pid][0]} -> {vr.length} (-{diff})", flush=True)
                    best[pid] = (vr.length, js, cand)

    if improved > 0:
        with best_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "program"])
            for pid in puzzles:
                w.writerow([pid, best[pid][1]])

    print(
        f"Medium sweep finished in {time.time() - t0:.1f}s: improved {improved} (-{saved} steps) "
        f"-> Score = {sum(v[0] for v in best.values()):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
