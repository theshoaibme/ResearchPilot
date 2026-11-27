"""Unified 10-core Santa Engine: Alternative-Basin HyperWalk + Joint-Cone Frontier Breaker."""

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
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0
    max_puzzles = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 2026999

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

    all_csvs = sorted((ROOT / "public_frontier").rglob("*.csv"))
    maps = [{r["id"]: r["program"] for r in csv.DictReader(p.open("r", encoding="utf-8"))} for p in all_csvs]

    # Select puzzles that have a genuinely distinct alternative lineage (overlap < 88%, len <= best + 5)
    jobs: list[tuple[Puzzle, list[list]]] = []
    for pid, p in puzzles.items():
        b_len, _, b_prog = best[pid]
        b_vs = extract_vset(p, b_prog)
        best_alt: tuple[float, list[list]] | None = None
        for m in maps:
            if pid not in m:
                continue
            vr = validate_program(p, m[pid])
            if not vr.valid or vr.length > b_len + 5:
                continue
            prog = json.loads(m[pid])
            vs = extract_vset(p, prog)
            if vs == b_vs:
                continue
            overlap = len(vs & b_vs) / max(1, len(b_vs))
            if overlap < 0.88:
                if best_alt is None or overlap < best_alt[0]:
                    best_alt = (overlap, prog)
        if best_alt is not None:
            jobs.append((p, best_alt[1]))

    # Prioritize longest puzzles first
    jobs.sort(key=lambda item: best[item[0].id][0], reverse=True)
    jobs = jobs[:max_puzzles]
    print(f"SantaEngine: annealing {len(jobs)} distinct alternative-basin puzzles ({secs}s each on {os.cpu_count()} cores)...", flush=True)

    def run_one(idx_job: tuple[int, tuple[Puzzle, list[list]]]) -> tuple[str, list[list] | None]:
        idx, (p, alt_prog) = idx_job
        inp = to_pfmt_block(p, alt_prog)
        proc = subprocess.run(
            [
                str(hw_bin),
                "--time",
                str(secs),
                "--seed",
                str(seed + idx * 10007),
                "--T",
                "0.35",
                "--cap",
                "5",
                "--rb",
                "0.65",
                "--pair",
                "0.25",
                "--variants",
                str(var_pfmt),
                "--cross",
                "40000",
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
        for pid, cand in ex.map(run_one, enumerate(jobs)):
            if cand is not None:
                js = json.dumps(cand, separators=(",", ":"))
                vr = validate_program(puzzles[pid], js)
                if vr.valid and vr.length < best[pid][0]:
                    diff = best[pid][0] - vr.length
                    saved += diff
                    improved += 1
                    print(f"  [ENGINE IMP] {pid}: {best[pid][0]} -> {vr.length} (-{diff})", flush=True)
                    best[pid] = (vr.length, js, cand)

    out_csv = ROOT / "submissions" / "santa_best_engine.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "program"])
        for pid in puzzles:
            w.writerow([pid, best[pid][1]])

    if improved > 0:
        with best_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "program"])
            for pid in puzzles:
                w.writerow([pid, best[pid][1]])

    print(
        f"SantaEngine finished in {time.time() - t0:.1f}s: improved {improved} (-{saved} steps) "
        f"-> Verified score = {sum(v[0] for v in best.values()):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
