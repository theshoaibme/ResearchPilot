"""Multi-core runner for HYPERWALK (Triwalk + Pollard-Rho + 2-Sink Hub + Multi-Bank Crossover)."""

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


def to_pfmt_block(p: Puzzle, prog: list[list]) -> str:
    hdr = " ".join(
        map(
            str,
            [
                "P",
                p.id,
                len(p.starting_values),
                *p.starting_values,
                len(p.targets),
                *p.targets,
                p.value_limit,
                p.registers,
                len(prog),
            ],
        )
    )
    body = "\n".join(f"{op} {i} {j} {d}" for op, i, j, d in prog)
    return hdr + "\n" + body + ("\n" if body else "")


def extract_vset(p: Puzzle, prog: list[list]) -> frozenset[int]:
    regs: list[int | None] = [None] * p.registers
    for idx, v in enumerate(p.starting_values):
        regs[idx] = v
    out: set[int] = set()
    for op, i, j, d in prog:
        a, b = regs[i], regs[j]
        assert a is not None and b is not None
        r = a + b if op == "+" else a - b if op == "-" else a * b if op == "*" else a // b
        regs[d] = r
        out.add(r)
    return frozenset(out)


def main() -> None:
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20261128

    eng_dir = pathlib.Path(__file__).resolve().parent
    hw_cpp = ROOT / "public_frontier" / "hyperwalk" / "hyperwalk.cpp"
    hw_bin = eng_dir / "hyperwalk_bin"

    subprocess.run(
        ["clang++", "-std=c++20", "-O3", str(hw_cpp), "-o", str(hw_bin)],
        check=True,
    )

    puzzles = load_puzzles(ROOT / "data" / "puzzles.csv")
    all_csvs = [
        p
        for p in sorted((ROOT / "public_frontier").rglob("*.csv"))
        + [
            ROOT / "submissions" / "frontier_best.csv",
            ROOT / "submissions" / "frontier_milp.csv",
            ROOT / "submissions" / "hyperwalk_best.csv",
            ROOT / "submissions" / "rust_best_v4.csv",
        ]
        if p.exists()
    ]

    merged_csv = ROOT / "submissions" / "hyperwalk_seed.csv"
    base_score = merge_best_submissions(ROOT / "data" / "puzzles.csv", all_csvs, merged_csv)
    print(f"Merged seed score across {len(all_csvs)} banks: {base_score:,}", flush=True)

    best: dict[str, tuple[int, str, list[list]]] = {}
    for row in csv.DictReader(merged_csv.open("r", encoding="utf-8")):
        pid = row["id"]
        vr = validate_program(puzzles[pid], row["program"])
        assert vr.valid
        best[pid] = (vr.length, row["program"], json.loads(row["program"]))

    # Build deduplicated multi-bank variants.pfmt across all CSVs
    var_pfmt_path = eng_dir / "variants.pfmt"
    n_variants = 0
    has_variant: set[str] = set()
    maps = [{r["id"]: r["program"] for r in csv.DictReader(p.open("r", encoding="utf-8"))} for p in all_csvs]
    with var_pfmt_path.open("w", encoding="utf-8") as vf:
        for pid, p in puzzles.items():
            base_vs = extract_vset(p, best[pid][2])
            seen = {base_vs}
            for m in maps:
                if pid not in m:
                    continue
                vr = validate_program(p, m[pid])
                if not vr.valid or vr.length > best[pid][0] + 10:
                    continue
                prog = json.loads(m[pid])
                vs = extract_vset(p, prog)
                if vs not in seen:
                    seen.add(vs)
                    vf.write(to_pfmt_block(p, prog))
                    has_variant.add(pid)
                    n_variants += 1

    print(
        f"Wrote {n_variants} distinct multi-bank variants across {len(has_variant)} puzzles to {var_pfmt_path}",
        flush=True,
    )

    only_var = len(sys.argv) > 3 and sys.argv[3] == "variants"
    if only_var:
        active = [puzzles[pid] for pid in sorted(has_variant)]
    else:
        active = [p for p in puzzles.values() if best[p.id][0] >= 30]
    print(f"Running HyperWalk ({secs}s/puzzle) on {len(active)} target puzzles...", flush=True)
    chunks = [active[i : i + 3] for i in range(0, len(active), 3)]

    def run_chunk(c_info: tuple[int, list[Puzzle]]) -> list[tuple[str, list[list]]]:
        c_idx, chunk = c_info
        inp = "".join(to_pfmt_block(p, best[p.id][2]) for p in chunk)
        proc = subprocess.run(
            [
                str(hw_bin),
                "--time",
                str(secs),
                "--seed",
                str(seed + c_idx * 991),
                "--variants",
                str(var_pfmt_path),
                "--cross",
                "40000",
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
                    print(f"  [HYPERWALK IMP] {pid}: {best[pid][0]} -> {vr.length} (-{diff})", flush=True)
                    best[pid] = (vr.length, js, cand)

    out_csv = ROOT / "submissions" / "hyperwalk_best.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "program"])
        for pid in puzzles:
            w.writerow([pid, best[pid][1]])

    print(
        f"HyperWalk finished in {time.time() - t0:.1f}s: improved {improved} puzzles (-{saved} steps) "
        f"-> Final verified score = {sum(v[0] for v in best.values()):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
