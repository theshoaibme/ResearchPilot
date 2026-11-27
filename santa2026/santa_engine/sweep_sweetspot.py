"""Targeted Outlier & Sweet-Spot Assault (50-190 instr, Wide/30-52 bit puzzles where 80% of recoverable steps hide).

1. Ranks all puzzles by peer-group excess instructions/target within (wide_flag, bit_bucket).
2. Wave 1: Upgraded Beam-LNS v3 (380k S2 dictionary + Zero-Recharge Horner Reuse + 1/2/3-target ruin) on top 350 outliers.
3. Wave 2: High-Barrier SuperWalk (T=0.48, cap=8..10, S2-remainder + S1-Pollard-Rho + 2-target bridge) on top 150 sweet-spot outliers.
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

csv.field_size_limit(1 << 28)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from validator import Puzzle, load_puzzles, validate_program  # noqa: E402
from cpp_engines.run_hyperwalk import to_pfmt_block  # noqa: E402
from santa_engine.run_superwalk import parse_superwalk_stdout, save_best  # noqa: E402


def rank_outliers(puzzles: dict[str, Puzzle], best: dict[str, tuple[int, str, list[list]]], lo: int, hi: int) -> list[tuple[float, Puzzle]]:
    buckets: dict[tuple[bool, int], list[tuple[float, Puzzle]]] = defaultdict(list)
    for pid, p in puzzles.items():
        L = best[pid][0]
        if not (lo <= L <= hi):
            continue
        starts = set(p.starting_values)
        tgs = [t for t in p.targets if t not in starts]
        if not tgs:
            continue
        max_b = max(t.bit_length() for t in tgs)
        min_b = min(t.bit_length() for t in tgs)
        wide = (max_b - min_b) > 4
        b_key = (wide, max_b // 5)
        ipt = L / len(tgs)
        buckets[b_key].append((ipt, p))

    scored: list[tuple[float, Puzzle]] = []
    for (wide, b_bin), items in buckets.items():
        ipts = sorted(x[0] for x in items)
        floor_est = ipts[max(0, len(ipts) // 6)]
        for ipt, p in items:
            L = best[p.id][0]
            # Extra priority for wide 50..155 puzzles (empirically 73% of recoverable frontier steps!)
            sweet_bonus = 0.35 if (wide and 50 <= L <= 155) else (0.15 if 50 <= L <= 185 else 0.0)
            excess = (ipt - floor_est) + sweet_bonus
            scored.append((excess, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def main() -> None:
    beam_n = int(sys.argv[1]) if len(sys.argv) > 1 else 360
    sw_n = int(sys.argv[2]) if len(sys.argv) > 2 else 140
    sw_secs = float(sys.argv[3]) if len(sys.argv) > 3 else 18.0
    base_seed = int(sys.argv[4]) if len(sys.argv) > 4 else 20264815

    beam_bin = ROOT / "santa_engine" / "beam_lns_bin"
    sw_bin = ROOT / "santa_engine" / "superwalk_bin"
    var_pfmt = ROOT / "cpp_engines" / "variants.pfmt"
    puzzles = load_puzzles(ROOT / "data" / "puzzles.csv")
    best_csv = ROOT / "submissions" / "santa_best.csv"

    best: dict[str, tuple[int, str, list[list]]] = {}
    for row in csv.DictReader(best_csv.open("r", encoding="utf-8")):
        pid = row["id"]
        vr = validate_program(puzzles[pid], row["program"])
        assert vr.valid
        best[pid] = (vr.length, row["program"], json.loads(row["program"]))

    start_total = sum(v[0] for v in best.values())
    ncores = os.cpu_count() or 10
    ranked = rank_outliers(puzzles, best, 42, 235)
    print(f"[SweetSpot] Start score = {start_total:,} | Ranked {len(ranked)} candidate puzzles on {ncores} cores", flush=True)
    print(
        "  Top 10 peer-excess outliers: "
        + ", ".join(f"{p.id}(len={best[p.id][0]},ex={ex:.2f})" for ex, p in ranked[:10]),
        flush=True,
    )

    lock = threading.Lock()
    improved = 0
    saved = 0
    t0 = time.time()

    # ==================== WAVE 1: BEAM-LNS v3 (380k S2 + HORNER REUSE) ====================
    wave1 = [p for _, p in ranked[:beam_n]]
    print(f"\n[Wave 1] Running S2+Horner Beam-LNS v3 across top {len(wave1)} outliers...", flush=True)
    unsched: dict[str, list[list[int]]] = {}
    done1 = 0

    def run_beam(idx_p: tuple[int, Puzzle]) -> None:
        nonlocal improved, saved, done1
        idx, p = idx_p
        with lock:
            cur_prog = best[p.id][2]
        inp = to_pfmt_block(p, cur_prog)
        with tempfile.TemporaryDirectory() as td:
            s_out = pathlib.Path(td) / "seeds.txt"
            proc = subprocess.run(
                [
                    str(beam_bin),
                    "--width",
                    "112",
                    "--prodk",
                    "6",
                    "--pairs",
                    "56",
                    "--triples",
                    "24",
                    "--seed",
                    str(base_seed + idx * 7919),
                    "--seed-out",
                    str(s_out),
                ],
                input=inp,
                text=True,
                capture_output=True,
                check=True,
            )
            cand = parse_superwalk_stdout(proc.stdout)
            vsets: list[list[int]] = []
            if s_out.exists():
                for ln in s_out.read_text().splitlines():
                    pts = ln.split()
                    if len(pts) >= 6 and pts[0] == "V":
                        vsets.append([int(x) for x in pts[6:]])

        with lock:
            done1 += 1
            if cand is not None:
                js = json.dumps(cand, separators=(",", ":"))
                vr = validate_program(p, js)
                if vr.valid and vr.length < best[p.id][0]:
                    diff = best[p.id][0] - vr.length
                    old_l = best[p.id][0]
                    best[p.id] = (vr.length, js, cand)
                    improved += 1
                    saved += diff
                    save_best(best_csv, puzzles, best)
                    print(
                        f"  *** [WAVE1 BEAM WIN] {p.id}: {old_l} -> {vr.length} (-{diff}) "
                        f"| SCORE = {sum(v[0] for v in best.values()):,} ***",
                        flush=True,
                    )
            for vs in vsets:
                if len(vs) < best[p.id][0]:
                    unsched.setdefault(p.id, []).append(vs)
            if done1 % 60 == 0 or done1 == len(wave1):
                print(
                    f"  [Wave 1 {done1}/{len(wave1)} | {time.time() - t0:.1f}s] "
                    f"wins={improved} (-{saved}) unsched={len(unsched)} score={sum(v[0] for v in best.values()):,}",
                    flush=True,
                )

    with ThreadPoolExecutor(ncores) as ex:
        list(ex.map(run_beam, enumerate(wave1)))

    # ==================== WAVE 2: HIGH-BARRIER SUPERWALK (T=0.48, cap=8..10) ON 50-175 SWEET SPOT ====================
    ranked2 = rank_outliers(puzzles, best, 48, 175)
    wave2 = [p for _, p in ranked2[:sw_n]]
    # Ensure any unsched candidates from Wave 1 are also included at the front!
    for pid in list(unsched.keys()):
        if puzzles[pid] not in wave2:
            wave2.insert(0, puzzles[pid])

    print(
        f"\n[Wave 2] Running High-Barrier SuperWalk ({sw_secs}s/puzzle, T=0.48, cap=8..10) on {len(wave2)} sweet-spot puzzles...",
        flush=True,
    )
    done2 = 0

    def run_sw(idx_p: tuple[int, Puzzle]) -> None:
        nonlocal improved, saved, done2
        idx, p = idx_p
        with lock:
            cur_prog = best[p.id][2]
            seed_vss = [vs for vs in unsched.get(p.id, []) if len(vs) < best[p.id][0]]
        inp = to_pfmt_block(p, cur_prog)
        # Alternate high-barrier settings (inspired by 90s trajectory climbs of +8..+11 on 50-150 puzzles!)
        temp = "0.50" if idx % 2 == 0 else "0.38"
        cap = "9" if idx % 2 == 0 else "7"
        with tempfile.TemporaryDirectory() as td:
            cmd = [
                str(sw_bin),
                "--time",
                str(sw_secs),
                "--seed",
                str(base_seed + 500000 + idx * 104729),
                "--T",
                temp,
                "--cap",
                cap,
                "--rb",
                "0.52",
                "--pair",
                "0.28",
                "--cross",
                "35000",
            ]
            if var_pfmt.exists():
                cmd.extend(["--variants", str(var_pfmt)])
            if seed_vss:
                sf = pathlib.Path(td) / "seeds.txt"
                with sf.open("w", encoding="utf-8") as f:
                    for vs in seed_vss:
                        f.write(
                            f"V {p.id} {best[p.id][0]} {len(vs)} {len(vs) - best[p.id][0]} {len(vs)} "
                            + " ".join(map(str, vs))
                            + "\n"
                        )
                cmd.extend(["--seed-sets", str(sf)])

            proc = subprocess.run(cmd, input=inp, text=True, capture_output=True, check=True)
            cand = parse_superwalk_stdout(proc.stdout)

        with lock:
            done2 += 1
            if cand is not None:
                js = json.dumps(cand, separators=(",", ":"))
                vr = validate_program(p, js)
                if vr.valid and vr.length < best[p.id][0]:
                    diff = best[p.id][0] - vr.length
                    old_l = best[p.id][0]
                    best[p.id] = (vr.length, js, cand)
                    improved += 1
                    saved += diff
                    save_best(best_csv, puzzles, best)
                    print(
                        f"  *** [WAVE2 SUPERWALK WIN] {p.id}: {old_l} -> {vr.length} (-{diff}) "
                        f"| SCORE = {sum(v[0] for v in best.values()):,} ***",
                        flush=True,
                    )
            if done2 % 20 == 0 or done2 == len(wave2):
                print(
                    f"  [Wave 2 {done2}/{len(wave2)} | {time.time() - t0:.1f}s] "
                    f"total_wins={improved} (-{saved}) score={sum(v[0] for v in best.values()):,}",
                    flush=True,
                )

    with ThreadPoolExecutor(ncores) as ex:
        list(ex.map(run_sw, enumerate(wave2)))

    end_total = sum(v[0] for v in best.values())
    print(
        f"\n[SweetSpot Done in {time.time() - t0:.1f}s] {start_total:,} -> {end_total:,} "
        f"(improved {improved} puzzles, -{saved} steps)",
        flush=True,
    )


if __name__ == "__main__":
    main()
