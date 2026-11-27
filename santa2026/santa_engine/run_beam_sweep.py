"""Fast 10-Core Full-Dataset Sweep: Beam-LNS v2 (1/2/3-Target Deep Ruin) + SuperWalk Scheduler Rescue."""

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
from concurrent.futures import ThreadPoolExecutor

csv.field_size_limit(1 << 28)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from validator import Puzzle, load_puzzles, validate_program  # noqa: E402
from cpp_engines.run_hyperwalk import to_pfmt_block  # noqa: E402
from santa_engine.run_superwalk import parse_superwalk_stdout, save_best  # noqa: E402


def main() -> None:
    lo_len = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    hi_len = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    width = int(sys.argv[3]) if len(sys.argv) > 3 else 112
    pairs = int(sys.argv[4]) if len(sys.argv) > 4 else 56
    triples = int(sys.argv[5]) if len(sys.argv) > 5 else 24
    base_seed = int(sys.argv[6]) if len(sys.argv) > 6 else 20269901

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
    targets = [p for p in puzzles.values() if lo_len <= best[p.id][0] <= hi_len]
    targets.sort(key=lambda p: best[p.id][0], reverse=True)

    ncores = os.cpu_count() or 10
    print(
        f"[BeamSweep] Start score = {start_total:,} | Sweeping {len(targets)} puzzles "
        f"(len in [{lo_len}, {hi_len}], W={width}, pairs={pairs}, triples={triples}) on {ncores} cores...",
        flush=True,
    )

    lock = threading.Lock()
    improved = 0
    saved = 0
    done_cnt = 0
    unsched_seeds: dict[str, list[list[int]]] = {}
    t0 = time.time()

    def run_one(idx_p: tuple[int, Puzzle]) -> None:
        nonlocal improved, saved, done_cnt
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
                    str(width),
                    "--prodk",
                    "9",
                    "--pairs",
                    str(pairs),
                    "--triples",
                    str(triples),
                    "--seed",
                    str(base_seed + idx * 10007),
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
            done_cnt += 1
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
                    cur_tot = sum(v[0] for v in best.values())
                    print(
                        f"  *** [BEAM DIRECT WIN] {p.id}: {old_l} -> {vr.length} (-{diff}) "
                        f"| NEW SCORE = {cur_tot:,} ***",
                        flush=True,
                    )
            for vs in vsets:
                if len(vs) < best[p.id][0]:
                    unsched_seeds.setdefault(p.id, []).append(vs)

            if done_cnt % 50 == 0 or done_cnt == len(targets):
                cur_tot = sum(v[0] for v in best.values())
                print(
                    f"  [Progress {done_cnt}/{len(targets)} | {time.time() - t0:.1f}s] "
                    f"direct_wins={improved} (-{saved}) | unsched_candidates={len(unsched_seeds)} | score={cur_tot:,}",
                    flush=True,
                )

    with ThreadPoolExecutor(ncores) as ex:
        list(ex.map(run_one, enumerate(targets)))

    # Rescue & collapse every unscheduled smaller closure via SuperWalk!
    active_unsched = {pid: [vs for vs in vss if len(vs) < best[pid][0]] for pid, vss in unsched_seeds.items()}
    active_unsched = {pid: vss for pid, vss in active_unsched.items() if vss}
    if active_unsched:
        print(
            f"[BeamSweep] Running SuperWalk 48-reg rescue & collapse on {len(active_unsched)} smaller-closure puzzles...",
            flush=True,
        )

        def rescue_one(pid_vss: tuple[str, list[list[int]]]) -> tuple[str, list[list] | None]:
            pid, vss = pid_vss
            p = puzzles[pid]
            inp = to_pfmt_block(p, best[pid][2])
            with tempfile.TemporaryDirectory() as td:
                sf = pathlib.Path(td) / "seeds.txt"
                with sf.open("w", encoding="utf-8") as f:
                    for vs in vss:
                        f.write(
                            f"V {pid} {best[pid][0]} {len(vs)} {len(vs) - best[pid][0]} {len(vs)} "
                            + " ".join(map(str, vs))
                            + "\n"
                        )
                cmd = [
                    str(sw_bin),
                    "--time",
                    "22.0",
                    "--seed",
                    str(base_seed + 77777),
                    "--T",
                    "0.26",
                    "--cap",
                    "4",
                    "--seed-sets",
                    str(sf),
                ]
                if var_pfmt.exists():
                    cmd.extend(["--variants", str(var_pfmt)])
                proc = subprocess.run(cmd, input=inp, text=True, capture_output=True, check=True)
                return pid, parse_superwalk_stdout(proc.stdout)

        with ThreadPoolExecutor(ncores) as ex:
            for pid, cand in ex.map(rescue_one, list(active_unsched.items())):
                if cand is not None:
                    js = json.dumps(cand, separators=(",", ":"))
                    vr = validate_program(puzzles[pid], js)
                    if vr.valid and vr.length < best[pid][0]:
                        diff = best[pid][0] - vr.length
                        old_l = best[pid][0]
                        best[pid] = (vr.length, js, cand)
                        improved += 1
                        saved += diff
                        save_best(best_csv, puzzles, best)
                        cur_tot = sum(v[0] for v in best.values())
                        print(
                            f"  *** [BEAM+SW RESCUE WIN] {pid}: {old_l} -> {vr.length} (-{diff}) "
                            f"| NEW SCORE = {cur_tot:,} ***",
                            flush=True,
                        )

    end_total = sum(v[0] for v in best.values())
    print(
        f"[BeamSweep Done in {time.time() - t0:.1f}s] {start_total:,} -> {end_total:,} "
        f"(improved {improved} puzzles, -{saved} steps)",
        flush=True,
    )


if __name__ == "__main__":
    main()
