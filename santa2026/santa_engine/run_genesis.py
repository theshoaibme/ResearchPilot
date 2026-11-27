"""10-Core Genesis Backbone Runner: k=3..9 Hub-Cluster Subgraph Ruin + Bottom-Up Generator Reconstruction."""

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
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    lo_len = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    hi_len = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    max_puzzles = int(sys.argv[4]) if len(sys.argv) > 4 else 500
    base_seed = int(sys.argv[5]) if len(sys.argv) > 5 else 202688801

    gen_bin = ROOT / "santa_engine" / "genesis_backbone_bin"
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
    targets = targets[:max_puzzles]

    ncores = os.cpu_count() or 10
    print(
        f"[GenesisBackbone] Start score = {start_total:,} | Running {len(targets)} puzzles "
        f"({secs}s/puzzle, len in [{lo_len}, {hi_len}]) across {ncores} cores...",
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
                    str(gen_bin),
                    "--time",
                    str(secs),
                    "--width",
                    "88",
                    "--prodk",
                    "5",
                    "--scratch",
                    "1",
                    "--seed",
                    str(base_seed + idx * 12347),
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
                        f"  *** [GENESIS DIRECT WIN] {p.id}: {old_l} -> {vr.length} (-{diff}) "
                        f"| NEW SCORE = {cur_tot:,} ***",
                        flush=True,
                    )
            for vs in vsets:
                if len(vs) < best[p.id][0]:
                    unsched_seeds.setdefault(p.id, []).append(vs)

            if done_cnt % 25 == 0 or done_cnt == len(targets):
                cur_tot = sum(v[0] for v in best.values())
                print(
                    f"  [Progress {done_cnt}/{len(targets)} | {time.time() - t0:.1f}s] "
                    f"wins={improved} (-{saved}) | unsched={len(unsched_seeds)} | score={cur_tot:,}",
                    flush=True,
                )

    with ThreadPoolExecutor(ncores) as ex:
        list(ex.map(run_one, enumerate(targets)))

    active_unsched = {pid: [vs for vs in vss if len(vs) < best[pid][0]] for pid, vss in unsched_seeds.items()}
    active_unsched = {pid: vss for pid, vss in active_unsched.items() if vss}
    if active_unsched:
        print(
            f"[GenesisBackbone] Scheduling & collapsing {len(active_unsched)} smaller-closure puzzles via SuperWalk...",
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
                    str(base_seed + 99991),
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
                            f"  *** [GENESIS+SW WIN] {pid}: {old_l} -> {vr.length} (-{diff}) "
                            f"| NEW SCORE = {cur_tot:,} ***",
                            flush=True,
                        )

    end_total = sum(v[0] for v in best.values())
    print(
        f"[GenesisBackbone Done in {time.time() - t0:.1f}s] {start_total:,} -> {end_total:,} "
        f"(improved {improved} puzzles, -{saved} steps)",
        flush=True,
    )


if __name__ == "__main__":
    main()
