use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
struct Inst {
    op: char,
    i: usize,
    j: usize,
    d: usize,
}

#[derive(Clone, Debug)]
struct Puzzle {
    id: String,
    starts: Vec<u64>,
    targets: Vec<u64>,
    value_limit: u64,
    registers: usize,
}

#[derive(Clone, Copy, Debug)]
struct DagNode {
    op: char,
    left_val: u64,
    right_val: u64,
}

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn parse_csv_row(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let bytes = line.as_bytes();
    let mut k = 0;
    while k < bytes.len() {
        let c = bytes[k] as char;
        if c == '"' {
            if in_quotes && k + 1 < bytes.len() && bytes[k + 1] == b'"' {
                cur.push('"');
                k += 1;
            } else {
                in_quotes = !in_quotes;
            }
        } else if c == ',' && !in_quotes {
            fields.push(std::mem::take(&mut cur));
        } else if c != '\r' {
            cur.push(c);
        }
        k += 1;
    }
    fields.push(cur);
    fields
}

fn parse_u64_array(s: &str) -> Vec<u64> {
    let mut res = Vec::new();
    let mut val: u64 = 0;
    let mut in_num = false;
    for b in s.bytes() {
        if b.is_ascii_digit() {
            val = val * 10 + (b - b'0') as u64;
            in_num = true;
        } else if in_num {
            res.push(val);
            val = 0;
            in_num = false;
        }
    }
    if in_num {
        res.push(val);
    }
    res
}

fn parse_program(s: &str) -> Option<Vec<Inst>> {
    let mut prog = Vec::new();
    let bytes = s.as_bytes();
    let mut k = 0;
    while k < bytes.len() {
        if bytes[k] == b'"' {
            k += 1;
            if k >= bytes.len() {
                return None;
            }
            let op = bytes[k] as char;
            while k < bytes.len() && bytes[k] != b',' {
                k += 1;
            }
            let mut nums = [0usize; 3];
            for slot in &mut nums {
                while k < bytes.len() && !bytes[k].is_ascii_digit() {
                    k += 1;
                }
                let mut v = 0usize;
                while k < bytes.len() && bytes[k].is_ascii_digit() {
                    v = v * 10 + (bytes[k] - b'0') as usize;
                    k += 1;
                }
                *slot = v;
            }
            prog.push(Inst {
                op,
                i: nums[0],
                j: nums[1],
                d: nums[2],
            });
        }
        k += 1;
    }
    Some(prog)
}

fn load_puzzles(path: &str) -> Vec<Puzzle> {
    let file = File::open(path).expect("Cannot open puzzles.csv");
    let reader = BufReader::new(file);
    let mut puzzles = Vec::new();
    for (idx, line_res) in reader.lines().enumerate() {
        let line = line_res.unwrap();
        if idx == 0 || line.trim().is_empty() {
            continue;
        }
        let cols = parse_csv_row(&line);
        if cols.len() < 5 {
            continue;
        }
        puzzles.push(Puzzle {
            id: cols[0].clone(),
            starts: parse_u64_array(&cols[1]),
            targets: parse_u64_array(&cols[2]),
            value_limit: cols[3].parse().unwrap(),
            registers: cols[4].parse().unwrap(),
        });
    }
    puzzles
}

fn load_seed_programs(path: &str) -> HashMap<String, Vec<Inst>> {
    let mut map = HashMap::new();
    let file = File::open(path).expect("Cannot open seed_submission.csv");
    let reader = BufReader::new(file);
    for (idx, line_res) in reader.lines().enumerate() {
        let line = line_res.unwrap();
        if idx == 0 || line.trim().is_empty() {
            continue;
        }
        let cols = parse_csv_row(&line);
        if cols.len() >= 2 {
            if let Some(prog) = parse_program(&cols[1]) {
                map.insert(cols[0].clone(), prog);
            }
        }
    }
    map
}

fn validate_program(p: &Puzzle, prog: &[Inst]) -> bool {
    if prog.len() > 4096 {
        return false;
    }
    let mut regs = vec![0u64; p.registers];
    let mut defined = vec![false; p.registers];
    let mut produced = HashSet::with_capacity(p.targets.len() * 2 + 32);
    for (idx, &v) in p.starts.iter().enumerate() {
        if idx >= p.registers {
            return false;
        }
        regs[idx] = v;
        defined[idx] = true;
        produced.insert(v);
    }
    for ins in prog {
        if ins.i >= p.registers || ins.j >= p.registers || ins.d >= p.registers {
            return false;
        }
        if !defined[ins.i] || !defined[ins.j] {
            return false;
        }
        let a = regs[ins.i] as u128;
        let b = regs[ins.j] as u128;
        let res = match ins.op {
            '+' => a + b,
            '-' => if a > b { a - b } else { return false; },
            '*' => a * b,
            '/' => if b != 0 && a % b == 0 { a / b } else { return false; },
            _ => return false,
        };
        if res == 0 || res > p.value_limit as u128 {
            return false;
        }
        let v = res as u64;
        regs[ins.d] = v;
        defined[ins.d] = true;
        produced.insert(v);
    }
    p.targets.iter().all(|t| produced.contains(t))
}

fn dce_prune(p: &Puzzle, prog: &[Inst]) -> Vec<Inst> {
    if prog.is_empty() {
        return Vec::new();
    }
    let mut regs = vec![0u64; p.registers];
    let target_set: HashSet<u64> = p.targets.iter().copied().collect();
    let mut seen_targets = HashSet::new();
    for (idx, &v) in p.starts.iter().enumerate() {
        regs[idx] = v;
        if target_set.contains(&v) {
            seen_targets.insert(v);
        }
    }
    let mut needed = vec![false; prog.len()];
    for (step, ins) in prog.iter().enumerate() {
        let a = regs[ins.i];
        let b = regs[ins.j];
        let res = match ins.op {
            '+' => a + b,
            '-' => a - b,
            '*' => a * b,
            '/' => a / b,
            _ => 0,
        };
        regs[ins.d] = res;
        if target_set.contains(&res) && !seen_targets.contains(&res) {
            seen_targets.insert(res);
            needed[step] = true;
        }
    }
    let mut live = vec![false; p.registers];
    for step in (0..prog.len()).rev() {
        let ins = prog[step];
        if needed[step] || live[ins.d] {
            needed[step] = true;
            live[ins.d] = false;
            live[ins.i] = true;
            live[ins.j] = true;
        }
    }
    prog.iter()
        .enumerate()
        .filter(|(s, _)| needed[*s])
        .map(|(_, &ins)| ins)
        .collect()
}

fn prog_to_dag(p: &Puzzle, prog: &[Inst]) -> HashMap<u64, DagNode> {
    let mut regs = vec![0u64; p.registers];
    for (i, &v) in p.starts.iter().enumerate() {
        regs[i] = v;
    }
    let starts_set: HashSet<u64> = p.starts.iter().copied().collect();
    let mut dag: HashMap<u64, DagNode> = HashMap::new();
    for ins in prog {
        let a = regs[ins.i];
        let b = regs[ins.j];
        let v = match ins.op {
            '+' => a + b,
            '-' => a - b,
            '*' => a * b,
            '/' => a / b,
            _ => 0,
        };
        regs[ins.d] = v;
        if !starts_set.contains(&v) {
            dag.entry(v).or_insert(DagNode {
                op: ins.op,
                left_val: a,
                right_val: b,
            });
        }
    }
    dag
}

fn prune_dag(p: &Puzzle, dag: &HashMap<u64, DagNode>) -> HashMap<u64, DagNode> {
    let starts_set: HashSet<u64> = p.starts.iter().copied().collect();
    let mut keep = HashMap::new();
    let mut stack: Vec<u64> = p.targets.clone();
    let mut visited: HashSet<u64> = HashSet::new();
    while let Some(v) = stack.pop() {
        if !visited.insert(v) || starts_set.contains(&v) {
            continue;
        }
        if let Some(&node) = dag.get(&v) {
            keep.insert(v, node);
            stack.push(node.left_val);
            stack.push(node.right_val);
        }
    }
    keep
}

fn has_cycle(dag: &HashMap<u64, DagNode>, starts_set: &HashSet<u64>) -> bool {
    let mut state: HashMap<u64, u8> = HashMap::with_capacity(dag.len());
    fn dfs(
        v: u64,
        dag: &HashMap<u64, DagNode>,
        starts_set: &HashSet<u64>,
        state: &mut HashMap<u64, u8>,
    ) -> bool {
        if starts_set.contains(&v) {
            return false;
        }
        match state.get(&v) {
            Some(&1) => return true,
            Some(&2) => return false,
            _ => {}
        }
        state.insert(v, 1);
        if let Some(n) = dag.get(&v) {
            if dfs(n.left_val, dag, starts_set, state) || dfs(n.right_val, dag, starts_set, state) {
                return true;
            }
        } else {
            return true; // missing node
        }
        state.insert(v, 2);
        false
    }
    for &k in dag.keys() {
        if dfs(k, dag, starts_set, &mut state) {
            return true;
        }
    }
    false
}

fn schedule_dag(p: &Puzzle, dag: &HashMap<u64, DagNode>) -> Option<Vec<Inst>> {
    let pruned = prune_dag(p, dag);
    let starts_set: HashSet<u64> = p.starts.iter().copied().collect();
    for &t in &p.targets {
        if !starts_set.contains(&t) && !pruned.contains_key(&t) {
            return None;
        }
    }
    if has_cycle(&pruned, &starts_set) {
        return None;
    }

    let mut remaining: HashSet<u64> = pruned.keys().copied().collect();
    let mut use_count: HashMap<u64, usize> = HashMap::new();
    for node in pruned.values() {
        *use_count.entry(node.left_val).or_insert(0) += 1;
        *use_count.entry(node.right_val).or_insert(0) += 1;
    }

    let mut regs = vec![0u64; p.registers];
    let mut defined = vec![false; p.registers];
    let mut val_to_reg: HashMap<u64, usize> = HashMap::new();
    for (i, &v) in p.starts.iter().enumerate() {
        regs[i] = v;
        defined[i] = true;
        val_to_reg.insert(v, i);
    }

    let mut prog = Vec::with_capacity(pruned.len());
    while !remaining.is_empty() {
        let mut ready: Vec<u64> = remaining
            .iter()
            .copied()
            .filter(|&v| {
                let n = pruned[&v];
                val_to_reg.contains_key(&n.left_val) && val_to_reg.contains_key(&n.right_val)
            })
            .collect();
        if ready.is_empty() {
            return None;
        }
        ready.sort_by_key(|&v| {
            let n = pruned[&v];
            let mut frees = 0i32;
            if use_count.get(&n.left_val).copied().unwrap_or(0) <= 1 && !starts_set.contains(&n.left_val) {
                frees += 1;
            }
            if n.right_val != n.left_val
                && use_count.get(&n.right_val).copied().unwrap_or(0) <= 1
                && !starts_set.contains(&n.right_val)
            {
                frees += 1;
            }
            let needed_later = use_count.get(&v).copied().unwrap_or(0) > 0;
            (-frees, needed_later as i32, v)
        });

        let v = ready[0];
        let n = pruned[&v];
        remaining.remove(&v);
        let ri = val_to_reg[&n.left_val];
        let rj = val_to_reg[&n.right_val];

        if let Some(c) = use_count.get_mut(&n.left_val) {
            *c = c.saturating_sub(1);
        }
        if n.right_val != n.left_val {
            if let Some(c) = use_count.get_mut(&n.right_val) {
                *c = c.saturating_sub(1);
            }
        }

        // Overwrite a dying operand in-place if possible, else free slot, else dead reg
        let mut dst = None;
        if use_count.get(&n.left_val).copied().unwrap_or(0) == 0 && !starts_set.contains(&n.left_val) {
            dst = Some(ri);
        } else if use_count.get(&n.right_val).copied().unwrap_or(0) == 0 && !starts_set.contains(&n.right_val) {
            dst = Some(rj);
        } else {
            for r in 0..p.registers {
                if !defined[r] {
                    dst = Some(r);
                    break;
                }
            }
            if dst.is_none() {
                for r in 0..p.registers {
                    let rv = regs[r];
                    if use_count.get(&rv).copied().unwrap_or(0) == 0 {
                        dst = Some(r);
                        break;
                    }
                }
            }
        }
        let d = dst?;
        if defined[d] {
            let old_v = regs[d];
            if val_to_reg.get(&old_v) == Some(&d) {
                val_to_reg.remove(&old_v);
            }
        }
        prog.push(Inst { op: n.op, i: ri, j: rj, d });
        regs[d] = v;
        defined[d] = true;
        val_to_reg.insert(v, d);
    }

    if validate_program(p, &prog) {
        Some(prog)
    } else {
        None
    }
}

fn build_p1(pool: &[u64], lim: u128) -> HashMap<u64, DagNode> {
    let mut p1: HashMap<u64, DagNode> = HashMap::with_capacity(pool.len() * pool.len());
    for i in 0..pool.len() {
        let a = pool[i] as u128;
        for j in i..pool.len() {
            let b = pool[j] as u128;
            let s = a + b;
            if s > 0 && s <= lim {
                p1.entry(s as u64).or_insert(DagNode { op: '+', left_val: a as u64, right_val: b as u64 });
            }
            let m = a * b;
            if m > 0 && m <= lim {
                p1.entry(m as u64).or_insert(DagNode { op: '*', left_val: a as u64, right_val: b as u64 });
            }
            if a > b {
                p1.entry((a - b) as u64).or_insert(DagNode { op: '-', left_val: a as u64, right_val: b as u64 });
            } else if b > a {
                p1.entry((b - a) as u64).or_insert(DagNode { op: '-', left_val: b as u64, right_val: a as u64 });
            }
            if b != 0 && a % b == 0 {
                p1.entry((a / b) as u64).or_insert(DagNode { op: '/', left_val: a as u64, right_val: b as u64 });
            }
            if a != 0 && b % a == 0 {
                p1.entry((b / a) as u64).or_insert(DagNode { op: '/', left_val: b as u64, right_val: a as u64 });
            }
        }
    }
    p1
}

// Private-Cone Value-Graph LNS with Plateau Walk & 1/2-step MITM Repair
fn lns_v2_optimize(p: &Puzzle, init_prog: &[Inst], seed: u64, max_iters: usize) -> Vec<Inst> {
    let mut best_prog = dce_prune(p, init_prog);
    let mut cur_dag = prog_to_dag(p, &best_prog);
    let starts_set: HashSet<u64> = p.starts.iter().copied().collect();
    let target_set: HashSet<u64> = p.targets.iter().copied().collect();
    let mut rng = Rng(seed.max(1));
    let lim = p.value_limit as u128;

    for _iter in 0..max_iters {
        let nodes: Vec<u64> = cur_dag.keys().copied().collect();
        if nodes.len() < 4 {
            break;
        }
        // Build consumer counts
        let mut consumers: HashMap<u64, Vec<u64>> = HashMap::with_capacity(cur_dag.len());
        for (&k, &n) in &cur_dag {
            consumers.entry(n.left_val).or_default().push(k);
            if n.right_val != n.left_val {
                consumers.entry(n.right_val).or_default().push(k);
            }
        }

        // Pick a root and destroy its private cone (up to max_cut nodes)
        let root = nodes[(rng.next_u64() as usize) % nodes.len()];
        let max_cut = 3 + ((rng.next_u64() as usize) % 8);
        let mut removed: HashSet<u64> = HashSet::new();
        let mut stack = vec![root];
        while let Some(v) = stack.pop() {
            if removed.len() >= max_cut {
                break;
            }
            if starts_set.contains(&v) || !cur_dag.contains_key(&v) || !removed.insert(v) {
                continue;
            }
            if let Some(n) = cur_dag.get(&v) {
                for &anc in &[n.left_val, n.right_val] {
                    if !starts_set.contains(&anc) && !removed.contains(&anc) {
                        let all_cons_removed = consumers
                            .get(&anc)
                            .map_or(true, |cs| cs.iter().all(|c| removed.contains(c)));
                        if all_cons_removed {
                            stack.push(anc);
                        }
                    }
                }
            }
        }

        if removed.len() < 2 {
            continue;
        }

        // Identify broken values U that must be repaired:
        // Any removed node that is a target OR is consumed by a kept node
        let mut broken: Vec<u64> = Vec::new();
        for &r in &removed {
            if target_set.contains(&r) {
                broken.push(r);
            } else if let Some(cs) = consumers.get(&r) {
                if cs.iter().any(|c| !removed.contains(c)) {
                    broken.push(r);
                }
            }
        }
        if broken.is_empty() || broken.len() > 3 {
            continue;
        }

        // Remaining pool D
        let mut pool_set: HashSet<u64> = starts_set.clone();
        for &k in cur_dag.keys() {
            if !removed.contains(&k) {
                pool_set.insert(k);
            }
        }
        let pool: Vec<u64> = pool_set.iter().copied().collect();
        let p1 = build_p1(&pool, lim);

        let mut cand_dag: HashMap<u64, DagNode> = cur_dag
            .iter()
            .filter(|(&k, _)| !removed.contains(&k))
            .map(|(&k, &v)| (k, v))
            .collect();

        let mut ok = true;
        let uphill = (rng.next_u64() % 50) == 0;
        let budget = removed.len() + if uphill { 1 } else { 0 };
        let mut added = 0usize;

        // Pre-sort a compact P1 key slice for MITM (P1 x P1)
        let mut p1_keys: Vec<u64> = p1.keys().copied().collect();
        p1_keys.sort_unstable();
        let mut p1_sample: Vec<u64> = Vec::with_capacity(300);
        let small_n = p1_keys.len().min(150);
        p1_sample.extend_from_slice(&p1_keys[..small_n]);
        if p1_keys.len() > small_n {
            let hi_start = (p1_keys.len() - p1_keys.len().min(150)).max(small_n);
            p1_sample.extend_from_slice(&p1_keys[hi_start..]);
        }

        for &u in &broken {
            if cand_dag.contains_key(&u) || starts_set.contains(&u) {
                continue;
            }
            // 1-step from pool
            if let Some(&n) = p1.get(&u) {
                cand_dag.insert(u, n);
                added += 1;
                continue;
            }
            // 2-step: u = x op d (x in p1, d in pool)
            let mut repaired = false;
            for &d in &pool {
                let cands = [
                    (u > d).then(|| (u - d, '+', false)),
                    ((u as u128) + (d as u128) <= lim).then(|| (u + d, '-', false)),
                    (d > u).then(|| (d - u, '-', true)),
                    (d > 1 && u % d == 0).then(|| (u / d, '*', false)),
                    ((u as u128) * (d as u128) <= lim).then(|| (u * d, '/', false)),
                ];
                for (need, op, d_first) in cands.into_iter().flatten() {
                    if let Some(&xn) = p1.get(&need) {
                        cand_dag.insert(need, xn);
                        let un = if d_first {
                            DagNode { op, left_val: d, right_val: need }
                        } else {
                            DagNode { op, left_val: need, right_val: d }
                        };
                        cand_dag.insert(u, un);
                        added += 2;
                        repaired = true;
                        break;
                    }
                }
                if repaired {
                    break;
                }
            }
            // 3-step MITM: u = x op y (x, y in p1)
            if !repaired && added + 3 <= budget {
                for &x in &p1_sample {
                    let cands = [
                        (u > x).then(|| (u - x, '+', false)),
                        ((u as u128) + (x as u128) <= lim).then(|| (u + x, '-', false)),
                        (x > u).then(|| (x - u, '-', true)),
                        (x > 1 && u % x == 0).then(|| (u / x, '*', false)),
                    ];
                    for (need, op, x_first) in cands.into_iter().flatten() {
                        if let Some(&yn) = p1.get(&need) {
                            if let Some(&xn) = p1.get(&x) {
                                cand_dag.insert(x, xn);
                                cand_dag.insert(need, yn);
                                let un = if x_first {
                                    DagNode { op, left_val: x, right_val: need }
                                } else {
                                    DagNode { op, left_val: need, right_val: x }
                                };
                                cand_dag.insert(u, un);
                                added += 3;
                                repaired = true;
                                break;
                            }
                        }
                    }
                    if repaired {
                        break;
                    }
                }
            }
            if !repaired {
                ok = false;
                break;
            }
        }

        if !ok || added > budget {
            continue;
        }

        cand_dag = prune_dag(p, &cand_dag);
        let max_accept = cur_dag.len() + if uphill { 1 } else { 0 };
        if cand_dag.len() <= max_accept {
            if let Some(sched) = schedule_dag(p, &cand_dag) {
                let pruned = dce_prune(p, &sched);
                if validate_program(p, &pruned) {
                    if pruned.len() < best_prog.len() {
                        best_prog = pruned.clone();
                    }
                    if pruned.len() <= max_accept {
                        cur_dag = prog_to_dag(p, &pruned);
                    }
                }
            }
        }
    }

    best_prog
}

fn serialize_prog(prog: &[Inst]) -> String {
    let mut s = String::with_capacity(prog.len() * 14 + 4);
    s.push('[');
    for (k, ins) in prog.iter().enumerate() {
        if k > 0 { s.push(','); }
        s.push_str(&format!("[\"{}\",{},{},{}]", ins.op, ins.i, ins.j, ins.d));
    }
    s.push(']');
    s
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <puzzles.csv> <seed.csv> <out.csv> [iters]", args[0]);
        std::process::exit(1);
    }
    let max_iters: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(150);
    let t0 = Instant::now();
    let puzzles = load_puzzles(&args[1]);
    let seed_map = load_seed_programs(&args[2]);

    let n = puzzles.len();
    let results: Vec<std::sync::Mutex<Option<Vec<Inst>>>> = (0..n).map(|_| std::sync::Mutex::new(None)).collect();
    let next_idx = AtomicUsize::new(0);
    let n_threads = thread::available_parallelism().map(|x| x.get()).unwrap_or(8);

    thread::scope(|s| {
        for _ in 0..n_threads {
            s.spawn(|| loop {
                let idx = next_idx.fetch_add(1, Ordering::Relaxed);
                if idx >= n { break; }
                let p = &puzzles[idx];
                let init = &seed_map[&p.id];
                let mut best = init.clone();
                let cand = lns_v2_optimize(p, &best, 314159 + (idx as u64) * 97, max_iters);
                if validate_program(p, &cand) && cand.len() < best.len() {
                    best = cand;
                }
                *results[idx].lock().unwrap() = Some(best);
            });
        }
    });

    let out_file = File::create(&args[3]).expect("Cannot create output CSV");
    let mut writer = BufWriter::new(out_file);
    writeln!(writer, "id,program").unwrap();
    let mut total: usize = 0;
    for (idx, p) in puzzles.iter().enumerate() {
        let prog = results[idx].lock().unwrap().take().unwrap();
        total += prog.len();
        let ser = serialize_prog(&prog);
        let escaped = ser.replace('"', "\"\"");
        writeln!(writer, "{},\"{}\"", p.id, escaped).unwrap();
    }
    writer.flush().unwrap();
    println!(
        "Rust LNS v2 ({} iters) finished {} puzzles in {:.2}s. Total score = {}",
        max_iters,
        n,
        t0.elapsed().as_secs_f64(),
        total
    );
}
