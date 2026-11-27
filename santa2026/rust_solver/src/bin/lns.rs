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
    val: u64,
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
            '-' => {
                if a <= b {
                    return false;
                }
                a - b
            }
            '*' => a * b,
            '/' => {
                if b == 0 || a % b != 0 {
                    return false;
                }
                a / b
            }
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

// Extract unique value-DAG nodes from a valid program (deduplicating identical values)
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
                val: v,
            });
        }
    }
    dag
}

// Prune DAG to only nodes reachable from targets
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

// Schedule a DAG into <= 48 registers if possible (greedy min-live-register topological sort)
fn schedule_dag(p: &Puzzle, dag: &HashMap<u64, DagNode>) -> Option<Vec<Inst>> {
    let pruned = prune_dag(p, dag);
    let starts_set: HashSet<u64> = p.starts.iter().copied().collect();
    for &t in &p.targets {
        if !starts_set.contains(&t) && !pruned.contains_key(&t) {
            return None;
        }
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
        // Find all ready nodes whose inputs are currently in registers
        let mut ready: Vec<u64> = remaining
            .iter()
            .copied()
            .filter(|&v| {
                let n = pruned[&v];
                val_to_reg.contains_key(&n.left_val) && val_to_reg.contains_key(&n.right_val)
            })
            .collect();

        if ready.is_empty() {
            return None; // Needs recomputation / spilling
        }

        // Prioritize nodes that free the most registers (inputs whose remaining use_count == 1)
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

        // Decrement use counts
        if let Some(c) = use_count.get_mut(&n.left_val) {
            *c = c.saturating_sub(1);
        }
        if n.right_val != n.left_val {
            if let Some(c) = use_count.get_mut(&n.right_val) {
                *c = c.saturating_sub(1);
            }
        }

        // Pick destination register: free slot, or a dead register (use_count == 0)
        let mut dst = None;
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
        let d = dst?;
        if defined[d] {
            let old_v = regs[d];
            if val_to_reg.get(&old_v) == Some(&d) {
                val_to_reg.remove(&old_v);
            }
        }
        prog.push(Inst {
            op: n.op,
            i: ri,
            j: rj,
            d,
        });
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

// Try to optimize a puzzle's DAG via LNS:
// 1. Replace any node `v` if `v` can be computed in 1 step from two earlier/other DAG values!
// 2. Replace a target's exclusive cone if a shorter 1-step or 2-step or q*d+r path exists from existing DAG values!
fn lns_optimize(p: &Puzzle, init_prog: &[Inst], seed: u64) -> Vec<Inst> {
    let mut best_prog = dce_prune(p, init_prog);
    let mut dag = prog_to_dag(p, &best_prog);
    let mut rng = Rng(seed.max(1));
    let lim = p.value_limit as u128;

    for _iter in 0..120 {
        let mut all_vals: Vec<u64> = p.starts.clone();
        all_vals.extend(dag.keys().copied());
        all_vals.sort_unstable();
        all_vals.dedup();
        let val_set: HashSet<u64> = all_vals.iter().copied().collect();

        // Pick a random non-start node
        let nodes: Vec<u64> = dag.keys().copied().collect();
        if nodes.is_empty() {
            break;
        }
        let target_v = nodes[(rng.next_u64() as usize) % nodes.len()];

        // Compute nodes that do NOT depend on target_v (safe pool to build target_v from!)
        let mut deps_of_target: HashSet<u64> = HashSet::new();
        deps_of_target.insert(target_v);
        let mut changed = true;
        while changed {
            changed = false;
            for (&k, &n) in &dag {
                if !deps_of_target.contains(&k)
                    && (deps_of_target.contains(&n.left_val) || deps_of_target.contains(&n.right_val))
                {
                    deps_of_target.insert(k);
                    changed = true;
                }
            }
        }
        let safe_vals: Vec<u64> = all_vals
            .iter()
            .copied()
            .filter(|v| !deps_of_target.contains(v))
            .collect();
        let safe_set: HashSet<u64> = safe_vals.iter().copied().collect();

        // 1. Check if target_v can be formed in 1 step from safe_vals!
        let mut new_nodes: Vec<DagNode> = Vec::new();
        for &a in &safe_vals {
            if target_v > a && safe_set.contains(&(target_v - a)) {
                new_nodes.push(DagNode { op: '+', left_val: a, right_val: target_v - a, val: target_v });
                break;
            }
            if (target_v as u128) + (a as u128) <= lim && safe_set.contains(&(target_v + a)) {
                new_nodes.push(DagNode { op: '-', left_val: target_v + a, right_val: a, val: target_v });
                break;
            }
            if a > 1 && target_v % a == 0 && safe_set.contains(&(target_v / a)) {
                new_nodes.push(DagNode { op: '*', left_val: a, right_val: target_v / a, val: target_v });
                break;
            }
            if a > 0 && (target_v as u128) * (a as u128) <= lim && safe_set.contains(&(target_v * a)) {
                new_nodes.push(DagNode { op: '/', left_val: target_v * a, right_val: a, val: target_v });
                break;
            }
        }

        // 2. If not 1-step, check 2-step replacement: target_v = (a op1 b) op2 c
        if new_nodes.is_empty() && safe_vals.len() <= 220 {
            let mut safe_r1: HashMap<u64, (char, u64, u64)> = HashMap::with_capacity(4096);
            for i in 0..safe_vals.len() {
                let a = safe_vals[i] as u128;
                for j in i..safe_vals.len() {
                    let b = safe_vals[j] as u128;
                    let s = a + b;
                    if s > 0 && s <= lim { safe_r1.entry(s as u64).or_insert(('+', a as u64, b as u64)); }
                    let m = a * b;
                    if m > 0 && m <= lim { safe_r1.entry(m as u64).or_insert(('*', a as u64, b as u64)); }
                    if a > b { safe_r1.entry((a - b) as u64).or_insert(('-', a as u64, b as u64)); }
                    else if b > a { safe_r1.entry((b - a) as u64).or_insert(('-', b as u64, a as u64)); }
                    if b != 0 && a % b == 0 { safe_r1.entry((a / b) as u64).or_insert(('/', a as u64, b as u64)); }
                    if a != 0 && b % a == 0 { safe_r1.entry((b / a) as u64).or_insert(('/', b as u64, a as u64)); }
                }
            }
            for &c in &safe_vals {
                let candidates = [
                    (target_v > c).then(|| (target_v - c, '+', false)),
                    ((target_v as u128) + (c as u128) <= lim).then(|| (target_v + c, '-', false)),
                    (c > target_v).then(|| (c - target_v, '-', true)),
                    (c > 1 && target_v % c == 0).then(|| (target_v / c, '*', false)),
                    ((target_v as u128) * (c as u128) <= lim).then(|| (target_v * c, '/', false)),
                ];
                let mut found = false;
                for cand in candidates.into_iter().flatten() {
                    let (need, op2, c_first) = cand;
                    if let Some(&(op1, x, y)) = safe_r1.get(&need) {
                        new_nodes.push(DagNode { op: op1, left_val: x, right_val: y, val: need });
                        if c_first {
                            new_nodes.push(DagNode { op: op2, left_val: c, right_val: need, val: target_v });
                        } else {
                            new_nodes.push(DagNode { op: op2, left_val: need, right_val: c, val: target_v });
                        }
                        found = true;
                        break;
                    }
                }
                if found {
                    break;
                }
            }
        }

        if !new_nodes.is_empty() {
            let mut cand_dag = dag.clone();
            for nn in new_nodes {
                cand_dag.insert(nn.val, nn);
            }
            cand_dag = prune_dag(p, &cand_dag);
            if cand_dag.len() < dag.len() {
                if let Some(sched) = schedule_dag(p, &cand_dag) {
                    let pruned = dce_prune(p, &sched);
                    if validate_program(p, &pruned) && pruned.len() < best_prog.len() {
                        best_prog = pruned;
                        dag = prog_to_dag(p, &best_prog);
                    }
                }
            }
        }
        let _ = val_set;
    }

    // Also try pure schedule_dag compaction on current dag
    if let Some(sched) = schedule_dag(p, &dag) {
        let pruned = dce_prune(p, &sched);
        if validate_program(p, &pruned) && pruned.len() < best_prog.len() {
            best_prog = pruned;
        }
    }

    best_prog
}

fn serialize_prog(prog: &[Inst]) -> String {
    let mut s = String::with_capacity(prog.len() * 14 + 4);
    s.push('[');
    for (k, ins) in prog.iter().enumerate() {
        if k > 0 {
            s.push(',');
        }
        s.push_str(&format!("[\"{}\",{},{},{}]", ins.op, ins.i, ins.j, ins.d));
    }
    s.push(']');
    s
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <puzzles.csv> <seed.csv> <out.csv>", args[0]);
        std::process::exit(1);
    }
    let t0 = Instant::now();
    let puzzles = load_puzzles(&args[1]);
    let seed_map = load_seed_programs(&args[2]);

    let n = puzzles.len();
    let results: Vec<std::sync::Mutex<Option<Vec<Inst>>>> =
        (0..n).map(|_| std::sync::Mutex::new(None)).collect();
    let next_idx = AtomicUsize::new(0);
    let n_threads = thread::available_parallelism().map(|x| x.get()).unwrap_or(8);

    thread::scope(|s| {
        for _ in 0..n_threads {
            s.spawn(|| loop {
                let idx = next_idx.fetch_add(1, Ordering::Relaxed);
                if idx >= n {
                    break;
                }
                let p = &puzzles[idx];
                let init = &seed_map[&p.id];
                let mut best = init.clone();
                for round in 0..4 {
                    let cand = lns_optimize(p, &best, 55555 + (idx as u64) * 31 + round);
                    if validate_program(p, &cand) && cand.len() < best.len() {
                        best = cand;
                    }
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
        "Rust LNS finished {} puzzles in {:.2}s. Total score = {}",
        n,
        t0.elapsed().as_secs_f64(),
        total
    );
}
