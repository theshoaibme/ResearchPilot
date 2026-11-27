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

#[derive(Clone, Copy)]
struct Recipe {
    op: char,
    r1: usize,
    r2: usize,
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

#[derive(Clone)]
struct BeamState<'a> {
    p: &'a Puzzle,
    regs: Vec<u64>,
    defined: Vec<bool>,
    use_age: Vec<i32>,
    pin_count: Vec<i32>,
    produced: HashSet<u64>,
    val_to_reg: HashMap<u64, usize>,
    prog: Vec<Inst>,
    step_counter: i32,
    one_reg: Option<usize>,
    cached_r1: HashMap<u64, Recipe>,
    r1_dirty: bool,
    target_set: HashSet<u64>,
}

impl<'a> BeamState<'a> {
    fn new(p: &'a Puzzle) -> Self {
        let mut regs = vec![0u64; p.registers];
        let mut defined = vec![false; p.registers];
        let mut use_age = vec![0i32; p.registers];
        let mut produced = HashSet::with_capacity(64);
        let mut val_to_reg = HashMap::with_capacity(64);
        let mut step_counter = 0;
        let mut one_reg = None;
        for (i, &v) in p.starts.iter().enumerate() {
            regs[i] = v;
            defined[i] = true;
            step_counter += 1;
            use_age[i] = step_counter;
            produced.insert(v);
            val_to_reg.insert(v, i);
            if v == 1 {
                one_reg = Some(i);
            }
        }
        Self {
            p,
            regs,
            defined,
            use_age,
            pin_count: vec![0; p.registers],
            produced,
            val_to_reg,
            prog: Vec::with_capacity(256),
            step_counter,
            one_reg,
            cached_r1: HashMap::with_capacity(4096),
            r1_dirty: true,
            target_set: p.targets.iter().copied().collect(),
        }
    }

    fn alloc_reg(&self, avoid1: Option<usize>, avoid2: Option<usize>) -> usize {
        for r in 0..self.p.registers {
            if !self.defined[r] {
                return r;
            }
        }
        let mut best_r = None;
        let mut best_score = i64::MAX;
        for r in 0..self.p.registers {
            if Some(r) == self.one_reg
                || Some(r) == avoid1
                || Some(r) == avoid2
                || self.pin_count[r] > 0
            {
                continue;
            }
            let mut score = self.use_age[r] as i64;
            if self.regs[r] <= 64 {
                score += 200_000;
            }
            if r < self.p.starts.len() {
                score += 80_000;
            }
            if self.target_set.contains(&self.regs[r]) {
                for &t in &self.p.targets {
                    if !self.produced.contains(&t) {
                        let diff = t.abs_diff(self.regs[r]);
                        if (diff as u128) * 10 <= (t as u128) {
                            score += 180_000;
                            break;
                        }
                    }
                }
            }
            if score < best_score {
                best_score = score;
                best_r = Some(r);
            }
        }
        best_r.unwrap_or(0)
    }

    fn emit(&mut self, op: char, i: usize, j: usize) -> Option<usize> {
        if self.prog.len() >= 4096 || !self.defined[i] || !self.defined[j] {
            return None;
        }
        let a = self.regs[i] as u128;
        let b = self.regs[j] as u128;
        let res = match op {
            '+' => a + b,
            '-' => if a > b { a - b } else { return None; },
            '*' => a * b,
            '/' => if b != 0 && a % b == 0 { a / b } else { return None; },
            _ => return None,
        };
        if res == 0 || res > self.p.value_limit as u128 {
            return None;
        }
        let v = res as u64;
        if let Some(&ex) = self.val_to_reg.get(&v) {
            if self.defined[ex] && self.regs[ex] == v {
                self.step_counter += 1;
                self.use_age[ex] = self.step_counter;
                return Some(ex);
            }
        }
        let dst = self.alloc_reg(Some(i), Some(j));
        if self.defined[dst] {
            let old_v = self.regs[dst];
            if self.val_to_reg.get(&old_v) == Some(&dst) {
                self.val_to_reg.remove(&old_v);
            }
        }
        self.prog.push(Inst { op, i, j, d: dst });
        self.regs[dst] = v;
        self.defined[dst] = true;
        self.step_counter += 1;
        self.use_age[dst] = self.step_counter;
        self.produced.insert(v);
        self.val_to_reg.insert(v, dst);
        self.r1_dirty = true;
        if v == 1 {
            self.one_reg = Some(dst);
        }
        Some(dst)
    }

    fn ensure_one(&mut self) -> Option<usize> {
        if let Some(r) = self.one_reg {
            if self.defined[r] && self.regs[r] == 1 {
                return Some(r);
            }
        }
        for r in 0..self.p.registers {
            if self.defined[r] {
                self.one_reg = self.emit('/', r, r);
                return self.one_reg;
            }
        }
        None
    }

    fn rebuild_r1(&mut self) {
        if !self.r1_dirty {
            return;
        }
        self.cached_r1.clear();
        let active: Vec<usize> = (0..self.p.registers).filter(|&r| self.defined[r]).collect();
        let lim = self.p.value_limit as u128;
        for idx1 in 0..active.len() {
            let i = active[idx1];
            let a = self.regs[i] as u128;
            for idx2 in idx1..active.len() {
                let j = active[idx2];
                let b = self.regs[j] as u128;
                let s = a + b;
                if s > 0 && s <= lim {
                    self.cached_r1.entry(s as u64).or_insert(Recipe { op: '+', r1: i, r2: j });
                }
                let m = a * b;
                if m > 0 && m <= lim {
                    self.cached_r1.entry(m as u64).or_insert(Recipe { op: '*', r1: i, r2: j });
                }
                if a > b {
                    self.cached_r1.entry((a - b) as u64).or_insert(Recipe { op: '-', r1: i, r2: j });
                } else if b > a {
                    self.cached_r1.entry((b - a) as u64).or_insert(Recipe { op: '-', r1: j, r2: i });
                }
                if b != 0 && a % b == 0 {
                    self.cached_r1.entry((a / b) as u64).or_insert(Recipe { op: '/', r1: i, r2: j });
                }
                if a != 0 && b % a == 0 {
                    self.cached_r1.entry((b / a) as u64).or_insert(Recipe { op: '/', r1: j, r2: i });
                }
            }
        }
        self.r1_dirty = false;
    }

    fn exact_small_cost(&self, x: u64) -> Option<i32> {
        if x == 0 { return None; }
        if let Some(&r) = self.val_to_reg.get(&x) {
            if self.defined[r] && self.regs[r] == x { return Some(0); }
        }
        if self.cached_r1.contains_key(&x) { return Some(1); }
        let lim = self.p.value_limit as u128;
        for r in 0..self.p.registers {
            if !self.defined[r] { continue; }
            let h = self.regs[r];
            if x > h && self.cached_r1.contains_key(&(x - h)) { return Some(2); }
            if (x as u128) + (h as u128) <= lim && self.cached_r1.contains_key(&(x + h)) { return Some(2); }
            if h > x && self.cached_r1.contains_key(&(h - x)) { return Some(2); }
            if h > 1 && x % h == 0 && self.cached_r1.contains_key(&(x / h)) { return Some(2); }
            if (x as u128) * (h as u128) <= lim && self.cached_r1.contains_key(&(x * h)) { return Some(2); }
        }
        None
    }

    fn est_cost(&self, x: u64) -> i32 {
        if let Some(c) = self.exact_small_cost(x) { return c; }
        let bits = 64 - x.leading_zeros() as i32;
        let pop = x.count_ones() as i32;
        (bits - 1) + (pop - 1)
    }

    fn build_2step(&mut self, target: u64) -> Option<usize> {
        if let Some(&r) = self.val_to_reg.get(&target) {
            if self.defined[r] && self.regs[r] == target {
                self.step_counter += 1;
                self.use_age[r] = self.step_counter;
                return Some(r);
            }
        }
        if let Some(&rec) = self.cached_r1.get(&target) {
            return self.emit(rec.op, rec.r1, rec.r2);
        }
        let lim = self.p.value_limit as u128;
        for r in 0..self.p.registers {
            if !self.defined[r] { continue; }
            let h = self.regs[r];
            let candidates = [
                (target > h).then(|| (target - h, '+', false)),
                ((target as u128) + (h as u128) <= lim).then(|| (target + h, '-', false)),
                (h > target).then(|| (h - target, '-', true)),
                (h > 1 && target % h == 0).then(|| (target / h, '*', false)),
                ((target as u128) * (h as u128) <= lim).then(|| (target * h, '/', false)),
            ];
            for (need, op, h_first) in candidates.into_iter().flatten() {
                if let Some(&rec) = self.cached_r1.get(&need) {
                    self.pin_count[r] += 1;
                    let mid = self.emit(rec.op, rec.r1, rec.r2);
                    self.pin_count[r] -= 1;
                    if let Some(m) = mid {
                        if self.defined[r] && self.regs[r] == h {
                            return if h_first { self.emit(op, r, m) } else { self.emit(op, m, r) };
                        }
                    }
                }
            }
        }
        None
    }

    // Top-K branching at depth == 0 to pick the actual shortest decomposition!
    fn build_value_beam(&mut self, target: u64, depth: usize) -> Option<usize> {
        if target == 1 { return self.ensure_one(); }
        self.rebuild_r1();
        if let Some(c) = self.exact_small_cost(target) {
            if c <= 2 {
                if let Some(res) = self.build_2step(target) { return Some(res); }
            }
        }

        if depth < 5 && target > 3 {
            let lim = self.p.value_limit as u128;
            let held_vals: Vec<u64> = (0..self.p.registers)
                .filter(|&r| self.defined[r] && self.regs[r] > 1)
                .map(|r| self.regs[r])
                .collect();
            let mut r1_keys: Vec<u64> = self.cached_r1.keys().copied().filter(|&k| k > 1).collect();
            r1_keys.sort_unstable();

            // Collect top-K candidates (est_cost, d, q, rem, is_sub)
            let mut cands: Vec<(i32, u64, u64, u64, bool)> = Vec::new();
            let mut consider_d = |d: u64, d_cost: i32| {
                if d >= target {
                    let rem = d - target;
                    cands.push((d_cost + self.est_cost(rem) + 1, d, 1, rem, true));
                    return;
                }
                let q = target / d;
                let rem = target % d;
                if rem == 0 {
                    cands.push((d_cost + self.est_cost(q) + 1, d, q, 0, false));
                } else {
                    if (q as u128) * (d as u128) <= lim {
                        let qc = if q == 1 { 0 } else { self.est_cost(q) + 1 };
                        cands.push((d_cost + qc + self.est_cost(rem) + 1, d, q, rem, false));
                    }
                    let q2 = q + 1;
                    let rem2 = d - rem;
                    if (q2 as u128) * (d as u128) <= lim {
                        let q2c = if q2 == 1 { 0 } else { self.est_cost(q2) + 1 };
                        cands.push((d_cost + q2c + self.est_cost(rem2) + 1, d, q2, rem2, true));
                    }
                }
            };

            for d in held_vals { consider_d(d, 0); }
            for d in r1_keys { consider_d(d, 1); }
            cands.sort_unstable_by_key(|x| x.0);

            let k_try = if depth == 0 { 3 } else { 1 };
            let bin_est = self.est_cost(target);
            let mut best_state: Option<(BeamState<'a>, usize)> = None;

            for &(est, d, q, rem, is_sub) in cands.iter().take(k_try) {
                if est >= bin_est { break; }
                let mut trial = if k_try == 1 {
                    std::mem::replace(self, BeamState::new(self.p))
                } else {
                    self.clone()
                };
                let ok_reg = if q == 1 {
                    trial.build_value_beam(d, depth + 1).and_then(|dr| {
                        trial.pin_count[dr] += 1;
                        let rr = trial.build_value_beam(rem, depth + 1);
                        trial.pin_count[dr] -= 1;
                        rr.and_then(|r| {
                            if trial.defined[dr] && trial.regs[dr] == d {
                                trial.emit(if is_sub { '-' } else { '+' }, dr, r)
                            } else { None }
                        })
                    })
                } else {
                    trial.build_value_beam(q, depth + 1).and_then(|qr| {
                        trial.pin_count[qr] += 1;
                        let dr = trial.build_value_beam(d, depth + 1);
                        let prod = dr.and_then(|dreg| {
                            if trial.regs[dreg] == d { trial.emit('*', qr, dreg) } else { None }
                        });
                        trial.pin_count[qr] -= 1;
                        prod.and_then(|pr| {
                            if rem == 0 { return Some(pr); }
                            trial.pin_count[pr] += 1;
                            let rr = trial.build_value_beam(rem, depth + 1);
                            trial.pin_count[pr] -= 1;
                            rr.and_then(|r| {
                                if trial.defined[pr] && trial.regs[pr] == q * d {
                                    trial.emit(if is_sub { '-' } else { '+' }, pr, r)
                                } else { None }
                            })
                        })
                    })
                };
                if let Some(res_r) = ok_reg {
                    if k_try == 1 {
                        *self = trial;
                        return Some(res_r);
                    }
                    if best_state.as_ref().map_or(true, |(bs, _)| trial.prog.len() < bs.prog.len()) {
                        best_state = Some((trial, res_r));
                    }
                } else if k_try == 1 {
                    *self = trial;
                    break;
                }
            }
            if let Some((bs, r)) = best_state {
                *self = bs;
                return Some(r);
            }
        }

        // Binary doubling fallback
        let one_r = self.ensure_one()?;
        let bits = 64 - target.leading_zeros() as i32;
        if bits <= 1 { return Some(one_r); }
        let mut cur = self.emit('+', one_r, one_r)?;
        if ((target >> (bits - 2)) & 1) == 1 { cur = self.emit('+', cur, one_r)?; }
        for b in (0..=(bits - 3)).rev() {
            cur = self.emit('+', cur, cur)?;
            if ((target >> b) & 1) == 1 { cur = self.emit('+', cur, one_r)?; }
        }
        Some(cur)
    }
}

fn solve_beam(p: &Puzzle) -> Vec<Inst> {
    let mut st = BeamState::new(p);
    let mut remaining: Vec<u64> = p.targets.iter().copied().filter(|t| !st.produced.contains(t)).collect();
    while !remaining.is_empty() {
        st.rebuild_r1();
        let mut best_idx = 0;
        let mut best_c = i32::MAX;
        for (i, &t) in remaining.iter().enumerate() {
            if st.produced.contains(&t) { best_idx = i; break; }
            let c = st.est_cost(t);
            if c < best_c {
                best_c = c;
                best_idx = i;
                if c <= 1 { break; }
            }
        }
        let t = remaining.remove(best_idx);
        if !st.produced.contains(&t) {
            st.build_value_beam(t, 0);
        }
    }
    let pruned = dce_prune(p, &st.prog);
    if validate_program(p, &pruned) { pruned } else { st.prog }
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
    if args.len() < 3 {
        eprintln!("Usage: {} <puzzles.csv> <out.csv>", args[0]);
        std::process::exit(1);
    }
    let t0 = Instant::now();
    let puzzles = load_puzzles(&args[1]);
    let n = puzzles.len();
    let results: Vec<std::sync::Mutex<Option<Vec<Inst>>>> = (0..n).map(|_| std::sync::Mutex::new(None)).collect();
    let next_idx = AtomicUsize::new(0);
    let n_threads = thread::available_parallelism().map(|x| x.get()).unwrap_or(8);

    thread::scope(|s| {
        for _ in 0..n_threads {
            s.spawn(|| loop {
                let idx = next_idx.fetch_add(1, Ordering::Relaxed);
                if idx >= n { break; }
                let prog = solve_beam(&puzzles[idx]);
                *results[idx].lock().unwrap() = Some(prog);
            });
        }
    });

    let out_file = File::create(&args[2]).expect("Cannot create output CSV");
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
    println!("Beam solved {} puzzles in {:.2}s. Total score = {}", n, t0.elapsed().as_secs_f64(), total);
}
