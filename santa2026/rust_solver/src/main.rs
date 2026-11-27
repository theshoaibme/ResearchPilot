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
    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
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
    if let Ok(file) = File::open(path) {
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

#[derive(Clone, Copy)]
struct Recipe {
    op: char,
    r1: usize,
    r2: usize,
}

#[derive(Clone, Copy)]
struct D2Recipe {
    r1_val: u64,
    op: char,
    h_reg: usize,
    h_first: bool,
}

struct SolverState<'a> {
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
    rng: Rng,
    cached_r1: HashMap<u64, Recipe>,
    r1_dirty: bool,
    target_set: HashSet<u64>,
}

impl<'a> SolverState<'a> {
    fn new(p: &'a Puzzle, seed: u64) -> Self {
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
        let target_set = p.targets.iter().copied().collect();
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
            rng: Rng(seed.max(1)),
            cached_r1: HashMap::with_capacity(4096),
            r1_dirty: true,
            target_set,
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
                        if (diff as u128) * 20 <= (t as u128) {
                            score += 150_000;
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
        if let Some(r) = best_r {
            return r;
        }
        for r in 0..self.p.registers {
            if Some(r) != avoid1 && Some(r) != avoid2 && self.pin_count[r] == 0 {
                return r;
            }
        }
        0
    }

    fn emit(&mut self, op: char, i: usize, j: usize, dst_opt: Option<usize>) -> Option<usize> {
        if self.prog.len() >= 4096 || !self.defined[i] || !self.defined[j] {
            return None;
        }
        let a = self.regs[i] as u128;
        let b = self.regs[j] as u128;
        let res = match op {
            '+' => a + b,
            '-' => {
                if a <= b {
                    return None;
                }
                a - b
            }
            '*' => a * b,
            '/' => {
                if b == 0 || a % b != 0 {
                    return None;
                }
                a / b
            }
            _ => return None,
        };
        if res == 0 || res > self.p.value_limit as u128 {
            return None;
        }
        let v = res as u64;
        if let Some(&existing) = self.val_to_reg.get(&v) {
            if self.defined[existing] && self.regs[existing] == v {
                self.step_counter += 1;
                self.use_age[existing] = self.step_counter;
                return Some(existing);
            }
        }
        let dst = dst_opt.unwrap_or_else(|| self.alloc_reg(Some(i), Some(j)));
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
                self.one_reg = self.emit('/', r, r, None);
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

    fn build_d2_map(&self) -> HashMap<u64, D2Recipe> {
        let mut d2: HashMap<u64, D2Recipe> = HashMap::with_capacity(8192);
        let lim = self.p.value_limit as u128;
        let active: Vec<(usize, u64)> = (0..self.p.registers)
            .filter(|&r| self.defined[r])
            .map(|r| (r, self.regs[r]))
            .collect();
        let mut r1_sorted: Vec<u64> = self.cached_r1.keys().copied().filter(|&k| k > 1).collect();
        r1_sorted.sort_unstable();
        let n_r1 = r1_sorted.len();
        let take_small = n_r1.min(125);
        let take_large = n_r1.min(125);
        let mut sample_keys: Vec<u64> = Vec::with_capacity(take_small + take_large);
        sample_keys.extend_from_slice(&r1_sorted[..take_small]);
        if n_r1 > take_small {
            let start_large = (n_r1 - take_large).max(take_small);
            sample_keys.extend_from_slice(&r1_sorted[start_large..]);
        }
        for r1v in sample_keys {
            let a = r1v as u128;
            for &(hr, hv) in &active {
                let b = hv as u128;
                let s = a + b;
                if s > 0 && s <= lim {
                    d2.entry(s as u64).or_insert(D2Recipe { r1_val: r1v, op: '+', h_reg: hr, h_first: false });
                }
                let m = a * b;
                if m > 0 && m <= lim {
                    d2.entry(m as u64).or_insert(D2Recipe { r1_val: r1v, op: '*', h_reg: hr, h_first: false });
                }
                if a > b {
                    d2.entry((a - b) as u64).or_insert(D2Recipe { r1_val: r1v, op: '-', h_reg: hr, h_first: false });
                } else if b > a {
                    d2.entry((b - a) as u64).or_insert(D2Recipe { r1_val: r1v, op: '-', h_reg: hr, h_first: true });
                }
                if b != 0 && a % b == 0 {
                    d2.entry((a / b) as u64).or_insert(D2Recipe { r1_val: r1v, op: '/', h_reg: hr, h_first: false });
                }
                if a != 0 && b % a == 0 {
                    d2.entry((b / a) as u64).or_insert(D2Recipe { r1_val: r1v, op: '/', h_reg: hr, h_first: true });
                }
            }
        }
        d2
    }

    fn exact_small_cost(&self, x: u64) -> Option<i32> {
        if x == 0 {
            return None;
        }
        if let Some(&r) = self.val_to_reg.get(&x) {
            if self.defined[r] && self.regs[r] == x {
                return Some(0);
            }
        }
        if self.cached_r1.contains_key(&x) {
            return Some(1);
        }
        let lim = self.p.value_limit as u128;
        for r in 0..self.p.registers {
            if !self.defined[r] {
                continue;
            }
            let h = self.regs[r];
            if x > h && self.cached_r1.contains_key(&(x - h)) {
                return Some(2);
            }
            if (x as u128) + (h as u128) <= lim && self.cached_r1.contains_key(&(x + h)) {
                return Some(2);
            }
            if h > x && self.cached_r1.contains_key(&(h - x)) {
                return Some(2);
            }
            if h > 1 && x % h == 0 && self.cached_r1.contains_key(&(x / h)) {
                return Some(2);
            }
            if (x as u128) * (h as u128) <= lim && self.cached_r1.contains_key(&(x * h)) {
                return Some(2);
            }
        }
        None
    }

    fn est_cost(&self, x: u64) -> i32 {
        if let Some(c) = self.exact_small_cost(x) {
            return c;
        }
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
            return self.emit(rec.op, rec.r1, rec.r2, None);
        }
        let lim = self.p.value_limit as u128;
        for r in 0..self.p.registers {
            if !self.defined[r] {
                continue;
            }
            let h = self.regs[r];
            let candidates = [
                (target > h).then(|| (target - h, '+', false)),
                ((target as u128) + (h as u128) <= lim).then(|| (target + h, '-', false)),
                (h > target).then(|| (h - target, '-', true)),
                (h > 1 && target % h == 0).then(|| (target / h, '*', false)),
                ((target as u128) * (h as u128) <= lim).then(|| (target * h, '/', false)),
            ];
            for cand in candidates.into_iter().flatten() {
                let (need, op, h_first) = cand;
                if let Some(&rec) = self.cached_r1.get(&need) {
                    self.pin_count[r] += 1;
                    let mid = self.emit(rec.op, rec.r1, rec.r2, None);
                    self.pin_count[r] -= 1;
                    if let Some(m) = mid {
                        if self.defined[r] && self.regs[r] == h {
                            return if h_first {
                                self.emit(op, r, m, None)
                            } else {
                                self.emit(op, m, r, None)
                            };
                        }
                    }
                }
            }
        }
        None
    }

    fn build_value(&mut self, target: u64, depth: usize, jitter: bool) -> Option<usize> {
        if target == 1 {
            return self.ensure_one();
        }
        self.rebuild_r1();
        if let Some(c) = self.exact_small_cost(target) {
            if c <= 2 {
                if let Some(res) = self.build_2step(target) {
                    return Some(res);
                }
            }
        }

        // 3-step Meet-in-the-Middle: target = r2_val op h
        if depth <= 2 {
            let lim = self.p.value_limit as u128;
            for r in 0..self.p.registers {
                if !self.defined[r] {
                    continue;
                }
                let h = self.regs[r];
                let candidates = [
                    (target > h).then(|| (target - h, '+', false)),
                    ((target as u128) + (h as u128) <= lim).then(|| (target + h, '-', false)),
                    (h > target).then(|| (h - target, '-', true)),
                    (h > 1 && target % h == 0).then(|| (target / h, '*', false)),
                    ((target as u128) * (h as u128) <= lim).then(|| (target * h, '/', false)),
                ];
                for cand in candidates.into_iter().flatten() {
                    let (need, op, h_first) = cand;
                    if self.exact_small_cost(need) == Some(2) {
                        self.pin_count[r] += 1;
                        let mid = self.build_2step(need);
                        self.pin_count[r] -= 1;
                        if let Some(m) = mid {
                            if self.defined[r] && self.regs[r] == h {
                                if let Some(res) = if h_first {
                                    self.emit(op, r, m, None)
                                } else {
                                    self.emit(op, m, r, None)
                                } {
                                    return Some(res);
                                }
                            }
                        }
                    }
                }
            }
        }

        // 4-step Meet-in-the-Middle: target = d2_val op r1_val
        if depth == 0 {
            let lim = self.p.value_limit as u128;
            let d2 = self.build_d2_map();
            let mut r1_pairs: Vec<(u64, Recipe)> = self.cached_r1.iter().map(|(&k, &v)| (k, v)).collect();
            r1_pairs.sort_unstable_by_key(|x| x.0);
            let mut r1_sample: Vec<(u64, Recipe)> = Vec::with_capacity(200);
            let half = r1_pairs.len().min(100);
            r1_sample.extend_from_slice(&r1_pairs[..half]);
            if r1_pairs.len() > half {
                let start_hi = (r1_pairs.len() - r1_pairs.len().min(100)).max(half);
                r1_sample.extend_from_slice(&r1_pairs[start_hi..]);
            }
            for (r1v, rec) in r1_sample {
                let candidates = [
                    (target > r1v).then(|| (target - r1v, '+', false)),
                    ((target as u128) + (r1v as u128) <= lim).then(|| (target + r1v, '-', false)),
                    (r1v > target).then(|| (r1v - target, '-', true)),
                    (r1v > 1 && target % r1v == 0).then(|| (target / r1v, '*', false)),
                ];
                for cand in candidates.into_iter().flatten() {
                    let (need, op, r1_first) = cand;
                    if let Some(&d2r) = d2.get(&need) {
                        let h_val = self.regs[d2r.h_reg];
                        if let Some(&r1_rec) = self.cached_r1.get(&d2r.r1_val) {
                            self.pin_count[d2r.h_reg] += 1;
                            let mid1 = self.emit(r1_rec.op, r1_rec.r1, r1_rec.r2, None);
                            self.pin_count[d2r.h_reg] -= 1;
                            if let Some(m1) = mid1 {
                                if self.defined[d2r.h_reg] && self.regs[d2r.h_reg] == h_val {
                                    let d2_reg = if d2r.h_first {
                                        self.emit(d2r.op, d2r.h_reg, m1, None)
                                    } else {
                                        self.emit(d2r.op, m1, d2r.h_reg, None)
                                    };
                                    if let Some(d2r_reg) = d2_reg {
                                        self.pin_count[d2r_reg] += 1;
                                        let r1_reg = self.emit(rec.op, rec.r1, rec.r2, None);
                                        self.pin_count[d2r_reg] -= 1;
                                        if let Some(r1r) = r1_reg {
                                            if self.defined[d2r_reg] && self.regs[d2r_reg] == need {
                                                let res = if r1_first {
                                                    self.emit(op, r1r, d2r_reg, None)
                                                } else {
                                                    self.emit(op, d2r_reg, r1r, None)
                                                };
                                                if res.is_some() {
                                                    return res;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if depth < 6 && target > 3 {
            let mut best_cost = self.est_cost(target);
            let mut best_d: u64 = 0;
            let mut best_q: u64 = 0;
            let mut best_rem: u64 = 0;
            let mut best_sub = false;
            let lim = self.p.value_limit as u128;

            let held_vals: Vec<u64> = (0..self.p.registers)
                .filter(|&r| self.defined[r] && self.regs[r] > 1)
                .map(|r| self.regs[r])
                .collect();
            let mut r1_keys: Vec<u64> = self.cached_r1.keys().copied().filter(|&k| k > 1).collect();
            r1_keys.sort_unstable();
            let mut r2_divs: Vec<u64> = Vec::new();
            if target > 1_000_000 && depth <= 2 {
                r2_divs.reserve(512);
                for &hv in &held_vals {
                    let h = hv as u128;
                    // Use both largest and smallest r1 keys for rich R2 divisors
                    let start_hi = r1_keys.len().saturating_sub(60);
                    for &rk in r1_keys[start_hi..].iter().chain(r1_keys.iter().take(20)) {
                        let prod = h * (rk as u128);
                        if prod > (rk as u128) && prod <= (target as u128) && prod <= lim {
                            r2_divs.push(prod as u64);
                        }
                    }
                }
                r2_divs.sort_unstable();
                r2_divs.dedup();
                if r2_divs.len() > 80 {
                    r2_divs.drain(..r2_divs.len() - 80);
                }
            }

            let mut consider_d = |d: u64, d_cost: i32| {
                if d >= target {
                    let rem = d - target;
                    let mut c = d_cost + self.est_cost(rem) + 1;
                    if jitter && (self.rng.next_u64() % 6 == 0) {
                        c += 1;
                    }
                    if c < best_cost {
                        best_cost = c;
                        best_d = d;
                        best_q = 1;
                        best_rem = rem;
                        best_sub = true;
                    }
                    return;
                }
                let q = target / d;
                let rem = target % d;
                if rem == 0 {
                    let mut c = d_cost + self.est_cost(q) + 1;
                    if jitter && (self.rng.next_u64() % 6 == 0) {
                        c += 1;
                    }
                    if c < best_cost {
                        best_cost = c;
                        best_d = d;
                        best_q = q;
                        best_rem = 0;
                        best_sub = false;
                    }
                } else {
                    if (q as u128) * (d as u128) <= lim {
                        let q_cost = if q == 1 { 0 } else { self.est_cost(q) + 1 };
                        let mut c_add = d_cost + q_cost + self.est_cost(rem) + 1;
                        if jitter && (self.rng.next_u64() % 6 == 0) {
                            c_add += 1;
                        }
                        if c_add < best_cost {
                            best_cost = c_add;
                            best_d = d;
                            best_q = q;
                            best_rem = rem;
                            best_sub = false;
                        }
                    }
                    let q2 = q + 1;
                    let rem2 = d - rem;
                    if (q2 as u128) * (d as u128) <= lim {
                        let q2_cost = if q2 == 1 { 0 } else { self.est_cost(q2) + 1 };
                        let mut c_sub = d_cost + q2_cost + self.est_cost(rem2) + 1;
                        if jitter && (self.rng.next_u64() % 6 == 0) {
                            c_sub += 1;
                        }
                        if c_sub < best_cost {
                            best_cost = c_sub;
                            best_d = d;
                            best_q = q2;
                            best_rem = rem2;
                            best_sub = true;
                        }
                    }
                }
            };

            for d in held_vals {
                consider_d(d, 0);
            }
            for d in r2_divs {
                consider_d(d, 2);
            }
            for d in r1_keys {
                consider_d(d, 1);
            }

            if best_d > 0 {
                if best_q == 1 {
                    if let Some(d_reg) = self.build_value(best_d, depth + 1, jitter) {
                        if self.regs[d_reg] == best_d {
                            self.pin_count[d_reg] += 1;
                            let rem_reg = self.build_value(best_rem, depth + 1, jitter);
                            self.pin_count[d_reg] -= 1;
                            if let Some(rr) = rem_reg {
                                if self.defined[d_reg] && self.regs[d_reg] == best_d {
                                    let op = if best_sub { '-' } else { '+' };
                                    if let Some(res) = self.emit(op, d_reg, rr, None) {
                                        return Some(res);
                                    }
                                }
                            }
                        }
                    }
                } else if let Some(q_reg) = self.build_value(best_q, depth + 1, jitter) {
                    if self.regs[q_reg] == best_q {
                        self.pin_count[q_reg] += 1;
                        let d_reg = self.build_value(best_d, depth + 1, jitter);
                        if let Some(dr) = d_reg {
                            if self.regs[dr] == best_d {
                                let prod_reg = self.emit('*', q_reg, dr, None);
                                self.pin_count[q_reg] -= 1;
                                if let Some(pr) = prod_reg {
                                    if best_rem == 0 {
                                        return Some(pr);
                                    }
                                    self.pin_count[pr] += 1;
                                    let rem_reg = self.build_value(best_rem, depth + 1, jitter);
                                    self.pin_count[pr] -= 1;
                                    if let Some(rr) = rem_reg {
                                        if self.defined[pr] && self.regs[pr] == best_q * best_d {
                                            let op = if best_sub { '-' } else { '+' };
                                            if let Some(res) = self.emit(op, pr, rr, None) {
                                                return Some(res);
                                            }
                                        }
                                    }
                                }
                            } else {
                                self.pin_count[q_reg] -= 1;
                            }
                        } else {
                            self.pin_count[q_reg] -= 1;
                        }
                    }
                }
            }
        }

        // Binary doubling fallback
        let one_r = self.ensure_one()?;
        let bits = 64 - target.leading_zeros() as i32;
        if bits <= 1 {
            return Some(one_r);
        }
        let mut cur = self.emit('+', one_r, one_r, None)?;
        if ((target >> (bits - 2)) & 1) == 1 {
            cur = self.emit('+', cur, one_r, None)?;
        }
        for b in (0..=(bits - 3)).rev() {
            cur = self.emit('+', cur, cur, None)?;
            if ((target >> b) & 1) == 1 {
                cur = self.emit('+', cur, one_r, None)?;
            }
        }
        Some(cur)
    }
}

fn solve_puzzle_mode(p: &Puzzle, mode: usize, seed: u64) -> Vec<Inst> {
    let mut st = SolverState::new(p, seed);
    let mut remaining: Vec<u64> = p
        .targets
        .iter()
        .copied()
        .filter(|t| !st.produced.contains(t))
        .collect();

    let use_jitter = mode >= 2;
    if mode == 1 {
        remaining.reverse();
    } else if mode >= 2 {
        st.rng.shuffle(&mut remaining);
    }

    while !remaining.is_empty() {
        st.rebuild_r1();
        let mut best_idx = 0;
        let mut best_c = i32::MAX;
        for (i, &t) in remaining.iter().enumerate() {
            if st.produced.contains(&t) {
                best_idx = i;
                break;
            }
            let mut c = st.est_cost(t);
            if use_jitter && (st.rng.next_u64() % 5 == 0) {
                c += 1;
            }
            if c < best_c {
                best_c = c;
                best_idx = i;
                if c <= 1 && !use_jitter {
                    break;
                }
            }
        }
        let t = remaining.remove(best_idx);
        if !st.produced.contains(&t) {
            st.build_value(t, 0, use_jitter);
        }
    }

    let pruned = dce_prune(p, &st.prog);
    if validate_program(p, &pruned) {
        pruned
    } else {
        st.prog
    }
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
    if args.len() < 3 {
        eprintln!("Usage: {} <puzzles.csv> <out.csv> [seed_submission.csv]", args[0]);
        std::process::exit(1);
    }
    let t0 = Instant::now();
    let puzzles = load_puzzles(&args[1]);
    let seed_map = if args.len() >= 4 {
        load_seed_programs(&args[3])
    } else {
        HashMap::new()
    };

    let n = puzzles.len();
    let results: Vec<std::sync::Mutex<Option<Vec<Inst>>>> =
        (0..n).map(|_| std::sync::Mutex::new(None)).collect();
    let next_idx = AtomicUsize::new(0);
    let invalid_count = AtomicUsize::new(0);
    let n_threads = thread::available_parallelism().map(|x| x.get()).unwrap_or(8);

    thread::scope(|s| {
        for _ in 0..n_threads {
            s.spawn(|| loop {
                let idx = next_idx.fetch_add(1, Ordering::Relaxed);
                if idx >= n {
                    break;
                }
                let p = &puzzles[idx];
                let mut best: Option<Vec<Inst>> = None;
                if let Some(seed_prog) = seed_map.get(&p.id) {
                    if validate_program(p, seed_prog) {
                        best = Some(seed_prog.clone());
                    }
                }
                for mode in 0..4 {
                    let cand = solve_puzzle_mode(p, mode, 987654 + (idx as u64) * 100 + mode as u64);
                    if validate_program(p, &cand) {
                        if best.as_ref().map_or(true, |b| cand.len() < b.len()) {
                            best = Some(cand);
                        }
                    }
                }
                if let Some(b) = best {
                    *results[idx].lock().unwrap() = Some(b);
                } else {
                    invalid_count.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
    });

    if invalid_count.load(Ordering::Relaxed) > 0 {
        eprintln!("ERROR: {} invalid puzzles!", invalid_count.load(Ordering::Relaxed));
        std::process::exit(1);
    }

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
    println!(
        "Rust solved {} puzzles in {:.2}s. Total score = {}",
        n,
        t0.elapsed().as_secs_f64(),
        total
    );
}
