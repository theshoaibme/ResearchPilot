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

// Value-level recipe in the dictionary
#[derive(Clone, Copy)]
enum ValSrc {
    Known,
    Op1(char, u64, u64),
    Op2(char, u64, u64, char, u64, u64), // ((a op1 b) op2 c) or (c op2 (a op1 b))
}

struct Dict {
    // value -> (cost, src)
    map: HashMap<u64, (u8, ValSrc)>,
    cost1_vals: Vec<u64>,
    all_vals: Vec<u64>,
}

fn build_dict(leaves: &[u64], limit: u64) -> Dict {
    let mut map: HashMap<u64, (u8, ValSrc)> = HashMap::with_capacity(65536);
    let lim = limit as u128;
    for &v in leaves {
        map.entry(v).or_insert((0, ValSrc::Known));
    }

    // 1-step from leaves
    let mut c1: Vec<(u64, char, u64, u64)> = Vec::with_capacity(8192);
    for i in 0..leaves.len() {
        let a = leaves[i] as u128;
        for j in i..leaves.len() {
            let b = leaves[j] as u128;
            let s = a + b;
            if s > 0 && s <= lim {
                let sv = s as u64;
                if !map.contains_key(&sv) {
                    map.insert(sv, (1, ValSrc::Op1('+', a as u64, b as u64)));
                    c1.push((sv, '+', a as u64, b as u64));
                }
            }
            let m = a * b;
            if m > 0 && m <= lim {
                let mv = m as u64;
                if !map.contains_key(&mv) {
                    map.insert(mv, (1, ValSrc::Op1('*', a as u64, b as u64)));
                    c1.push((mv, '*', a as u64, b as u64));
                }
            }
            if a > b {
                let dv = (a - b) as u64;
                if !map.contains_key(&dv) {
                    map.insert(dv, (1, ValSrc::Op1('-', a as u64, b as u64)));
                    c1.push((dv, '-', a as u64, b as u64));
                }
            } else if b > a {
                let dv = (b - a) as u64;
                if !map.contains_key(&dv) {
                    map.insert(dv, (1, ValSrc::Op1('-', b as u64, a as u64)));
                    c1.push((dv, '-', b as u64, a as u64));
                }
            }
            if b != 0 && a % b == 0 {
                let qv = (a / b) as u64;
                if !map.contains_key(&qv) {
                    map.insert(qv, (1, ValSrc::Op1('/', a as u64, b as u64)));
                    c1.push((qv, '/', a as u64, b as u64));
                }
            }
            if a != 0 && b % a == 0 {
                let qv = (b / a) as u64;
                if !map.contains_key(&qv) {
                    map.insert(qv, (1, ValSrc::Op1('/', b as u64, a as u64)));
                    c1.push((qv, '/', b as u64, a as u64));
                }
            }
        }
    }

    // Also add x+x and x*x for 1-step entries (cost 2)
    for &(xv, op1, x1, x2) in &c1 {
        let x = xv as u128;
        if x + x <= lim {
            let v = (x + x) as u64;
            map.entry(v).or_insert((2, ValSrc::Op2(op1, x1, x2, '+', xv, xv)));
        }
        if x * x <= lim {
            let v = (x * x) as u64;
            map.entry(v).or_insert((2, ValSrc::Op2(op1, x1, x2, '*', xv, xv)));
        }
    }

    // 2-step: c1 op leaf (cap c1 at 2500 for denser dictionary)
    let c1_cap = c1.len().min(2500);
    for &(xv, op1, x1, x2) in &c1[..c1_cap] {
        let a = xv as u128;
        for &lv in leaves {
            let b = lv as u128;
            let s = a + b;
            if s > 0 && s <= lim {
                map.entry(s as u64).or_insert((2, ValSrc::Op2(op1, x1, x2, '+', xv, lv)));
            }
            let m = a * b;
            if m > 0 && m <= lim {
                map.entry(m as u64).or_insert((2, ValSrc::Op2(op1, x1, x2, '*', xv, lv)));
            }
            if a > b {
                map.entry((a - b) as u64).or_insert((2, ValSrc::Op2(op1, x1, x2, '-', xv, lv)));
            } else if b > a {
                map.entry((b - a) as u64).or_insert((2, ValSrc::Op2(op1, x1, x2, '-', lv, xv)));
            }
            if b != 0 && a % b == 0 {
                map.entry((a / b) as u64).or_insert((2, ValSrc::Op2(op1, x1, x2, '/', xv, lv)));
            }
            if a != 0 && b % a == 0 {
                map.entry((b / a) as u64).or_insert((2, ValSrc::Op2(op1, x1, x2, '/', lv, xv)));
            }
        }
    }

    let mut cost1_vals: Vec<u64> = leaves.to_vec();
    cost1_vals.extend(c1.iter().map(|x| x.0));
    cost1_vals.sort_unstable();
    cost1_vals.dedup();

    let mut all_vals: Vec<u64> = map.keys().copied().collect();
    all_vals.sort_unstable();

    Dict { map, cost1_vals, all_vals }
}

// Backward beam search for a single target `goal` using `Dict`
// Returns a list of (op, a_val, b_val, out_val) in topological order
fn backward_beam(goal: u64, dict: &Dict, limit: u64) -> Option<Vec<(char, u64, u64, u64)>> {
    if dict.map.get(&goal).map_or(false, |&(c, _)| c == 0) {
        return Some(Vec::new());
    }

    #[derive(Clone)]
    struct Node {
        cur: u64,
        steps: Vec<(char, u64, u64, u64)>, // stored in reverse (goal -> leaves)
        cost: usize,
        score: f64,
    }

    let materialize_dict_val = |v: u64, out: &mut Vec<(char, u64, u64, u64)>| {
        if let Some(&(_, src)) = dict.map.get(&v) {
            match src {
                ValSrc::Known => {}
                ValSrc::Op1(op, a, b) => {
                    out.push((op, a, b, v));
                }
                ValSrc::Op2(op1, x1, x2, op2, l, r) => {
                    let mid = match op1 {
                        '+' => x1 + x2,
                        '-' => x1 - x2,
                        '*' => x1 * x2,
                        '/' => x1 / x2,
                        _ => 0,
                    };
                    out.push((op1, x1, x2, mid));
                    out.push((op2, l, r, v));
                }
            }
        }
    };

    let lim = limit as u128;
    let beam_w = 24usize;
    let alpha = 0.18f64;

    let mut beam = vec![Node {
        cur: goal,
        steps: Vec::new(),
        cost: 0,
        score: (goal as f64).log2() * alpha,
    }];

    let mut best_complete: Option<Vec<(char, u64, u64, u64)>> = None;
    let mut best_len = usize::MAX;

    for depth in 0..14 {
        let mut next_beam: Vec<Node> = Vec::new();
        for node in &beam {
            if let Some(&(dc, _)) = dict.map.get(&node.cur) {
                if node.cost + (dc as usize) < best_len {
                    let mut full = node.steps.clone();
                    full.reverse();
                    let mut prefix = Vec::new();
                    materialize_dict_val(node.cur, &mut prefix);
                    prefix.extend(full);
                    // Deduplicate by output value
                    let mut seen = HashSet::new();
                    let mut deduped = Vec::new();
                    for s in prefix {
                        if seen.insert(s.3) {
                            deduped.push(s);
                        }
                    }
                    if deduped.len() < best_len {
                        best_len = deduped.len();
                        best_complete = Some(deduped);
                    }
                }
                continue;
            }
            if node.cost >= best_len {
                continue;
            }

            let v = node.cur;
            // Build divisor list: cost1_vals, plus large cost-2 vals (< 2^28) at depth <= 2
            let extra_root: Vec<u64> = if depth <= 2 {
                let cap = 1u64 << 28;
                let pos = dict.all_vals.partition_point(|&x| x <= cap && x < v);
                let take_n = if depth == 0 { 250 } else { 100 };
                let start = pos.saturating_sub(take_n);
                dict.all_vals[start..pos].to_vec()
            } else {
                Vec::new()
            };
            // 1. Reduce via c * f + r or (c+1) * f - r' where f in cost1_vals (or extra_root), r in dict
            for &f in dict.cost1_vals.iter().chain(extra_root.iter()) {
                if f <= 1 { continue; }
                let fc = dict.map[&f].0 as usize;
                if v % f == 0 {
                    let c = v / f;
                    if c >= 1 {
                        let nc = node.cost + fc + 1;
                        let mut st = node.steps.clone();
                        st.push(('*', c, f, v));
                        next_beam.push(Node {
                            cur: c,
                            steps: st,
                            cost: nc,
                            score: nc as f64 + (c.max(1) as f64).log2() * alpha,
                        });
                    }
                } else if v > f {
                    let c = v / f;
                    let r = v % f;
                    if let Some(&(rc, _)) = dict.map.get(&r) {
                        let prod = c * f;
                        let nc = node.cost + fc + (rc as usize) + 2;
                        let mut st = node.steps.clone();
                        st.push(('+', prod, r, v));
                        st.push(('*', c, f, prod));
                        next_beam.push(Node {
                            cur: c,
                            steps: st,
                            cost: nc,
                            score: nc as f64 + (c.max(1) as f64).log2() * alpha,
                        });
                    }
                    let c2 = c + 1;
                    let r2 = f - r;
                    if (c2 as u128) * (f as u128) <= lim {
                        if let Some(&(rc2, _)) = dict.map.get(&r2) {
                            let prod = c2 * f;
                            let nc = node.cost + fc + (rc2 as usize) + 2;
                            let mut st = node.steps.clone();
                            st.push(('-', prod, r2, v));
                            st.push(('*', c2, f, prod));
                            next_beam.push(Node {
                                cur: c2,
                                steps: st,
                                cost: nc,
                                score: nc as f64 + (c2.max(1) as f64).log2() * alpha,
                            });
                        }
                    }
                }
            }

            // 2. Also allow v = a + c or v = a - c for large a in all_vals (good for shared/root!)
            if depth <= 2 {
                // Binary search around `v` in dict.all_vals
                let pos = dict.all_vals.partition_point(|&x| x < v);
                let lo = pos.saturating_sub(25);
                let hi = (pos + 25).min(dict.all_vals.len());
                for &a in &dict.all_vals[lo..hi] {
                    if a == v || a == 0 { continue; }
                    let ac = dict.map[&a].0 as usize;
                    let (c, op, lhs, rhs) = if v > a {
                        (v - a, '+', a, v - a)
                    } else {
                        (a - v, '-', a, a - v)
                    };
                    if c < v {
                        let nc = node.cost + ac + 1;
                        let mut st = node.steps.clone();
                        st.push((op, lhs, rhs, v));
                        next_beam.push(Node {
                            cur: c,
                            steps: st,
                            cost: nc,
                            score: nc as f64 + (c.max(1) as f64).log2() * alpha,
                        });
                    }
                }
            }
        }

        if next_beam.is_empty() {
            break;
        }
        next_beam.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        // Deduplicate by `cur`
        let mut seen_cur = HashSet::new();
        beam.clear();
        for n in next_beam {
            if seen_cur.insert(n.cur) {
                beam.push(n);
                if beam.len() >= beam_w {
                    break;
                }
            }
        }
    }

    // Materialize any dict dependencies for all steps in best_complete
    best_complete.map(|steps| {
        let mut full = Vec::new();
        let mut have: HashSet<u64> = dict
            .map
            .iter()
            .filter(|(_, &(c, _))| c == 0)
            .map(|(&k, _)| k)
            .collect();
        for (op, a, b, out) in steps {
            if !have.contains(&a) && dict.map.contains_key(&a) {
                materialize_dict_val(a, &mut full);
                have.insert(a);
            }
            if !have.contains(&b) && dict.map.contains_key(&b) {
                materialize_dict_val(b, &mut full);
                have.insert(b);
            }
            if have.insert(out) {
                full.push((op, a, b, out));
            }
        }
        // Final dedup preserving order
        let mut seen = HashSet::new();
        let mut res = Vec::new();
        for s in full {
            if seen.insert(s.3) {
                res.push(s);
            }
        }
        res
    })
}

struct Exec<'a> {
    p: &'a Puzzle,
    regs: Vec<u64>,
    defined: Vec<bool>,
    use_age: Vec<i32>,
    produced: HashSet<u64>,
    val_to_reg: HashMap<u64, usize>,
    prog: Vec<Inst>,
    step_counter: i32,
}

impl<'a> Exec<'a> {
    fn new(p: &'a Puzzle) -> Self {
        let mut regs = vec![0u64; p.registers];
        let mut defined = vec![false; p.registers];
        let mut use_age = vec![0i32; p.registers];
        let mut produced: HashSet<u64> = HashSet::new();
        let mut val_to_reg: HashMap<u64, usize> = HashMap::new();
        let mut step_counter = 0i32;
        for (i, &v) in p.starts.iter().enumerate() {
            regs[i] = v;
            defined[i] = true;
            step_counter += 1;
            use_age[i] = step_counter;
            produced.insert(v);
            val_to_reg.insert(v, i);
        }
        Self { p, regs, defined, use_age, produced, val_to_reg, prog: Vec::new(), step_counter }
    }

    fn emit(&mut self, op: char, a: u64, b: u64, pin_set: Option<&HashSet<u64>>) -> Option<usize> {
        let ra = *self.val_to_reg.get(&a)?;
        let rb = *self.val_to_reg.get(&b)?;
        let av = self.regs[ra] as u128;
        let bv = self.regs[rb] as u128;
        let res = match op {
            '+' => av + bv,
            '-' => if av > bv { av - bv } else { return None; },
            '*' => av * bv,
            '/' => if bv != 0 && av % bv == 0 { av / bv } else { return None; },
            _ => return None,
        };
        if res == 0 || res > self.p.value_limit as u128 { return None; }
        let v = res as u64;
        if let Some(&ex) = self.val_to_reg.get(&v) {
            if self.defined[ex] && self.regs[ex] == v {
                self.step_counter += 1;
                self.use_age[ex] = self.step_counter;
                return Some(ex);
            }
        }
        let mut dst = None;
        for r in 0..self.p.registers {
            if !self.defined[r] { dst = Some(r); break; }
        }
        if dst.is_none() {
            let mut best_r = None;
            let mut best_sc = i64::MAX;
            for r in 0..self.p.registers {
                if r == ra || r == rb || pin_set.map_or(false, |ps| ps.contains(&self.regs[r])) {
                    continue;
                }
                let mut sc = self.use_age[r] as i64;
                if self.regs[r] == 1 { sc += 500_000; }
                if self.regs[r] <= 64 { sc += 150_000; }
                if sc < best_sc { best_sc = sc; best_r = Some(r); }
            }
            dst = best_r;
        }
        let d = dst?;
        if self.defined[d] {
            let old = self.regs[d];
            if self.val_to_reg.get(&old) == Some(&d) { self.val_to_reg.remove(&old); }
        }
        self.prog.push(Inst { op, i: ra, j: rb, d });
        self.regs[d] = v;
        self.defined[d] = true;
        self.step_counter += 1;
        self.use_age[d] = self.step_counter;
        self.produced.insert(v);
        self.val_to_reg.insert(v, d);
        Some(d)
    }
}

fn solve_bbeam_puzzle(p: &Puzzle, mode: usize) -> Option<Vec<Inst>> {
    let mut ex = Exec::new(p);

    if !ex.val_to_reg.contains_key(&1) {
        let s0 = p.starts[0];
        ex.emit('/', s0, s0, None)?;
    }
    let seed_consts = mode >= 2;
    if seed_consts {
        for &(op, a, b) in &[
            ('+', 1, 1), // 2
            ('+', 2, 1), // 3
            ('+', 2, 2), // 4
            ('+', 4, 1), // 5
            ('+', 4, 2), // 6
            ('+', 4, 4), // 8
            ('+', 8, 1), // 9
            ('+', 8, 2), // 10
            ('+', 8, 4), // 12
            ('+', 8, 8), // 16
        ] {
            let _ = ex.emit(op, a, b, None);
        }
    }

    let mut targets = p.targets.clone();
    if mode % 2 == 1 {
        targets.reverse();
    }

    for &t in &targets {
        if ex.produced.contains(&t) { continue; }
        let mut held: Vec<u64> = (0..p.registers)
            .filter(|&r| ex.defined[r])
            .map(|r| ex.regs[r])
            .collect();
        held.sort_unstable();
        held.dedup();
        let mut leaves: Vec<u64> = Vec::with_capacity(75);
        for &v in &held {
            if v < 65536 && leaves.len() < 45 {
                leaves.push(v);
            }
        }
        for &v in held.iter().rev() {
            if leaves.len() >= 75 { break; }
            if v >= 65536 {
                leaves.push(v);
            }
        }

        let dict = build_dict(&leaves, p.value_limit);
        if let Some(steps) = backward_beam(t, &dict, p.value_limit) {
            let mut pin_set: HashSet<u64> = HashSet::new();
            for &(_, a, b, _) in &steps {
                pin_set.insert(a);
                pin_set.insert(b);
            }
            let mut ok = true;
            for (op, a, b, _out) in steps {
                if !ex.val_to_reg.contains_key(&a) || !ex.val_to_reg.contains_key(&b) {
                    ok = false;
                    break;
                }
                if ex.emit(op, a, b, Some(&pin_set)).is_none() {
                    ok = false;
                    break;
                }
            }
            if !ok {
                return None;
            }
        } else {
            // Fallback: NAF / binary doubling from 1 if backward_beam didn't hit dict
            let bits = 64 - t.leading_zeros() as i32;
            if bits > 1 {
                let mut cur_v = 1u64;
                if let Some(d) = ex.emit('+', 1, 1, None) {
                    cur_v = ex.regs[d];
                }
                if ((t >> (bits - 2)) & 1) == 1 {
                    if let Some(d) = ex.emit('+', cur_v, 1, None) {
                        cur_v = ex.regs[d];
                    }
                }
                for b in (0..=(bits - 3)).rev() {
                    if let Some(d) = ex.emit('+', cur_v, cur_v, None) {
                        cur_v = ex.regs[d];
                    }
                    if ((t >> b) & 1) == 1 {
                        if let Some(d) = ex.emit('+', cur_v, 1, None) {
                            cur_v = ex.regs[d];
                        }
                    }
                }
            }
        }
    }

    let pruned = dce_prune(p, &ex.prog);
    if validate_program(p, &pruned) { Some(pruned) } else { None }
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
                let p = &puzzles[idx];
                let mut best: Option<Vec<Inst>> = None;
                for mode in 0..4 {
                    if let Some(cand) = solve_bbeam_puzzle(p, mode) {
                        if best.as_ref().map_or(true, |b| cand.len() < b.len()) {
                            best = Some(cand);
                        }
                    }
                }
                *results[idx].lock().unwrap() = best;
            });
        }
    });

    let mut solved = 0usize;
    let mut total = 0usize;
    let out_file = File::create(&args[2]).expect("Cannot create output CSV");
    let mut writer = BufWriter::new(out_file);
    writeln!(writer, "id,program").unwrap();
    for (idx, p) in puzzles.iter().enumerate() {
        if let Some(prog) = results[idx].lock().unwrap().take() {
            solved += 1;
            total += prog.len();
            let ser = serialize_prog(&prog);
            let escaped = ser.replace('"', "\"\"");
            writeln!(writer, "{},\"{}\"", p.id, escaped).unwrap();
        }
    }
    writer.flush().unwrap();
    println!(
        "BBeam solved {}/{} puzzles in {:.2}s. Sum of solved = {}",
        solved,
        n,
        t0.elapsed().as_secs_f64(),
        total
    );
}
