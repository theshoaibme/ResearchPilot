#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using u64 = uint64_t;
using u128 = unsigned __int128;

struct Inst {
    char op; // '+', '-', '*', '/'
    int i;
    int j;
    int d;
};

struct Puzzle {
    std::string id;
    std::vector<u64> starts;
    std::vector<u64> targets;
    u64 value_limit = 0;
    int registers = 48;
};

static std::vector<std::string> parse_csv_row(const std::string& line) {
    std::vector<std::string> fields;
    std::string cur;
    bool in_quotes = false;
    for (size_t k = 0; k < line.size(); ++k) {
        char c = line[k];
        if (c == '"') {
            if (in_quotes && k + 1 < line.size() && line[k + 1] == '"') {
                cur.push_back('"');
                ++k;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (c == ',' && !in_quotes) {
            fields.push_back(std::move(cur));
            cur.clear();
        } else if (c != '\r') {
            cur.push_back(c);
        }
    }
    fields.push_back(std::move(cur));
    return fields;
}

static std::vector<u64> parse_u64_array(const std::string& s) {
    std::vector<u64> res;
    u64 val = 0;
    bool in_num = false;
    for (char c : s) {
        if (c >= '0' && c <= '9') {
            val = val * 10 + static_cast<u64>(c - '0');
            in_num = true;
        } else if (in_num) {
            res.push_back(val);
            val = 0;
            in_num = false;
        }
    }
    if (in_num) res.push_back(val);
    return res;
}

static std::vector<Puzzle> load_puzzles(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open " + path);
    }
    std::string line;
    std::getline(in, line); // header
    std::vector<Puzzle> puzzles;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto cols = parse_csv_row(line);
        if (cols.size() < 5) continue;
        Puzzle p;
        p.id = cols[0];
        p.starts = parse_u64_array(cols[1]);
        p.targets = parse_u64_array(cols[2]);
        p.value_limit = std::stoull(cols[3]);
        p.registers = std::stoi(cols[4]);
        puzzles.push_back(std::move(p));
    }
    return puzzles;
}

static bool validate_program(const Puzzle& p, const std::vector<Inst>& prog) {
    if (prog.size() > 4096) return false;
    std::vector<u64> regs(p.registers, 0);
    std::vector<bool> defined(p.registers, false);
    std::unordered_set<u64> produced;
    produced.reserve(p.targets.size() * 2 + 64);

    for (size_t i = 0; i < p.starts.size(); ++i) {
        if (i >= static_cast<size_t>(p.registers)) return false;
        regs[i] = p.starts[i];
        defined[i] = true;
        produced.insert(p.starts[i]);
    }

    for (const auto& ins : prog) {
        if (ins.i < 0 || ins.i >= p.registers || ins.j < 0 || ins.j >= p.registers ||
            ins.d < 0 || ins.d >= p.registers) {
            return false;
        }
        if (!defined[ins.i] || !defined[ins.j]) return false;
        u64 a = regs[ins.i];
        u64 b = regs[ins.j];
        u128 res = 0;
        if (ins.op == '+') {
            res = static_cast<u128>(a) + b;
        } else if (ins.op == '-') {
            if (a <= b) return false;
            res = a - b;
        } else if (ins.op == '*') {
            res = static_cast<u128>(a) * b;
        } else if (ins.op == '/') {
            if (b == 0 || a % b != 0) return false;
            res = a / b;
        } else {
            return false;
        }
        if (res == 0 || res > p.value_limit) return false;
        regs[ins.d] = static_cast<u64>(res);
        defined[ins.d] = true;
        produced.insert(static_cast<u64>(res));
    }

    for (u64 t : p.targets) {
        if (produced.find(t) == produced.end()) return false;
    }
    return true;
}

// Backward liveness dead-code elimination
static std::vector<Inst> dce_prune(const Puzzle& p, const std::vector<Inst>& prog) {
    if (prog.empty()) return prog;
    // Forward pass to find first step producing each target
    std::vector<u64> regs(p.registers, 0);
    std::unordered_set<u64> target_set(p.targets.begin(), p.targets.end());
    std::unordered_set<u64> seen_targets;
    for (size_t i = 0; i < p.starts.size(); ++i) {
        regs[i] = p.starts[i];
        if (target_set.count(p.starts[i])) seen_targets.insert(p.starts[i]);
    }

    std::vector<bool> needed(prog.size(), false);
    for (size_t step = 0; step < prog.size(); ++step) {
        const auto& ins = prog[step];
        u64 a = regs[ins.i], b = regs[ins.j];
        u64 res = 0;
        if (ins.op == '+') res = a + b;
        else if (ins.op == '-') res = a - b;
        else if (ins.op == '*') res = a * b;
        else res = a / b;
        regs[ins.d] = res;
        if (target_set.count(res) && !seen_targets.count(res)) {
            seen_targets.insert(res);
            needed[step] = true;
        }
    }

    std::vector<bool> live_reg(p.registers, false);
    for (int step = static_cast<int>(prog.size()) - 1; step >= 0; --step) {
        const auto& ins = prog[step];
        if (needed[step] || live_reg[ins.d]) {
            needed[step] = true;
            live_reg[ins.d] = false; // overwritten here
            live_reg[ins.i] = true;
            live_reg[ins.j] = true;
        }
    }

    std::vector<Inst> out;
    out.reserve(prog.size());
    for (size_t step = 0; step < prog.size(); ++step) {
        if (needed[step]) out.push_back(prog[step]);
    }
    return out;
}

struct StepRecipe {
    char op;
    int r1;
    int r2;
};

struct SolverState {
    const Puzzle& p;
    std::vector<u64> regs;
    std::vector<bool> defined;
    std::vector<int> use_age;
    std::unordered_set<u64> produced;
    std::unordered_map<u64, int> val_to_reg;
    std::vector<Inst> prog;
    std::vector<int> pin_count;
    int step_counter = 0;
    int one_reg = -1;

    explicit SolverState(const Puzzle& puz)
        : p(puz), regs(puz.registers, 0), defined(puz.registers, false), use_age(puz.registers, 0), pin_count(puz.registers, 0) {
        for (size_t i = 0; i < p.starts.size(); ++i) {
            regs[i] = p.starts[i];
            defined[i] = true;
            use_age[i] = ++step_counter;
            produced.insert(p.starts[i]);
            val_to_reg[p.starts[i]] = static_cast<int>(i);
            if (p.starts[i] == 1) one_reg = static_cast<int>(i);
        }
    }

    int alloc_reg(int avoid1 = -1, int avoid2 = -1) {
        for (int r = 0; r < p.registers; ++r) {
            if (!defined[r]) return r;
        }
        int best_r = -1;
        int best_score = std::numeric_limits<int>::max();
        for (int r = 0; r < p.registers; ++r) {
            if (r == one_reg || r == avoid1 || r == avoid2 || pin_count[r] > 0) continue;
            int score = use_age[r];
            if (regs[r] <= 64) score += 100000;
            if (score < best_score) {
                best_score = score;
                best_r = r;
            }
        }
        if (best_r == -1) {
            for (int r = 0; r < p.registers; ++r) {
                if (r != avoid1 && r != avoid2 && pin_count[r] == 0) return r;
            }
            for (int r = 0; r < p.registers; ++r) {
                if (r != avoid1 && r != avoid2) return r;
            }
            return 0;
        }
        return best_r;
    }

    int emit(char op, int i, int j, int dst = -1) {
        u64 a = regs[i], b = regs[j];
        u128 res = 0;
        if (op == '+') res = static_cast<u128>(a) + b;
        else if (op == '-') res = a - b;
        else if (op == '*') res = static_cast<u128>(a) * b;
        else res = a / b;
        if (res == 0 || res > p.value_limit) return -1;
        u64 v = static_cast<u64>(res);
        auto it = val_to_reg.find(v);
        if (it != val_to_reg.end() && defined[it->second] && regs[it->second] == v) {
            use_age[it->second] = ++step_counter;
            return it->second;
        }
        if (dst < 0) dst = alloc_reg(i, j);
        if (defined[dst]) {
            auto old_it = val_to_reg.find(regs[dst]);
            if (old_it != val_to_reg.end() && old_it->second == dst) {
                val_to_reg.erase(old_it);
            }
        }
        prog.push_back({op, i, j, dst});
        regs[dst] = v;
        defined[dst] = true;
        use_age[dst] = ++step_counter;
        produced.insert(v);
        val_to_reg[v] = dst;
        if (v == 1) one_reg = dst;
        return dst;
    }

    void ensure_one() {
        if (one_reg != -1 && defined[one_reg] && regs[one_reg] == 1) return;
        for (int r = 0; r < p.registers; ++r) {
            if (defined[r]) {
                one_reg = emit('/', r, r);
                return;
            }
        }
    }

    std::unordered_map<u64, StepRecipe> build_r1() const {
        std::unordered_map<u64, StepRecipe> r1;
        r1.reserve(4096);
        std::vector<int> active;
        for (int r = 0; r < p.registers; ++r) {
            if (defined[r]) active.push_back(r);
        }
        for (size_t idx1 = 0; idx1 < active.size(); ++idx1) {
            int i = active[idx1];
            u64 a = regs[i];
            for (size_t idx2 = idx1; idx2 < active.size(); ++idx2) {
                int j = active[idx2];
                u64 b = regs[j];
                u128 s = static_cast<u128>(a) + b;
                if (s > 0 && s <= p.value_limit) r1.try_emplace(static_cast<u64>(s), StepRecipe{'+', i, j});
                u128 m = static_cast<u128>(a) * b;
                if (m > 0 && m <= p.value_limit) r1.try_emplace(static_cast<u64>(m), StepRecipe{'*', i, j});
                if (a > b) r1.try_emplace(a - b, StepRecipe{'-', i, j});
                else if (b > a) r1.try_emplace(b - a, StepRecipe{'-', j, i});
                if (b != 0 && a % b == 0) r1.try_emplace(a / b, StepRecipe{'/', i, j});
                if (a != 0 && b % a == 0) r1.try_emplace(b / a, StepRecipe{'/', j, i});
            }
        }
        return r1;
    }

    // Approximate cost to build a small/medium value using binary/held-value decomposition
    int est_cost(u64 x, const std::unordered_map<u64, StepRecipe>& r1) const {
        if (x == 0) return 100000;
        if (val_to_reg.count(x)) return 0;
        if (r1.count(x)) return 1;
        // 2-step check against held registers
        for (int r = 0; r < p.registers; ++r) {
            if (!defined[r]) continue;
            u64 h = regs[r];
            if (x > h && r1.count(x - h)) return 2;
            if (static_cast<u128>(x) + h <= p.value_limit && r1.count(x + h)) return 2;
            if (h > x && r1.count(h - x)) return 2;
            if (h > 1 && x % h == 0 && r1.count(x / h)) return 2;
            if (static_cast<u128>(x) * h <= p.value_limit && r1.count(x * h)) return 2;
        }
        int bits = 64 - __builtin_clzll(x);
        int pop = __builtin_popcountll(x);
        return (bits - 1) + (pop - 1);
    }

    int build_value(u64 target, int depth = 0) {
        if (target == 1) {
            ensure_one();
            return one_reg;
        }
        if (auto it = val_to_reg.find(target); it != val_to_reg.end() && defined[it->second] && regs[it->second] == target) {
            use_age[it->second] = ++step_counter;
            return it->second;
        }
        auto r1 = build_r1();
        if (auto it = r1.find(target); it != r1.end()) {
            return emit(it->second.op, it->second.r1, it->second.r2);
        }

        // 2-step exact check: target = r1_val op h
        for (int r = 0; r < p.registers; ++r) {
            if (!defined[r]) continue;
            u64 h = regs[r];
            if (target > h) {
                if (auto it = r1.find(target - h); it != r1.end()) {
                    int mid = emit(it->second.op, it->second.r1, it->second.r2);
                    if (mid >= 0) return emit('+', mid, r);
                }
            }
            if (static_cast<u128>(target) + h <= p.value_limit) {
                if (auto it = r1.find(target + h); it != r1.end()) {
                    int mid = emit(it->second.op, it->second.r1, it->second.r2);
                    if (mid >= 0) return emit('-', mid, r);
                }
            }
            if (h > target) {
                if (auto it = r1.find(h - target); it != r1.end()) {
                    int mid = emit(it->second.op, it->second.r1, it->second.r2);
                    if (mid >= 0) return emit('-', r, mid);
                }
            }
            if (h > 1 && target % h == 0) {
                if (auto it = r1.find(target / h); it != r1.end()) {
                    int mid = emit(it->second.op, it->second.r1, it->second.r2);
                    if (mid >= 0) return emit('*', mid, r);
                }
            }
            if (static_cast<u128>(target) * h <= p.value_limit) {
                if (auto it = r1.find(target * h); it != r1.end()) {
                    int mid = emit(it->second.op, it->second.r1, it->second.r2);
                    if (mid >= 0) return emit('/', mid, r);
                }
            }
        }

        // Quotient-remainder decomposition using held registers or R1 values
        if (depth < 6 && target > 3) {
            int best_cost = est_cost(target, r1);
            u64 best_d = 0, best_q = 0, best_rem = 0;
            bool best_sub = false;
            bool d_from_r1 = false;
            StepRecipe d_rec{};

            auto consider_d = [&](u64 d, bool from_r1, StepRecipe rec) {
                if (d <= 1 || d >= target) return;
                u64 q = target / d;
                u64 rem = target % d;
                if (rem == 0) {
                    int c = (from_r1 ? 1 : 0) + est_cost(q, r1) + 1;
                    if (c < best_cost) {
                        best_cost = c;
                        best_d = d;
                        best_q = q;
                        best_rem = 0;
                        best_sub = false;
                        d_from_r1 = from_r1;
                        d_rec = rec;
                    }
                } else {
                    int c_add = (from_r1 ? 1 : 0) + est_cost(q, r1) + est_cost(rem, r1) + 2;
                    if (c_add < best_cost && static_cast<u128>(q) * d <= p.value_limit) {
                        best_cost = c_add;
                        best_d = d;
                        best_q = q;
                        best_rem = rem;
                        best_sub = false;
                        d_from_r1 = from_r1;
                        d_rec = rec;
                    }
                    u64 q2 = q + 1;
                    u64 rem2 = d - rem;
                    if (static_cast<u128>(q2) * d <= p.value_limit) {
                        int c_sub = (from_r1 ? 1 : 0) + est_cost(q2, r1) + est_cost(rem2, r1) + 2;
                        if (c_sub < best_cost) {
                            best_cost = c_sub;
                            best_d = d;
                            best_q = q2;
                            best_rem = rem2;
                            best_sub = true;
                            d_from_r1 = from_r1;
                            d_rec = rec;
                        }
                    }
                }
            };

            for (int r = 0; r < p.registers; ++r) {
                if (defined[r]) consider_d(regs[r], false, {});
            }
            for (const auto& kv : r1) {
                if (kv.first > 1 && kv.first < target) {
                    consider_d(kv.first, true, kv.second);
                }
            }

            if (best_d > 0) {
                int q_reg = build_value(best_q, depth + 1);
                if (q_reg >= 0 && regs[q_reg] == best_q) {
                    pin_count[q_reg]++;
                    int d_reg = build_value(best_d, depth + 1);
                    if (d_reg >= 0 && regs[d_reg] == best_d) {
                        int prod_reg = emit('*', q_reg, d_reg);
                        pin_count[q_reg]--;
                        if (prod_reg >= 0) {
                            if (best_rem == 0) return prod_reg;
                            pin_count[prod_reg]++;
                            int rem_reg = build_value(best_rem, depth + 1);
                            pin_count[prod_reg]--;
                            if (rem_reg >= 0 && defined[prod_reg] && regs[prod_reg] == best_q * best_d) {
                                int res = emit(best_sub ? '-' : '+', prod_reg, rem_reg);
                                if (res >= 0) return res;
                            }
                        }
                    } else {
                        pin_count[q_reg]--;
                    }
                }
            }
        }

        // Binary doubling fallback
        ensure_one();
        int dst = alloc_reg(one_reg);
        int bits = 64 - __builtin_clzll(target);
        if (bits <= 1) return one_reg;
        int cur = emit('+', one_reg, one_reg, dst);
        if (cur < 0) return -1;
        if ((target >> (bits - 2)) & 1ULL) cur = emit('+', cur, one_reg, cur);
        for (int b = bits - 3; b >= 0 && cur >= 0; --b) {
            cur = emit('+', cur, cur, cur);
            if (cur >= 0 && ((target >> b) & 1ULL)) cur = emit('+', cur, one_reg, cur);
        }
        return cur;
    }
};

static std::vector<Inst> solve_puzzle(const Puzzle& p, int mode) {
    SolverState st(p);
    std::vector<u64> remaining;
    for (u64 t : p.targets) {
        if (!st.produced.count(t)) remaining.push_back(t);
    }
    if (mode == 1) {
        std::reverse(remaining.begin(), remaining.end());
    }

    while (!remaining.empty()) {
        // Pick next target: if mode == 0 or 1, pick closest/cheapest target!
        auto r1 = st.build_r1();
        size_t best_idx = 0;
        int best_c = std::numeric_limits<int>::max();
        for (size_t i = 0; i < remaining.size(); ++i) {
            if (st.produced.count(remaining[i])) {
                best_idx = i;
                best_c = 0;
                break;
            }
            int c = st.est_cost(remaining[i], r1);
            if (c < best_c) {
                best_c = c;
                best_idx = i;
                if (c <= 1) break;
            }
        }
        u64 t = remaining[best_idx];
        remaining.erase(remaining.begin() + static_cast<long>(best_idx));
        if (!st.produced.count(t)) {
            st.build_value(t, 0);
        }
    }

    auto pruned = dce_prune(p, st.prog);
    if (validate_program(p, pruned)) return pruned;
    return st.prog;
}

static std::vector<Inst> solve_binary_fallback(const Puzzle& p) {
    SolverState st(p);
    st.ensure_one();
    int one_r = st.one_reg;
    for (u64 target : p.targets) {
        if (st.produced.count(target)) continue;
        int dst = st.alloc_reg(one_r);
        int bits = 64 - __builtin_clzll(target);
        if (bits <= 1) continue;
        int cur = st.emit('+', one_r, one_r, dst);
        if ((target >> (bits - 2)) & 1ULL) cur = st.emit('+', cur, one_r, cur);
        for (int b = bits - 3; b >= 0; --b) {
            cur = st.emit('+', cur, cur, cur);
            if ((target >> b) & 1ULL) cur = st.emit('+', cur, one_r, cur);
        }
    }
    return st.prog;
}

static std::string serialize_prog(const std::vector<Inst>& prog) {
    std::ostringstream oss;
    oss << "[";
    for (size_t k = 0; k < prog.size(); ++k) {
        if (k > 0) oss << ",";
        oss << "[\"" << prog[k].op << "\"," << prog[k].i << "," << prog[k].j << "," << prog[k].d << "]";
    }
    oss << "]";
    return oss.str();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <puzzles.csv> <out.csv>\n";
        return 1;
    }
    auto t0 = std::chrono::steady_clock::now();
    auto puzzles = load_puzzles(argv[1]);
    std::vector<std::string> serialized(puzzles.size());
    std::vector<size_t> lengths(puzzles.size(), 0);
    std::atomic<size_t> next_idx{0};
    std::atomic<size_t> invalid_count{0};

    unsigned int n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 4;

    std::vector<std::thread> workers;
    for (unsigned int t = 0; t < n_threads; ++t) {
        workers.emplace_back([&]() {
            while (true) {
                size_t idx = next_idx.fetch_add(1);
                if (idx >= puzzles.size()) break;
                const auto& p = puzzles[idx];
                std::vector<Inst> best_prog;
                for (int mode = 0; mode < 2; ++mode) {
                    auto cand = solve_puzzle(p, mode);
                    if (validate_program(p, cand)) {
                        if (best_prog.empty() || cand.size() < best_prog.size()) {
                            best_prog = std::move(cand);
                        }
                    }
                }
                if (best_prog.empty()) {
                    auto fb = solve_binary_fallback(p);
                    if (validate_program(p, fb)) {
                        best_prog = std::move(fb);
                    }
                }
                if (best_prog.empty() || !validate_program(p, best_prog)) {
                    invalid_count.fetch_add(1);
                } else {
                    lengths[idx] = best_prog.size();
                    serialized[idx] = serialize_prog(best_prog);
                }
            }
        });
    }
    for (auto& w : workers) w.join();

    if (invalid_count > 0) {
        std::cerr << "ERROR: " << invalid_count << " invalid puzzles!\n";
        return 1;
    }

    std::ofstream out(argv[2]);
    out << "id,program\n";
    size_t total = 0;
    for (size_t i = 0; i < puzzles.size(); ++i) {
        total += lengths[i];
        out << puzzles[i].id << ",\"" << [&]() {
            std::string escaped;
            escaped.reserve(serialized[i].size() + 64);
            for (char c : serialized[i]) {
                if (c == '"') escaped += "\"\"";
                else escaped.push_back(c);
            }
            return escaped;
        }() << "\"\n";
    }

    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Solved " << puzzles.size() << " puzzles in " << secs << "s. Total score = " << total << "\n";
    return 0;
}
