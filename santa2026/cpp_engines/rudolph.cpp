// Apache 2.0 — Adapted from alexchilton / konbu17 / Tony Li
// Rudolph: backward beam search over a 2-step dictionary, then register allocation by next use.
// Mechanism credited to Tony Li / konbu17 (public notebooks, 2026-11-18); this is our own implementation.
//
// stdin: one puzzle per line: id limit n_starts n_targets starts... targets...   (targets ascending)
// args:  width alpha n_done n_recent
// stdout: id <json program>
//
// Per target: leaves = 1, starts, the most recent previous targets (up to n_done leaves), then the most
// recently created values (up to n_recent). The dictionary holds every value reachable from the leaves
// in 0, 1 or 2 instructions. The beam reduces the target v backward:
//   v = c + a, v = a - c          (a in the dictionary, c the new residual)
//   v = c*f, v = c*f + r, v = (c)*f - r'   (f a dictionary value, r or r' in the dictionary)
// ranked by instructions spent + alpha*log2(residual), until the residual is in the dictionary.
// Every created value gets a definition (op, a, b); compile() emits them in dependency order and keeps
// at most 48 values alive, evicting the one needed farthest in the future and rebuilding it on demand.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;
typedef long long Int;

static int WIDTH = 10; static double ALPHA = 0.18; static int N_DONE = 60; static int N_RECENT = 80;
static int SMALL_BITS = 16; static int SMALL_N = 40; static int HEAD = 1; static int RR_ROUNDS = 2; static int PAIR_ROUNDS = 2; static int PROD_K = 0; static int GROW = 4; static int SLACK = 2;   // finished recipes within SLACK of the best, picked by true distinct-value cost; practice 0 33,132, 1 32,856, 2 32,842   // dictionary growth rounds on a failed search; single-start 39: 8,594 -> 7,974, practice 33,340 -> 33,153   // pair ruin rounds; practice 33,340 vs 33,470 (2x time); public programs 14,362 -> 14,362 on 100   // ruin-and-recreate rounds; practice 33,604 vs 33,738 (accept by compiled length, not needed values: that read +60 worse)   // cheapest-first picks before ascending order; practice: 0 34,560, 1 34,181, 2 34,628, 3 34,283, 4 34,237, 6 34,876; best of all six per puzzle 33,516   // leaves below 2^SMALL_BITS taken first, at most SMALL_N

struct Hash {   // open addressing value -> id, cleared between targets
    vector<Int> k; vector<int> id; vector<size_t> used; size_t mask;
    Hash(int bits) : k(size_t(1) << bits, 0), id(size_t(1) << bits), mask((size_t(1) << bits) - 1) {}
    size_t slot(Int v) const { uint64_t h = uint64_t(v) * 0x9e3779b97f4a7c15ULL; h ^= h >> 29; size_t i = h & mask;
        while (k[i] && k[i] != v) i = (i + 1) & mask; return i; }
    int get(Int v) const { size_t i = slot(v); return k[i] ? id[i] : -1; }
    bool put(Int v, int x) { if (used.size() * 4 > k.size() * 3) return false; size_t i = slot(v); if (k[i]) return false;
        k[i] = v; id[i] = x; used.push_back(i); return true; }
    void clear() { for (size_t i : used) k[i] = 0; used.clear(); }
};

struct E { Int v; int a, b; char op; int cost; };

struct Dict {
    Int lim; Hash h{22}; vector<E> e; vector<pair<Int,int>> sorted, sorted1; int n0 = 0, n1 = 0;
    void add(Int v, int a, int b, char op, int c) { if (v > 0 && v <= lim && h.put(v, e.size())) e.push_back({v, a, b, op, c}); }
    void comb(int i, int j, int c) {
        Int x = e[i].v, y = e[j].v;
        if (x < y) { swap(x, y); swap(i, j); }
        add(x + y, i, j, '+', c);
        if (x != y) add(x - y, i, j, '-', c);
        if (x <= lim / y) add(x * y, i, j, '*', c);
        if (x % y == 0) add(x / y, i, j, '/', c);
    }
    const set<Int>* known = nullptr;   // values the program already holds: an entry equal to one costs nothing
    void build(const vector<Int>& leaves) {
        h.clear(); e.clear(); sorted.clear(); sorted1.clear();
        for (Int v : leaves) add(v, -1, -1, 0, 0);
        n0 = e.size();
        for (int i = 0; i < n0; i++) for (int j = 0; j <= i; j++) comb(i, j, 1);
        n1 = e.size();
        for (int i = n0; i < n1; i++) for (int j = 0; j < n0; j++) comb(i, j, 2);
        for (int i = n0; i < n1; i++) comb(i, i, 2);   // a one-step value with itself: x+x, x*x (2+2 = 4, 10*10); single-start 7,974 -> 7,943, practice 33,153 -> 33,132
        if (known) for (auto& x : e)   // practice 32,842 -> 32,689
            if (x.a >= 0 && known->count(x.v)) { x.a = x.b = -1; x.op = 0; x.cost = 0; }
        sorted.reserve(e.size());
        for (int i = 0; i < (int)e.size(); i++) sorted.push_back({e[i].v, i});
        sort(sorted.begin(), sorted.end());
        sorted1.assign(sorted.begin(), sorted.end());
        sorted1.erase(remove_if(sorted1.begin(), sorted1.end(), [&](const pair<Int,int>& x) { return e[x.second].cost > 1; }), sorted1.end());
    }
};

struct St { Int v; int cost, parent; int f, a; char op; double rank; };
// op: '+' parent = child*f + a ; '-' parent = child*f - a ; 'r' parent = a - child*f   (f = -1 means factor 1)
// product link: 'p' parent = child + f*a ; 'q' parent = f*a - child

struct Search {
    Dict& d; int best = 1 << 30, end = -1, term = -1;
    vector<St> hist, nxt; unordered_map<Int,int> seen;
    vector<array<int,3>> cands;   // finished recipes within SLACK of the best: (estimated cost, hist index, terminal id)
    Search(Dict& d) : d(d) {}
    int ec(int id) const { return id < 0 ? 0 : 1 + d.e[id].cost; }
    void offer(int p, Int c, int f, int a, char op) {
        if (c <= 0 || c > d.lim) return;
        int cost = hist[p].cost + ec(f) + ec(a);
        if (cost >= best) return;
        int t = d.h.get(c);
        if (t >= 0) {
            int tot = cost + d.e[t].cost;
            if (tot <= best + SLACK) { cands.push_back({tot, (int)hist.size(), t}); hist.push_back({c, cost, p, f, a, op, 0}); }
            if (tot < best) { best = tot; end = cands.back()[1]; term = t; }
            return;
        }
        if (cost + 2 >= best || c >= hist[p].v) return;
        double rank = cost + ALPHA * log2((double)c);
        if ((int)nxt.size() == WIDTH && rank >= nxt.back().rank) return;
        auto s = seen.find(c); if (s != seen.end() && s->second <= cost) return;
        for (auto it = nxt.begin(); it != nxt.end(); ++it) if (it->v == c) { if (it->cost <= cost) return; nxt.erase(it); break; }
        St st{c, cost, p, f, a, op, rank};
        auto it = upper_bound(nxt.begin(), nxt.end(), rank, [](double r, const St& x) { return r < x.rank; });
        nxt.insert(it, st); if ((int)nxt.size() > WIDTH) nxt.pop_back();
    }
    void expand(int p) {
        Int v = hist[p].v;
        auto it = lower_bound(d.sorted.begin(), d.sorted.end(), make_pair(v, -1));
        for (int k = -6; k < 6; k++) {   // additive: nearest dictionary values
            long pos = (it - d.sorted.begin()) + k; if (pos < 0 || pos >= (long)d.sorted.size()) continue;
            auto [a, id] = d.sorted[pos]; if (a == v) continue;
            offer(p, v > a ? v - a : a - v, -1, id, v > a ? '+' : 'r');
        }
        auto i1 = lower_bound(d.sorted1.begin(), d.sorted1.end(), make_pair(v, -1));
        for (int k = -6; k < 6; k++) {   // additive: nearest cheap (cost <= 1) values, which the full list crowds out
            long pos = (i1 - d.sorted1.begin()) + k; if (pos < 0 || pos >= (long)d.sorted1.size()) continue;
            auto [a, id] = d.sorted1[pos]; if (a == v) continue;
            offer(p, v > a ? v - a : a - v, -1, id, v > a ? '+' : 'r');
        }
        int fend = p == 0 ? (int)d.e.size() : d.n1;   // root: factors from the whole dictionary below 2^28 (Fable idea 7)
        for (int i = 0; i < fend; i++) {   // multiplicative: factor from the cost <= 1 part (deeper levels)
            Int f = d.e[i].v; if (f < 2 || f > v || (i >= d.n1 && f >= (Int(1) << 28))) continue;
            Int q = v / f, r = v % f;
            if (i < d.n1 && PROD_K) {   // product link (thisray, public 11-18): v = f*Y +- c, Y a dictionary value near v/f, recurse on c
                auto iy = lower_bound(d.sorted.begin(), d.sorted.end(), make_pair(q, -1));
                for (int k = -PROD_K; k < PROD_K; k++) {
                    long pos = (iy - d.sorted.begin()) + k; if (pos < 0 || pos >= (long)d.sorted.size()) continue;
                    auto [y, yi] = d.sorted[pos]; if (y < 2 || y > d.lim / f) continue;
                    Int w = f * y; if (w == v) continue;
                    offer(p, w < v ? v - w : w - v, i, yi, w < v ? 'p' : 'q');
                }
            }
            if (!r) { offer(p, q, i, -1, '+'); continue; }
            int ri = d.h.get(r); if (ri >= 0) offer(p, q, i, ri, '+');
            if (f - r <= d.lim - v) { ri = d.h.get(f - r); if (ri >= 0) offer(p, q + 1, i, ri, '-'); }
        }
    }
    void run(Int target) {
        hist.push_back({target, 0, -1, -1, -1, 0, 0});
        int t = d.h.get(target); if (t >= 0) { best = d.e[t].cost; end = 0; term = t; return; }
        vector<int> front{0};
        for (int depth = 0; depth < 24 && !front.empty(); depth++) {
            nxt.clear();
            for (int p : front) if (hist[p].cost + 2 < best) expand(p);
            front.clear();
            for (auto& s : nxt) { if (s.cost + 2 >= best) continue; seen[s.v] = s.cost; front.push_back(hist.size()); hist.push_back(s); }
        }
    }
};

struct Puzzle {
    Int lim; vector<Int> starts, targets;
    map<Int, array<Int,3>> def;   // value -> op, a, b
    set<Int> have; vector<Int> order;
    Int make(char op, Int a, Int b) {
        Int v = op == '+' ? a + b : op == '-' ? a - b : op == '*' ? a * b : a / b;
        if (!have.count(v)) { have.insert(v); def[v] = {op, a, b}; order.push_back(v); }
        return v;
    }
    Int mat(Dict& d, int id) { const E& x = d.e[id]; if (x.a < 0) return x.v; Int a = mat(d, x.a), b = mat(d, x.b); return make(x.op, a, b); }
    // Instructions a finished recipe really adds: distinct new values, so a value used twice or already built counts once
    // or not at all (the beam's sum counts it each time). 2026-11-18, scored_0887: 10/5 = 2, 10 - 2 = 8 reuses 10.
    int true_cost(Dict& d, Search& s, int end, int term) {
        unordered_set<Int> nv;
        auto ent = [&](auto&& self, int id) -> void { if (id < 0) return; const E& x = d.e[id]; if (x.a < 0 || have.count(x.v)) return; if (!nv.insert(x.v).second) return; self(self, x.a); self(self, x.b); };
        ent(ent, term); int steps = 0;
        for (int p = end; s.hist[p].parent >= 0; p = s.hist[p].parent) {
            const St& h = s.hist[p]; ent(ent, h.f); ent(ent, h.a);
            steps += (h.op == 'p' || h.op == 'q') ? 2 : (h.f >= 0) + (h.a >= 0);
        }
        return (int)nv.size() + steps;
    }
    void pick(Dict& d, Search& s) {
        if (SLACK == 0 || s.cands.size() < 2) return;
        int bt = 1 << 30;
        for (auto& c : s.cands) if (c[0] <= s.best + SLACK) { int t = true_cost(d, s, c[1], c[2]); if (t < bt) { bt = t; s.end = c[1]; s.term = c[2]; } }
    }
    Int unwind(Dict& d, Search& s) {   // rebuild the program from a finished search
        pick(d, s);
        Int v = mat(d, s.term);
        for (int p = s.end; s.hist[p].parent >= 0; p = s.hist[p].parent) {
            const St& h = s.hist[p];
            if (h.op == 'p' || h.op == 'q') { Int w = make('*', mat(d, h.f), mat(d, h.a)); v = h.op == 'p' ? make('+', v, w) : make('-', w, v); continue; }
            if (h.f >= 0) v = make('*', v, mat(d, h.f));
            if (h.a >= 0) { Int a = mat(d, h.a); v = h.op == 'r' ? make('-', a, v) : make(h.op, v, a); }
        }
        return v;
    }
    Int binary(Int v) { if (have.count(v)) return v; Int h = binary(v / 2); Int t = make('+', h, h); return v % 2 ? make('+', t, 1) : t; }
    void solve() {
        for (Int s : starts) have.insert(s);
        make('/', starts[0], starts[0]);
        Dict d; d.lim = lim; vector<Int> done;
        auto build = [&]() {
            vector<Int> leaves; unordered_set<Int> u;
            auto add = [&](Int v) { if (u.insert(v).second) leaves.push_back(v); };
            add(1); for (Int s : starts) add(s);
            int ns = 0;   // small values first: residuals end small, where the dictionary is sparse (Fable 2026-11-18: -6.2% on 32 practice puzzles)
            for (auto it = order.rbegin(); it != order.rend() && ns < SMALL_N; ++it) if (*it < (Int(1) << SMALL_BITS) && !u.count(*it)) { add(*it); ns++; }
            for (auto it = done.rbegin(); it != done.rend() && (int)leaves.size() < N_DONE; ++it) add(*it);
            for (auto it = order.rbegin(); it != order.rend() && (int)leaves.size() < N_RECENT; ++it) add(*it);
            d.known = &have; d.build(leaves);
        };
        // Seed cheap values the next dictionary can combine: squares of the 3 largest, doublings of the 3 smallest.
        // Unused ones cost nothing: compile() emits only what the targets need. (2026-11-18: on single-start puzzles our
        // first target took 2,107 instructions over 39 puzzles against 827 public, all in the binary fallback.)
        auto grow = [&]() {
            vector<Int> cur(have.begin(), have.end()); sort(cur.begin(), cur.end());
            int n = 0; for (auto it = cur.rbegin(); it != cur.rend() && n < 3; ++it) if (*it > 1 && *it <= lim / *it) { make('*', *it, *it); n++; }
            n = 0; for (Int v : cur) if (n < 3 && v <= lim / 2) { make('+', v, v); n++; }
        };
        auto realize = [&](Search& s, Int T) {
            if (s.end < 0) { binary(T); return; }
            Int v = unwind(d, s);
            if (v != T) { fprintf(stderr, "reconstruction failed\n"); exit(1); }
        };
        vector<Int> rest(targets.begin(), targets.end());
        for (int pick = 0; !rest.empty(); pick++) {
            int k = 0;
            if (!have.count(rest[0])) {
                build();
                if (pick < HEAD) {   // the first targets: build the cheapest one first, the dictionary is tiny (Fable idea 2)
                    int bc = 1 << 30;
                    for (int c = 0; c < (int)rest.size(); c++) {
                        if (have.count(rest[c])) { k = c; bc = -1; break; }
                        Search t(d); t.run(rest[c]); int cost = t.end < 0 ? (1 << 29) : t.best;
                        if (cost < bc) { bc = cost; k = c; }
                    }
                }
                Int T = rest[k];
                if (!have.count(T)) for (int g = 0; ; g++) {   // no recipe from a tiny dictionary (single-start puzzles): grow it, not binary
                    Search s(d); s.run(T);
                    if (s.end >= 0 || g == GROW) { realize(s, T); break; }
                    grow(); build();
                }
            }
            done.push_back(rest[k]); rest.erase(rest.begin() + k);
        }
    }
    // Values the program needs: every target's dependencies, optionally without target `skip` (starts excluded).
    set<Int> closure(Int skip, bool use_skip) {
        set<Int> need; set<Int> st(starts.begin(), starts.end()); vector<Int> stack;
        for (Int t : targets) if (!(use_skip && t == skip) && !st.count(t)) stack.push_back(t);
        while (!stack.empty()) {
            Int v = stack.back(); stack.pop_back();
            if (st.count(v) || !need.insert(v).second) continue;
            auto& a = def.at(v); stack.push_back(a[1]); stack.push_back(a[2]);
        }
        return need;
    }
    // Ruin and recreate (Fable idea 3; Santa 2025 2nd place, Santa 2019 LNS): drop the values only target T needs,
    // rebuild T with leaves from everything the rest of the program holds, keep it only if the program got shorter.
    int improve(int rounds) {
        int gained = 0; Dict d; d.lim = lim;
        for (int r = 0; r < rounds; r++) {
            int before = gained;
            for (Int T : targets) {
                if (count(starts.begin(), starts.end(), T)) continue;
                set<Int> rest = closure(T, true);
                if (rest.count(T)) continue;                       // another target's chain makes T anyway
                int need0 = (int)closure(0, false).size();
                int old = (int)compile().size();   // the score: compiled length, rebuilds included (closure size missed register rebuilds, 2026-11-18)
                auto def0 = def; auto have0 = have; auto order0 = order;
                for (auto it = def.begin(); it != def.end();) { if (!rest.count(it->first)) { have.erase(it->first); it = def.erase(it); } else ++it; }
                order.clear(); for (Int v : order0) if (have.count(v)) order.push_back(v);
                vector<Int> pool(rest.begin(), rest.end()); for (Int s0 : starts) pool.push_back(s0);
                vector<Int> leaves; unordered_set<Int> u; auto add = [&](Int v) { if (u.insert(v).second) leaves.push_back(v); };
                if (have.count(1)) add(1);
                for (Int s0 : starts) add(s0);
                sort(pool.begin(), pool.end());
                int ns = 0; for (Int v : pool) if (v < (Int(1) << SMALL_BITS) && ns < SMALL_N) { add(v); ns++; }
                double lt = log2((double)T);   // then the pool values closest to T in size
                sort(pool.begin(), pool.end(), [&](Int x, Int y) { return fabs(log2((double)x) - lt) < fabs(log2((double)y) - lt); });
                for (Int v : pool) { if ((int)leaves.size() >= N_RECENT) break; add(v); }
                d.known = &have; d.build(leaves); Search s(d); s.run(T);
                bool ok = false;
                if (s.end >= 0 && s.best < need0 - (int)rest.size()) {
                    Int v = unwind(d, s);
                    ok = v == T && (int)compile().size() < old;
                }
                if (ok) gained += old - (int)compile().size();
                else { def = def0; have = have0; order = order0; }
            }
            if (gained == before) break;
        }
        return gained;
    }
    // Pair ruin (2026-11-18): one-target ruin finds nothing on the public programs (141,432 -> 141,431), maybe because their
    // targets share chains, so few values belong to one target alone. Drop the values only T and its partner need (the
    // target sharing most of T's dependencies), rebuild both, smaller first, keep it only if the compiled program got shorter.
    set<Int> deps(Int t) {
        set<Int> need; set<Int> st(starts.begin(), starts.end()); vector<Int> stack{t};
        while (!stack.empty()) { Int v = stack.back(); stack.pop_back(); if (st.count(v) || !need.insert(v).second) continue; auto& a = def.at(v); stack.push_back(a[1]); stack.push_back(a[2]); }
        return need;
    }
    bool rebuild(Dict& d, Int T) {
        vector<Int> pool(have.begin(), have.end());
        vector<Int> leaves; unordered_set<Int> u; auto add = [&](Int v) { if (u.insert(v).second) leaves.push_back(v); };
        if (have.count(1)) add(1);
        for (Int s0 : starts) add(s0);
        sort(pool.begin(), pool.end());
        int ns = 0; for (Int v : pool) if (v < (Int(1) << SMALL_BITS) && ns < SMALL_N) { add(v); ns++; }
        double lt = log2((double)T);
        sort(pool.begin(), pool.end(), [&](Int x, Int y) { return fabs(log2((double)x) - lt) < fabs(log2((double)y) - lt); });
        for (Int v : pool) { if ((int)leaves.size() >= N_RECENT) break; add(v); }
        d.known = &have; d.build(leaves); Search s(d); s.run(T);
        if (s.end < 0) { binary(T); return have.count(T); }
        Int v = unwind(d, s);
        return v == T;
    }
    int improve_pairs(int rounds) {
        int gained = 0; Dict d; d.lim = lim; set<Int> st(starts.begin(), starts.end());
        for (int r = 0; r < rounds; r++) {
            int before = gained;
            for (Int T : targets) {
                if (st.count(T)) continue;
                auto dT = deps(T); Int U = -1; size_t bo = 0;
                for (Int t : targets) {
                    if (t == T || st.count(t)) continue;
                    auto dt = deps(t); size_t o = 0; for (Int v : dt) o += dT.count(v);
                    if (o > bo) { bo = o; U = t; }
                }
                if (U < 0) continue;
                set<Int> keep; vector<Int> stack;   // what every other target needs
                for (Int t : targets) if (t != T && t != U && !st.count(t)) stack.push_back(t);
                while (!stack.empty()) { Int v = stack.back(); stack.pop_back(); if (st.count(v) || !keep.insert(v).second) continue; auto& a = def.at(v); stack.push_back(a[1]); stack.push_back(a[2]); }
                if (keep.count(T) || keep.count(U)) continue;
                int old = (int)compile().size();
                auto def0 = def; auto have0 = have; auto order0 = order;
                for (auto it = def.begin(); it != def.end();) { if (!keep.count(it->first)) { have.erase(it->first); it = def.erase(it); } else ++it; }
                order.clear(); for (Int v : order0) if (have.count(v)) order.push_back(v);
                bool ok = rebuild(d, min(T, U)) && rebuild(d, max(T, U));
                int now_ = ok ? (int)compile().size() : old;
                if (ok && now_ < old) gained += old - now_;
                else { def = def0; have = have0; order = order0; }
            }
            if (gained == before) break;
        }
        return gained;
    }
    // Two schedules, the shorter compile wins: depth-first by target, and creation order (the order the values were made;
    // for a replayed public program that is its own order, which it compiles back to exactly, 2026-11-18).
    vector<array<int,4>> compile() {
        set<Int> placed(starts.begin(), starts.end()); vector<Int> seq;
        auto visit = [&](auto&& self, Int v) -> void {
            if (!placed.insert(v).second) return;
            auto& a = def.at(v); self(self, a[1]); self(self, a[2]); seq.push_back(v); };
        for (Int t : targets) visit(visit, t);
        set<Int> need(seq.begin(), seq.end()); vector<Int> made;
        for (Int v : order) if (need.count(v)) made.push_back(v);
        vector<array<int,4>> best;
        for (auto* q : {&seq, &made}) for (bool fs : {false, true}) {
            auto p = schedule(*q, fs);
            if (!p.empty() && (best.empty() || p.size() < best.size())) best = p;
        }
        return best;
    }
    // free_seeds: a start value with no further use may be overwritten; empty result if a later rebuild then needs it.
    vector<array<int,4>> schedule(const vector<Int>& seq, bool free_seeds) {
        map<Int, vector<int>> uses;   // reversed list of use positions; back() = next use
        for (int i = (int)seq.size() - 1; i >= 0; i--) { auto& a = def.at(seq[i]); uses[a[1]].push_back(i); if (a[2] != a[1]) uses[a[2]].push_back(i); }
        array<Int,48> bank{}; for (size_t i = 0; i < starts.size(); i++) bank[i] = starts[i];
        set<Int> seeds(starts.begin(), starts.end());
        vector<array<int,4>> prog; int now = 0;
        bool lost = false;
        auto emit = [&](auto&& self, Int v, uint64_t lock) -> int {
            for (int k = 0; k < 48; k++) if (bank[k] == v) return k;
            if (seeds.count(v)) { lost = true; return 0; }
            auto& a = def.at(v);
            int ra = self(self, a[1], lock); int rb = self(self, a[2], lock | (1ULL << ra));
            int rd = -1; long far = -1;
            for (int k = 0; k < 48; k++) {
                if (lock >> k & 1) continue;
                Int old = bank[k];
                if (!old) { rd = k; break; }
                auto& us = uses[old];
                while (!us.empty() && us.back() < now) us.pop_back();
                if (k == ra || k == rb) {   // an operand's register may take the result once the operand is dead after this step
                    if (v != seq[now] || (seeds.count(old) && !free_seeds)) continue;   // not inside a rebuild
                    if (!(us.empty() || (us.size() == 1 && us.back() == now))) continue;
                    if (far < (1L << 40)) { far = 1L << 40; rd = k; }
                    continue;
                }
                if (seeds.count(old) && (!free_seeds || !us.empty())) continue;   // a start value cannot be rebuilt
                long nu = us.empty() ? (1L << 40) : us.back();
                if (nu > far) { far = nu; rd = k; }
            }
            if (rd < 0) { // all locked or seeds: overwrite a source that is not locked
                for (int k = 0; k < 48; k++) if (!(lock >> k & 1) && !seeds.count(bank[k])) { rd = k; break; }
            }
            bank[rd] = v; prog.push_back({(int)a[0], ra, rb, rd}); return rd;
        };
        for (int i = 0; i < (int)seq.size() && !lost; i++) { now = i; emit(emit, seq[i], 0); }
        return lost ? vector<array<int,4>>{} : prog;
    }
};

#ifndef RUDOLPH_NO_MAIN
int main(int argc, char** argv) {
    if (argc > 1) WIDTH = atoi(argv[1]); if (argc > 2) ALPHA = atof(argv[2]);
    if (argc > 3) N_DONE = atoi(argv[3]); if (argc > 4) N_RECENT = atoi(argv[4]);
    if (argc > 5) SMALL_BITS = atoi(argv[5]); if (argc > 6) SMALL_N = atoi(argv[6]); if (argc > 7) HEAD = atoi(argv[7]); if (argc > 8) RR_ROUNDS = atoi(argv[8]); if (argc > 9) PAIR_ROUNDS = atoi(argv[9]); if (argc > 10) PROD_K = atoi(argv[10]); if (argc > 11) GROW = atoi(argv[11]); if (argc > 12) SLACK = atoi(argv[12]);
    string line;
    while (getline(cin, line)) {
        if (line.empty()) continue;
        istringstream in(line); string id; Puzzle P; int ns, nt; in >> id >> P.lim >> ns >> nt;
        P.starts.resize(ns); P.targets.resize(nt);
        for (auto& x : P.starts) in >> x; for (auto& x : P.targets) in >> x;
        P.solve(); P.improve(RR_ROUNDS); P.improve_pairs(PAIR_ROUNDS); auto prog = P.compile();
        string out = "[";
        for (size_t i = 0; i < prog.size(); i++) {
            char buf[64]; snprintf(buf, sizeof buf, "%s[\"%c\",%d,%d,%d]", i ? "," : "", (char)prog[i][0], prog[i][1], prog[i][2], prog[i][3]);
            out += buf;
        }
        out += "]";
        printf("%s %s\n", id.c_str(), out.c_str()); fflush(stdout);
    }
}
#endif
