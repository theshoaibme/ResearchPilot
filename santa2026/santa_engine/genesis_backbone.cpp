// GENESIS_BACKBONE: Hub-Cluster Subgraph Ruin (k = 3..9 Targets + Shared Hubs!) & Cheapest-First Generator Reconstruction
// Breaks frozen global backbones that 1/2/3-target LNS cannot move:
//   Mode 1 (Hub-Cluster Subgraph Ruin on Frontier DAGs):
//     Identifies internal intermediate hubs H feeding k in [3..9] targets. Drops ALL k targets AND their entire
//     exclusive subgraph (including H and all semi-global sub-hubs, often 15..55 nodes at once!), then runs
//     multi-trajectory collective backward-beam re-synthesis (cheapest-first + randomized target orders + multi-alpha)
//     where each rebuilt target immediately expands S0(cost=0) & S1(cost=1) for the remaining ruined targets!
//   Mode 2 (Bottom-Up Generator Reconstruction from Scratch for Structured Puzzles):
//     Reconstructs secret generator trees bottom-up via dynamic ALL-target cheapest-next lookahead + inter-target
//     delta/GCD seeding.

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
using namespace std;

using u64 = uint64_t;
using u128 = __uint128_t;
using i64 = int64_t;

struct Rng {
    u64 s = 0x9e3779b97f4a7c15ULL;
    inline u64 next() {
        u64 z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    inline u64 operator()(u64 n) { return next() % n; }
};

static inline double now_sec() {
    using clk = chrono::steady_clock;
    static const auto t0 = clk::now();
    return chrono::duration<double>(clk::now() - t0).count();
}

static inline u64 mulmod(u64 a, u64 b, u64 m) { return (u128)a * b % m; }
static u64 powmod(u64 a, u64 e, u64 m) {
    u64 r = 1;
    while (e) { if (e & 1) r = mulmod(r, a, m); a = mulmod(a, a, m); e >>= 1; }
    return r;
}
static bool is_prime64(u64 n) {
    if (n < 2) return false;
    for (u64 p : {2ULL,3ULL,5ULL,7ULL,11ULL,13ULL,17ULL,19ULL,23ULL,29ULL,31ULL,37ULL}) {
        if (n == p) return true;
        if (n % p == 0) return false;
    }
    u64 d = n - 1, s = 0;
    while ((d & 1) == 0) { d >>= 1; s++; }
    for (u64 a : {2ULL,325ULL,9375ULL,28178ULL,450775ULL,9780504ULL,1795265022ULL}) {
        if (a % n == 0) continue;
        u64 x = powmod(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool comp = true;
        for (u64 r = 1; r < s; r++) {
            x = mulmod(x, x, n);
            if (x == n - 1) { comp = false; break; }
        }
        if (comp) return false;
    }
    return true;
}
static u64 pollard64(u64 n, Rng& rng, int max_steps = 5000) {
    if ((n & 1) == 0) return 2;
    for (int attempt = 0; attempt < 3; attempt++) {
        u64 c = rng(n - 1) + 1, y = rng(n - 1) + 1;
        auto f = [&](u64 x) { return (mulmod(x, x, n) + c) % n; };
        u64 g = 1, q = 1, r = 1, x = y, ys = y;
        int steps = 0;
        const u64 m = 64;
        while (g == 1 && steps < max_steps) {
            x = y;
            for (u64 i = 0; i < r; i++) y = f(y);
            u64 k = 0;
            while (k < r && g == 1 && steps < max_steps) {
                ys = y;
                u64 lim = min(m, r - k);
                for (u64 i = 0; i < lim; i++) {
                    y = f(y);
                    q = mulmod(q, x > y ? x - y : y - x, n);
                    steps++;
                }
                g = std::gcd(q, n);
                k += m;
            }
            r <<= 1;
        }
        if (g == n) {
            do { ys = f(ys); g = std::gcd(x > ys ? x - ys : ys - x, n); } while (g == 1);
        }
        if (g > 1 && g < n) return g;
    }
    return n;
}
static void factor_divs(u64 n, Rng& rng, vector<u64>& divs) {
    divs.clear();
    if (n <= 3) return;
    static const u64 SP[] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97};
    vector<u64> fs;
    u64 tmp = n;
    for (u64 p : SP) while (tmp % p == 0) { fs.push_back(p); tmp /= p; }
    if (tmp > 1) {
        vector<u64> st = {tmp};
        int budget = 12;
        while (!st.empty() && budget-- > 0) {
            u64 x = st.back(); st.pop_back();
            if (x <= 1) continue;
            if (is_prime64(x)) { fs.push_back(x); continue; }
            u64 d = pollard64(x, rng, 4500);
            if (d == x) { fs.push_back(x); continue; }
            st.push_back(d); st.push_back(x / d);
        }
    }
    sort(fs.begin(), fs.end());
    vector<pair<u64, int>> pf;
    for (u64 p : fs) {
        if (pf.empty() || pf.back().first != p) pf.push_back({p, 1});
        else pf.back().second++;
    }
    divs.push_back(1);
    for (auto& [p, e] : pf) {
        size_t csz = divs.size();
        u64 mult = 1;
        for (int k = 1; k <= e && divs.size() < 256; k++) {
            mult *= p;
            for (size_t j = 0; j < csz && divs.size() < 256; j++) divs.push_back(divs[j] * mult);
        }
    }
}

struct Hash {
    vector<u64> k;
    vector<int> id;
    vector<size_t> used;
    size_t mask;
    explicit Hash(int bits = 20) : k(1ULL << bits, 0), id(1ULL << bits, -1), mask((1ULL << bits) - 1) {}
    inline size_t slot(u64 v) const {
        u64 h = v * 0x9e3779b97f4a7c15ULL;
        h ^= h >> 29;
        size_t i = h & mask;
        while (k[i] && k[i] != v) i = (i + 1) & mask;
        return i;
    }
    inline int get(u64 v) const {
        if (!v) return -1;
        size_t i = slot(v);
        return k[i] ? id[i] : -1;
    }
    inline bool put(u64 v, int x) {
        if (!v || used.size() * 4 > k.size() * 3) return false;
        size_t i = slot(v);
        if (k[i]) return false;
        k[i] = v; id[i] = x; used.push_back(i);
        return true;
    }
    void clear() {
        for (size_t i : used) k[i] = 0;
        used.clear();
    }
};

struct E { u64 v; int a, b; char op; int cost; };

struct Dict {
    u64 lim = 0;
    Hash h{21}; // 2M slots for S0 + full S1 + rich S2 hash entries!
    vector<E> e;
    vector<pair<u64, int>> sorted0, sorted;
    vector<int> smallS1;
    int n0 = 0, n1 = 0, n2 = 0;

    inline void add(u64 v, int a, int b, char op, int c) {
        if (v > 0 && v <= lim && h.put(v, (int)e.size())) e.push_back({v, a, b, op, c});
    }
    inline void comb(int i, int j, int c) {
        u64 x = e[i].v, y = e[j].v;
        if (x < y) { swap(x, y); swap(i, j); }
        if (x <= lim - y) add(x + y, i, j, '+', c);
        if (x > y) add(x - y, i, j, '-', c);
        if (y > 1 && x <= lim / y) add(x * y, i, j, '*', c);
        if (y > 1 && x % y == 0) add(x / y, i, j, '/', c);
    }

    void buildFullS1(const vector<u64>& knownVals, bool includeS2 = true) {
        h.clear(); e.clear(); sorted0.clear(); sorted.clear(); smallS1.clear();
        for (u64 v : knownVals) add(v, -1, -1, 0, 0);
        n0 = (int)e.size();
        for (int i = 0; i < n0; i++) {
            for (int j = 0; j <= i; j++) comb(i, j, 1);
        }
        n1 = (int)e.size();
        sorted0.reserve(n0);
        for (int i = 0; i < n0; i++) sorted0.push_back({e[i].v, i});
        sort(sorted0.begin(), sorted0.end());

        sorted.reserve(n1);
        for (int i = 0; i < n1; i++) {
            sorted.push_back({e[i].v, i});
            if (i >= n0 && e[i].v >= 2 && e[i].v < (1ULL << 21)) smallS1.push_back(i);
        }
        sort(sorted.begin(), sorted.end());

        // Fast S2 hash expansion (no sort overhead -> O(1) 2-step remainder & terminal hits!)
        if (includeS2 && n1 > n0) {
            for (int i = n0; i < n1; i++) comb(i, i, 2); // x+x, x*x for all S1!
            int anchorStep = max(1, n0 / 32);
            int s1Lim = min((int)smallS1.size(), 1400);
            for (int si = 0; si < s1Lim && e.size() < 360000; si++) {
                int id1 = smallS1[si];
                for (int j = 0; j < n0; j += anchorStep) comb(id1, j, 2);
            }
        }
        n2 = (int)e.size();
    }
};

struct St {
    u64 v;
    int cost, parent, f, a;
    char op;
    double rank;
    array<int, 8> used{-1, -1, -1, -1, -1, -1, -1, -1};
    uint8_t nUsed = 0;
};

struct BeamSearch {
    Dict& d;
    Rng& rng;
    int width = 80;
    int prodK = 5;
    int slack = 2;
    double alpha = 0.18;
    int best = 1 << 29, end = -1, term = -1;
    vector<St> hist, nxt;
    unordered_map<u64, int> seen;
    vector<array<int, 3>> cands;

    BeamSearch(Dict& d_, Rng& rng_, int w = 80, int pk = 5, int sl = 2, double al = 0.18)
        : d(d_), rng(rng_), width(w), prodK(pk), slack(sl), alpha(al) {}

    // Exact incremental creation cost of dictionary entry `id` given already-created `used` IDs along path!
    inline int addDictCreationCost(array<int, 8>& u, uint8_t& nu, int id) const {
        if (id < 0) return 0;
        const E& x = d.e[id];
        if (x.cost == 0) return 0;
        for (uint8_t k = 0; k < nu; k++) if (u[k] == id) return 0; // ALREADY CREATED ON THIS PATH!
        int add = 1;
        if (x.cost == 2) {
            add += addDictCreationCost(u, nu, x.a);
            if (x.b != x.a) add += addDictCreationCost(u, nu, x.b);
        }
        if (nu < 8) u[nu++] = id;
        return add;
    }

    void offer(int p, u64 c, int f, int a, char op, int maxAllowed) {
        if (c == 0 || c > d.lim) return;
        array<int, 8> u = hist[p].used;
        uint8_t nu = hist[p].nUsed;
        int stepOps = (op == 'p' || op == 'q') ? 2 : ((f >= 0 ? 1 : 0) + (a >= 0 ? 1 : 0));
        int cost = hist[p].cost + stepOps + addDictCreationCost(u, nu, f) + addDictCreationCost(u, nu, a);
        if (cost > maxAllowed || cost >= best + slack) return;
        int t = d.h.get(c);
        if (t >= 0) {
            array<int, 8> ut = u;
            uint8_t nut = nu;
            int tot = cost + addDictCreationCost(ut, nut, t);
            if (tot <= maxAllowed && tot <= best + slack) {
                int hidx = (int)hist.size();
                St termSt{c, cost, p, f, a, op, 0.0, ut, nut};
                hist.push_back(termSt);
                cands.push_back({tot, hidx, t});
                if (tot < best) { best = tot; end = hidx; term = t; }
            }
            return;
        }
        if (cost + 1 > maxAllowed || cost + 1 >= best + slack || c >= hist[p].v) return;
        double rk = cost + alpha * log2((double)c);
        if ((int)nxt.size() == width && rk >= nxt.back().rank) return;
        auto s = seen.find(c);
        if (s != seen.end() && s->second <= cost) return;
        for (auto it = nxt.begin(); it != nxt.end(); ++it) {
            if (it->v == c) {
                if (it->cost <= cost) return;
                nxt.erase(it);
                break;
            }
        }
        St st{c, cost, p, f, a, op, rk, u, nu};
        auto it = upper_bound(nxt.begin(), nxt.end(), rk, [](double r, const St& x) { return r < x.rank; });
        nxt.insert(it, st);
        if ((int)nxt.size() > width) nxt.pop_back();
    }

    void expand(int p, int depth, int maxAllowed) {
        u64 v = hist[p].v;
        auto it0 = lower_bound(d.sorted0.begin(), d.sorted0.end(), make_pair(v, -1));
        for (int k = -6; k < 6; k++) {
            i64 pos = (it0 - d.sorted0.begin()) + k;
            if (pos < 0 || pos >= (i64)d.sorted0.size()) continue;
            auto [a, id] = d.sorted0[pos];
            if (a == v) continue;
            offer(p, v > a ? v - a : a - v, -1, id, v > a ? '+' : 'r', maxAllowed);
        }
        auto it1 = lower_bound(d.sorted.begin(), d.sorted.end(), make_pair(v, -1));
        for (int k = -8; k < 8; k++) {
            i64 pos = (it1 - d.sorted.begin()) + k;
            if (pos < 0 || pos >= (i64)d.sorted.size()) continue;
            auto [a, id] = d.sorted[pos];
            if (a == v) continue;
            offer(p, v > a ? v - a : a - v, -1, id, v > a ? '+' : 'r', maxAllowed);
        }

        for (int i = 0; i < d.n0; i++) {
            u64 f = d.e[i].v;
            if (f < 2 || f > v) continue;
            u64 q = v / f, r = v % f;
            if (prodK > 0) {
                auto iy0 = lower_bound(d.sorted0.begin(), d.sorted0.end(), make_pair(q, -1));
                for (int k = -3; k <= 3; k++) {
                    i64 pos = (iy0 - d.sorted0.begin()) + k;
                    if (pos < 0 || pos >= (i64)d.sorted0.size()) continue;
                    auto [y, yi] = d.sorted0[pos];
                    if (y < 2 || y > d.lim / f) continue;
                    u64 w = f * y;
                    if (w != v) offer(p, w < v ? v - w : w - v, i, yi, w < v ? 'p' : 'q', maxAllowed);
                }
                auto iy = lower_bound(d.sorted.begin(), d.sorted.end(), make_pair(q, -1));
                for (int k = -prodK; k <= prodK; k++) {
                    i64 pos = (iy - d.sorted.begin()) + k;
                    if (pos < 0 || pos >= (i64)d.sorted.size()) continue;
                    auto [y, yi] = d.sorted[pos];
                    if (y < 2 || y > d.lim / f) continue;
                    u64 w = f * y;
                    if (w != v) offer(p, w < v ? v - w : w - v, i, yi, w < v ? 'p' : 'q', maxAllowed);
                }
            }
            if (!r) { offer(p, q, i, -1, '+', maxAllowed); continue; }
            int ri = d.h.get(r);
            if (ri >= 0) offer(p, q, i, ri, '+', maxAllowed);
            if (f - r <= d.lim - v) {
                ri = d.h.get(f - r);
                if (ri >= 0) offer(p, q + 1, i, ri, '-', maxAllowed);
            }
        }

        if (depth == 0) {
            size_t limS1 = min<size_t>(d.smallS1.size(), 2500);
            for (size_t idx = 0; idx < limS1; idx++) {
                int i = d.smallS1[idx];
                u64 f = d.e[i].v;
                if (f > v) continue;
                u64 q = v / f, r = v % f;
                if (!r) { offer(p, q, i, -1, '+', maxAllowed); continue; }
                int ri = d.h.get(r);
                if (ri >= 0) offer(p, q, i, ri, '+', maxAllowed);
                if (f - r <= d.lim - v) {
                    ri = d.h.get(f - r);
                    if (ri >= 0) offer(p, q + 1, i, ri, '-', maxAllowed);
                }
            }
            if (v >= (1ULL << 20)) {
                vector<u64> divs;
                int nTr = min(16, d.n0 + 1);
                for (int tr = 0; tr < nTr; tr++) {
                    int offId = (tr == 0) ? -1 : (int)rng(d.n0);
                    u64 r = (tr == 0) ? 0 : d.e[offId].v;
                    u64 ns[2]; char ops[2]; int nc = 0;
                    if (r == 0) { ns[nc] = v; ops[nc++] = '+'; }
                    else {
                        if (v > r) { ns[nc] = v - r; ops[nc++] = '+'; }
                        if (v <= d.lim - r) { ns[nc] = v + r; ops[nc++] = '-'; }
                    }
                    for (int ci = 0; ci < nc; ci++) {
                        factor_divs(ns[ci], rng, divs);
                        for (u64 A : divs) {
                            if (A < 2 || A >= ns[ci]) continue;
                            int fid = d.h.get(A);
                            if (fid >= 0) offer(p, ns[ci] / A, fid, offId, ops[ci], maxAllowed);
                        }
                    }
                }
            }
        }
    }

    void run(u64 target, int maxAllowed) {
        hist.clear(); nxt.clear(); seen.clear(); cands.clear();
        best = 1 << 29; end = -1; term = -1;
        hist.push_back({target, 0, -1, -1, -1, 0, 0.0});
        int t = d.h.get(target);
        if (t >= 0) {
            best = d.e[t].cost;
            if (best <= maxAllowed) { end = 0; term = t; cands.push_back({best, 0, t}); }
            return;
        }
        vector<int> front{0};
        for (int depth = 0; depth < 22 && !front.empty(); depth++) {
            nxt.clear();
            for (int p : front) {
                if (hist[p].cost + 1 <= maxAllowed && hist[p].cost + 1 <= best + slack) {
                    expand(p, depth, maxAllowed);
                }
            }
            front.clear();
            for (auto& s : nxt) {
                if (s.cost + 1 > maxAllowed || s.cost + 1 > best + slack) continue;
                seen[s.v] = s.cost;
                front.push_back((int)hist.size());
                hist.push_back(s);
            }
        }
    }

    bool extractCone(int hIdx, int tId, const unordered_set<u64>& have, vector<pair<u64, array<u64, 3>>>& newDefs) const {
        newDefs.clear();
        unordered_set<u64> localHave = have;
        auto emitV = [&](char op, u64 a, u64 b) -> u64 {
            u64 v = op == '+' ? a + b : op == '-' ? a - b : op == '*' ? a * b : a / b;
            if (!localHave.count(v)) {
                localHave.insert(v);
                newDefs.push_back({v, {(u64)op, a, b}});
            }
            return v;
        };
        auto mat = [&](auto& self, int id) -> u64 {
            const E& x = d.e[id];
            if (x.a < 0 || localHave.count(x.v)) return x.v;
            u64 a = self(self, x.a), b = self(self, x.b);
            return emitV(x.op, a, b);
        };
        u64 v = mat(mat, tId);
        vector<int> path;
        for (int p = hIdx; hist[p].parent >= 0; p = hist[p].parent) path.push_back(p);
        for (int p : path) {
            const St& h = hist[p];
            if (h.op == 'p' || h.op == 'q') {
                u64 w = emitV('*', mat(mat, h.f), mat(mat, h.a));
                v = (h.op == 'p') ? emitV('+', v, w) : emitV('-', w, v);
                continue;
            }
            if (h.f >= 0) v = emitV('*', v, mat(mat, h.f));
            if (h.a >= 0) {
                u64 a = mat(mat, h.a);
                v = (h.op == 'r') ? emitV('-', a, v) : emitV(h.op, v, a);
            }
        }
        return v == hist[0].v;
    }
};

struct Puzzle {
    string id;
    vector<u64> start, targets;
    u64 L;
    int regs;
    vector<array<int, 4>> prog;
};

static bool readPuzzle(istream& in, Puzzle& p) {
    string tag;
    while (in >> tag) if (tag == "P") break;
    if (!in) return false;
    p = Puzzle();
    int ns, nt, np;
    in >> p.id >> ns; p.start.resize(ns);
    for (int i = 0; i < ns; i++) in >> p.start[i];
    in >> nt; p.targets.resize(nt);
    for (int i = 0; i < nt; i++) in >> p.targets[i];
    in >> p.L >> p.regs >> np; p.prog.resize(np);
    for (int i = 0; i < np; i++) {
        string op; int a, b, d;
        in >> op >> a >> b >> d;
        p.prog[i] = {(int)op[0], a, b, d};
    }
    return true;
}

static bool verifyProgExact(const Puzzle& p, const vector<array<int, 4>>& prog) {
    if (prog.size() > 4096) return false;
    vector<u64> r(p.regs, 0);
    vector<char> def(p.regs, 0);
    unordered_set<u64> need(p.targets.begin(), p.targets.end());
    for (size_t i = 0; i < p.start.size(); i++) {
        r[i] = p.start[i]; def[i] = 1; need.erase(p.start[i]);
    }
    for (auto& ins : prog) {
        int op = ins[0], i = ins[1], j = ins[2], d = ins[3];
        if (i < 0 || i >= p.regs || j < 0 || j >= p.regs || d < 0 || d >= p.regs) return false;
        if (!def[i] || !def[j]) return false;
        u64 a = r[i], b = r[j], v = 0;
        if (op == '+') { if (a > p.L - b) return false; v = a + b; }
        else if (op == '-') { if (a <= b) return false; v = a - b; }
        else if (op == '*') { if ((u128)a * b > (u128)p.L) return false; v = a * b; }
        else if (op == '/') { if (b == 0 || a % b != 0) return false; v = a / b; }
        else return false;
        if (v < 1 || v > p.L) return false;
        r[d] = v; def[d] = 1; need.erase(v);
    }
    return need.empty();
}

struct DagState {
    const Puzzle& P;
    unordered_set<u64> starts;
    map<u64, array<u64, 3>> def;
    vector<u64> order;

    DagState(const DagState&) = default;
    DagState& operator=(const DagState& o) { def = o.def; order = o.order; return *this; }
    DagState& operator=(DagState&& o) noexcept { def = std::move(o.def); order = std::move(o.order); return *this; }

    explicit DagState(const Puzzle& p) : P(p), starts(p.start.begin(), p.start.end()) {
        vector<u64> reg(p.regs, 0);
        for (size_t i = 0; i < p.start.size(); i++) reg[i] = p.start[i];
        unordered_set<u64> seen(starts);
        for (auto& ins : p.prog) {
            u64 va = reg[ins[1]], vb = reg[ins[2]];
            u64 v = ins[0] == '+' ? va + vb : ins[0] == '-' ? va - vb : ins[0] == '*' ? va * vb : va / vb;
            reg[ins[3]] = v;
            if (seen.insert(v).second) {
                def[v] = {(u64)ins[0], va, vb};
                order.push_back(v);
            }
        }
    }

    unordered_set<u64> closureExcept(const unordered_set<u64>& skipTargets) const {
        unordered_set<u64> need;
        vector<u64> st;
        for (u64 t : P.targets) if (!skipTargets.count(t) && !starts.count(t)) st.push_back(t);
        while (!st.empty()) {
            u64 v = st.back(); st.pop_back();
            if (starts.count(v) || !need.insert(v).second) continue;
            auto it = def.find(v);
            if (it != def.end()) { st.push_back(it->second[1]); st.push_back(it->second[2]); }
        }
        return need;
    }

    void pruneToTargets() {
        auto keep = closureExcept({});
        for (auto it = def.begin(); it != def.end();) {
            if (!keep.count(it->first)) it = def.erase(it);
            else ++it;
        }
        vector<u64> newOrd;
        unordered_set<u64> placed(starts);
        auto visit = [&](auto& self, u64 v) -> void {
            if (!placed.insert(v).second) return;
            auto& a = def.at(v);
            self(self, a[1]); self(self, a[2]);
            newOrd.push_back(v);
        };
        for (u64 v : order) if (keep.count(v)) visit(visit, v);
        for (u64 t : P.targets) if (!starts.count(t)) visit(visit, t);
        order = std::move(newOrd);
    }

    vector<array<int, 4>> compileBest(Rng& rng) const {
        unordered_set<u64> keep = closureExcept({});
        vector<u64> dfsSeq, ordSeq;
        {
            unordered_set<u64> placed(starts);
            auto visit = [&](auto& self, u64 v) -> void {
                if (!placed.insert(v).second) return;
                auto& a = def.at(v);
                self(self, a[1]); self(self, a[2]);
                dfsSeq.push_back(v);
            };
            for (u64 t : P.targets) visit(visit, t);
        }
        for (u64 v : order) if (keep.count(v)) ordSeq.push_back(v);

        auto tryScheduleSeq = [&](const vector<u64>& seq, bool freeSeeds) -> vector<array<int, 4>> {
            unordered_map<u64, vector<int>> uses;
            for (int i = (int)seq.size() - 1; i >= 0; i--) {
                auto& a = def.at(seq[i]);
                uses[a[1]].push_back(i);
                if (a[2] != a[1]) uses[a[2]].push_back(i);
            }
            vector<u64> bank(P.regs, 0);
            for (size_t i = 0; i < P.start.size(); i++) bank[i] = P.start[i];
            vector<array<int, 4>> prog;
            int now = 0;
            bool lost = false;
            auto emit = [&](auto& self, u64 v, u64 lock) -> int {
                for (int k = 0; k < P.regs; k++) if (bank[k] == v) return k;
                if (starts.count(v)) { lost = true; return 0; }
                auto& a = def.at(v);
                int ra = self(self, a[1], lock);
                int rb = self(self, a[2], lock | (1ULL << ra));
                int rd = -1; long far = -1;
                for (int k = 0; k < P.regs; k++) {
                    if ((lock >> k) & 1) continue;
                    u64 old = bank[k];
                    if (!old) { rd = k; break; }
                    auto& us = uses[old];
                    while (!us.empty() && us.back() < now) us.pop_back();
                    if (k == ra || k == rb) {
                        if (v != seq[now] || (starts.count(old) && !freeSeeds)) continue;
                        if (!(us.empty() || (us.size() == 1 && us.back() == now))) continue;
                        if (far < (1L << 40)) { far = 1L << 40; rd = k; }
                        continue;
                    }
                    if (starts.count(old) && (!freeSeeds || !us.empty())) continue;
                    long nu = us.empty() ? (1L << 40) : us.back();
                    if (nu > far) { far = nu; rd = k; }
                }
                if (rd < 0) {
                    for (int k = 0; k < P.regs; k++) if (!((lock >> k) & 1) && !starts.count(bank[k])) { rd = k; break; }
                }
                if (rd < 0) { lost = true; return 0; }
                bank[rd] = v;
                prog.push_back({(int)a[0], ra, rb, rd});
                return rd;
            };
            for (int i = 0; i < (int)seq.size() && !lost; i++) { now = i; emit(emit, seq[i], 0); }
            if (lost || !verifyProgExact(P, prog)) return {};
            return prog;
        };

        vector<array<int, 4>> best;
        for (const auto* sq : {&dfsSeq, &ordSeq}) {
            for (bool fs : {false, true}) {
                auto p = tryScheduleSeq(*sq, fs);
                if (!p.empty() && (best.empty() || p.size() < best.size())) best = p;
            }
        }
        for (int att = 0; att < 12 && (best.empty() || best.size() > keep.size()); att++) {
            unordered_map<u64, int> remU, indeg;
            unordered_map<u64, vector<u64>> adj;
            for (u64 v : keep) {
                auto& a = def.at(v);
                remU[a[1]]++; if (a[2] != a[1]) remU[a[2]]++;
                if (!starts.count(a[1])) { adj[a[1]].push_back(v); indeg[v]++; }
                if (a[2] != a[1] && !starts.count(a[2])) { adj[a[2]].push_back(v); indeg[v]++; }
            }
            vector<u64> rdy, gSeq;
            for (u64 v : keep) if (indeg[v] == 0) rdy.push_back(v);
            while (!rdy.empty()) {
                int pick = 0; double bestSc = -1e18;
                for (int i = 0; i < (int)rdy.size(); i++) {
                    u64 x = rdy[i]; auto& a = def.at(x);
                    int freed = (remU[a[1]] == 1 ? 1 : 0) + (a[2] != a[1] && remU[a[2]] == 1 ? 1 : 0);
                    int creates = remU[x] > 0 ? 1 : 0;
                    double sc = (freed - creates) * 100.0 + (remU[x] == 0 ? 50.0 : -remU[x] * 3.0) + (rng(1000) * 0.015);
                    if (sc > bestSc) { bestSc = sc; pick = i; }
                }
                u64 x = rdy[pick]; rdy[pick] = rdy.back(); rdy.pop_back();
                gSeq.push_back(x);
                auto& a = def.at(x);
                remU[a[1]]--; if (a[2] != a[1]) remU[a[2]]--;
                for (u64 ch : adj[x]) if (--indeg[ch] == 0) rdy.push_back(ch);
            }
            if (gSeq.size() == keep.size()) {
                for (bool fs : {true, false}) {
                    auto p = tryScheduleSeq(gSeq, fs);
                    if (!p.empty() && (best.empty() || p.size() < best.size())) best = p;
                }
            }
        }
        return best;
    }
};

using Cone = vector<pair<u64, array<u64, 3>>>;

int main(int argc, char** argv) {
    double secsPerPuzzle = 8.0;
    int width = 88, prodK = 5, doFromScratch = 1, nEarthquake = 0;
    u64 seed = 20261128555ULL;
    string seedOutFile;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "--time") secsPerPuzzle = atof(argv[++i]);
        else if (a == "--width") width = atoi(argv[++i]);
        else if (a == "--prodk") prodK = atoi(argv[++i]);
        else if (a == "--scratch") doFromScratch = atoi(argv[++i]);
        else if (a == "--earthquake") nEarthquake = atoi(argv[++i]);
        else if (a == "--seed") seed = strtoull(argv[++i], nullptr, 10);
        else if (a == "--seed-out") seedOutFile = argv[++i];
    }

    ofstream seedOf;
    if (!seedOutFile.empty()) seedOf.open(seedOutFile, ios::app);

    ios::sync_with_stdio(false);
    Puzzle P;
    u64 pIdx = 0;
    const double alphas[4] = {0.18, 0.14, 0.22, 0.16};

    while (readPuzzle(cin, P)) {
        if (!verifyProgExact(P, P.prog)) continue;
        double t0 = now_sec();
        Rng rng(seed ^ (hash<string>{}(P.id) + (++pIdx) * 13337ULL));
        DagState st(P);
        st.pruneToTargets();
        int origLen = (int)P.prog.size();
        vector<array<int, 4>> bestProg = P.prog;
        int bestLen = origLen;
        int minClosure = (int)st.def.size();
        vector<u64> minClosureVals = st.order;

        auto acceptNewState = [&](DagState&& candSt) -> bool {
            candSt.pruneToTargets();
            bool shrunk = false;
            if ((int)candSt.def.size() < minClosure) {
                minClosure = (int)candSt.def.size();
                minClosureVals = candSt.order;
            }
            auto cProg = candSt.compileBest(rng);
            if (!cProg.empty() && (int)cProg.size() < bestLen) {
                bestLen = (int)cProg.size();
                bestProg = std::move(cProg);
                st = std::move(candSt);
                shrunk = true;
            } else if ((int)candSt.def.size() < (int)st.def.size()) {
                st = std::move(candSt);
                shrunk = true;
            }
            return shrunk;
        };

        Dict dict;
        dict.lim = P.L;

        // ==================== MODE 2: BOTTOM-UP CHEAPEST-FIRST GENERATOR RECONSTRUCTION ====================
        // Especially potent on Structured puzzles (starts <= 32) where targets come from a hidden generator DAG!
        u64 maxStart = 0;
        for (u64 s : P.start) maxStart = max(maxStart, s);
        if (doFromScratch && maxStart <= 32 && P.targets.size() >= 6 && now_sec() - t0 < secsPerPuzzle * 0.35) {
            for (int trial = 0; trial < 2 && now_sec() - t0 < secsPerPuzzle * 0.35; trial++) {
                DagState scr(P);
                scr.def.clear(); scr.order.clear();
                unordered_set<u64> have = scr.starts;
                auto addOp = [&](char op, u64 a, u64 b) -> u64 {
                    u64 v = op == '+' ? a + b : op == '-' ? a - b : op == '*' ? a * b : a / b;
                    if (!have.count(v)) {
                        have.insert(v);
                        scr.def[v] = {(u64)op, a, b};
                        scr.order.push_back(v);
                    }
                    return v;
                };
                if (!have.count(1)) addOp('/', P.start[0], P.start[0]);

                vector<u64> remT;
                for (u64 t : P.targets) if (!have.count(t)) remT.push_back(t);
                sort(remT.begin(), remT.end());
                remT.erase(unique(remT.begin(), remT.end()), remT.end());

                bool aborted = false;
                double al = (trial == 0) ? 0.18 : 0.15;
                while (!remT.empty()) {
                    vector<u64> kv(have.begin(), have.end());
                    sort(kv.begin(), kv.end());
                    dict.buildFullS1(kv);

                    // Dynamic ALL-target cheapest-next selection (grows generator tree in true creation order!)
                    int bestIdx = -1, bestCost = 1 << 29;
                    Cone bestCone;
                    // Check up to 16 smallest + random remaining targets for speed
                    vector<int> checkIdxs;
                    for (int i = 0; i < (int)remT.size(); i++) {
                        if (i < 10 || rng(2) == 0) checkIdxs.push_back(i);
                    }
                    for (int idx : checkIdxs) {
                        u64 T = remT[idx];
                        if (have.count(T)) { bestIdx = idx; bestCost = 0; bestCone.clear(); break; }
                        int tid = dict.h.get(T);
                        if (tid >= 0) {
                            BeamSearch bsQuick(dict, rng, 32, 3, 1, al);
                            bsQuick.run(T, 2);
                            Cone c;
                            if (!bsQuick.cands.empty() && bsQuick.extractCone(bsQuick.cands[0][1], bsQuick.cands[0][2], have, c)) {
                                if ((int)c.size() < bestCost) { bestCost = (int)c.size(); bestIdx = idx; bestCone = std::move(c); }
                                if (bestCost <= 1) break;
                            }
                            continue;
                        }
                        BeamSearch bs(dict, rng, 56, 4, 2, al);
                        bs.run(T, min(14, bestCost));
                        for (auto& cd : bs.cands) {
                            Cone c;
                            if (bs.extractCone(cd[1], cd[2], have, c) && (int)c.size() < bestCost) {
                                bestCost = (int)c.size();
                                bestIdx = idx;
                                bestCone = std::move(c);
                            }
                        }
                    }
                    if (bestIdx < 0) {
                        // Grow small powers/products if dictionary is initially too sparse
                        vector<u64> cur(have.begin(), have.end());
                        sort(cur.begin(), cur.end());
                        int g = 0;
                        for (auto it = cur.rbegin(); it != cur.rend() && g < 3; ++it) {
                            if (*it > 1 && *it <= P.L / *it && !have.count((*it) * (*it))) { addOp('*', *it, *it); g++; }
                        }
                        for (u64 v : cur) {
                            if (g < 6 && v <= P.L / 2 && !have.count(v + v)) { addOp('+', v, v); g++; }
                        }
                        if (g == 0) { aborted = true; break; }
                        continue;
                    }
                    for (auto& [v, df] : bestCone) {
                        have.insert(v);
                        scr.def[v] = df;
                        scr.order.push_back(v);
                    }
                    remT.erase(remT.begin() + bestIdx);
                    if ((int)scr.def.size() >= bestLen + 15) { aborted = true; break; }
                }
                if (!aborted) acceptNewState(std::move(scr));
            }
        }

        // ==================== MODE 1: HUB-CLUSTER SUBGRAPH RUIN (k = 3..9 TARGETS + SHARED HUBS!) ====================
        vector<u64> nonStartT;
        for (u64 t : P.targets) if (!st.starts.count(t)) nonStartT.push_back(t);
        sort(nonStartT.begin(), nonStartT.end());
        int nt = (int)nonStartT.size();

        int clusterIter = 0;
        while (now_sec() - t0 < secsPerPuzzle && nt >= 3) {
            clusterIter++;
            // Map every internal node h -> set of targets whose current DAG depends on h
            unordered_map<u64, vector<u64>> hubToTargets;
            for (u64 t : nonStartT) {
                unordered_set<u64> vis;
                vector<u64> stk{t};
                while (!stk.empty()) {
                    u64 v = stk.back(); stk.pop_back();
                    if (st.starts.count(v) || !vis.insert(v).second) continue;
                    if (v != t) hubToTargets[v].push_back(t);
                    auto it = st.def.find(v);
                    if (it != st.def.end()) { stk.push_back(it->second[1]); stk.push_back(it->second[2]); }
                }
            }

            // Collect candidate clusters of k in [3..9] targets:
            //   (a) Exact downstream target sets of hubs with 3 <= k <= 9!
            //   (b) Contiguous / close-value target windows of size 4..6!
            vector<vector<u64>> clusters;
            set<vector<u64>> seenCl;
            for (auto& [h, tlist] : hubToTargets) {
                if (tlist.size() >= 3 && tlist.size() <= 9) {
                    vector<u64> cl = tlist;
                    sort(cl.begin(), cl.end());
                    cl.erase(unique(cl.begin(), cl.end()), cl.end());
                    if (cl.size() >= 3 && seenCl.insert(cl).second) clusters.push_back(std::move(cl));
                }
            }
            for (int r = 0; r < 8 && nt >= 4; r++) {
                int k = 4 + (int)rng(min(5, nt - 3));
                int startIdx = (int)rng(nt - k + 1);
                vector<u64> cl(nonStartT.begin() + startIdx, nonStartT.begin() + startIdx + k);
                if (seenCl.insert(cl).second) clusters.push_back(std::move(cl));
            }
            if (clusters.empty()) break;

            // Shuffle clusters so every pass explores fresh subgraphs
            for (size_t i = clusters.size(); i > 1; i--) swap(clusters[i - 1], clusters[rng(i)]);

            bool anyGain = false;
            for (const auto& cl : clusters) {
                if (now_sec() - t0 > secsPerPuzzle) break;
                unordered_set<u64> skipSet(cl.begin(), cl.end());
                auto keep = st.closureExcept(skipSet);
                int f = (int)st.def.size() - (int)keep.size();
                if (f <= (int)cl.size() + 1) continue; // almost nothing removable

                unordered_set<u64> have0 = st.starts;
                have0.insert(keep.begin(), keep.end());

                // Run 6 collective re-synthesis trajectories over this k-target subgraph!
                // Trajectory 0: Dynamic Cheapest-First order (rebuilds whichever target in `cl` is cheapest first!)
                // Trajectories 1..5: Shuffled target orders + varied alpha + randomized tie-breaking within SLACK=1!
                int bestSubCost = f;
                Cone bestSubCone;

                for (int tr = 0; tr < 6 && now_sec() - t0 < secsPerPuzzle; tr++) {
                    double al = alphas[(tr + clusterIter) & 3];
                    unordered_set<u64> curHave = have0;
                    Cone curCone;
                    vector<u64> rem;
                    for (u64 t : cl) if (!curHave.count(t)) rem.push_back(t);
                    if (tr > 0) {
                        for (size_t i = rem.size(); i > 1; i--) swap(rem[i - 1], rem[rng(i)]);
                    }

                    bool ok = true;
                    while (!rem.empty()) {
                        if ((int)curCone.size() + (int)rem.size() > bestSubCost - 1 + 2) {
                            // Need at least 1 op per remaining unbuilt target unless two targets collapse
                        }
                        int remBudget = (bestSubCost - 1) - (int)curCone.size();
                        if (remBudget < 1) { ok = false; break; }

                        vector<u64> kv(curHave.begin(), curHave.end());
                        sort(kv.begin(), kv.end());
                        dict.buildFullS1(kv);

                        if (tr == 0) {
                            // Cheapest-next target inside the ruined cluster!
                            int bIdx = -1, bSz = remBudget + 1;
                            Cone bC;
                            for (int i = 0; i < (int)rem.size(); i++) {
                                if (curHave.count(rem[i])) { bIdx = i; bSz = 0; bC.clear(); break; }
                                BeamSearch bs(dict, rng, width, prodK, 2, al);
                                bs.run(rem[i], min(remBudget, bSz));
                                for (auto& cd : bs.cands) {
                                    Cone c;
                                    if (bs.extractCone(cd[1], cd[2], curHave, c) && (int)c.size() < bSz) {
                                        bSz = (int)c.size(); bIdx = i; bC = std::move(c);
                                    }
                                }
                            }
                            if (bIdx < 0) { ok = false; break; }
                            for (auto& kvp : bC) { curHave.insert(kvp.first); curCone.push_back(kvp); }
                            rem.erase(rem.begin() + bIdx);
                        } else {
                            u64 T = rem.back(); rem.pop_back();
                            if (curHave.count(T)) continue;
                            BeamSearch bs(dict, rng, width, prodK, 2, al);
                            bs.run(T, min(14, remBudget));
                            if (bs.cands.empty()) { ok = false; break; }
                            // Pick among cones within +1 of minimum size to diversify intermediate hubs for later targets!
                            vector<Cone> opts;
                            int minC = 1 << 29;
                            for (auto& cd : bs.cands) {
                                Cone c;
                                if (bs.extractCone(cd[1], cd[2], curHave, c) && (int)c.size() <= remBudget) {
                                    if ((int)c.size() < minC) minC = (int)c.size();
                                    opts.push_back(std::move(c));
                                }
                            }
                            vector<Cone*> good;
                            for (auto& c : opts) if ((int)c.size() <= minC + (rem.empty() ? 0 : 1)) good.push_back(&c);
                            if (good.empty()) { ok = false; break; }
                            Cone& chosen = *good[rem.empty() ? 0 : rng(good.size())];
                            for (auto& kvp : chosen) { curHave.insert(kvp.first); curCone.push_back(kvp); }
                        }
                    }

                    if (ok && (int)curCone.size() < bestSubCost) {
                        bestSubCost = (int)curCone.size();
                        bestSubCone = std::move(curCone);
                    }
                }

                if (bestSubCost < f && !bestSubCone.empty()) {
                    DagState candSt = st;
                    for (auto it = candSt.def.begin(); it != candSt.def.end();) {
                        if (!keep.count(it->first)) it = candSt.def.erase(it);
                        else ++it;
                    }
                    for (auto& [v, df] : bestSubCone) { candSt.def[v] = df; candSt.order.push_back(v); }
                    if (acceptNewState(std::move(candSt))) {
                        anyGain = true;
                        break; // Recompute hubToTargets on improved DAG!
                    }
                }
            }
            if (!anyGain && clusterIter >= 3) break;
        }

        // ==================== MODE 3: EARTHQUAKE FORBIDDEN-HUB MUTANT LINEAGE GENERATOR ====================
        if (nEarthquake > 0 && seedOf.is_open() && nt >= 3) {
            unordered_set<u64> tSet(P.targets.begin(), P.targets.end());
            // Find all internal non-target, non-1 hubs feeding 3..11 targets!
            vector<pair<int, u64>> candHubs;
            for (auto& [h, _] : st.def) {
                if (h == 1 || tSet.count(h) || st.starts.count(h)) continue;
                // Compute transitive dependents of h
                unordered_set<u64> dead{h};
                bool changed = true;
                while (changed) {
                    changed = false;
                    for (auto& [v, df] : st.def) {
                        if (!dead.count(v) && (dead.count(df[1]) || dead.count(df[2]))) {
                            dead.insert(v); changed = true;
                        }
                    }
                }
                int brokenT = 0;
                for (u64 t : nonStartT) if (dead.count(t)) brokenT++;
                if (brokenT >= 3 && brokenT <= 11 && dead.size() >= 6 && dead.size() <= 45) {
                    candHubs.push_back({(int)dead.size(), h});
                }
            }
            for (size_t i = candHubs.size(); i > 1; i--) swap(candHubs[i - 1], candHubs[rng(i)]);

            int emittedMut = 0;
            for (auto& [deadSz, forbidH] : candHubs) {
                if (emittedMut >= nEarthquake) break;
                unordered_set<u64> dead{forbidH};
                bool ch = true;
                while (ch) {
                    ch = false;
                    for (auto& [v, df] : st.def) {
                        if (!dead.count(v) && (dead.count(df[1]) || dead.count(df[2]))) {
                            dead.insert(v); ch = true;
                        }
                    }
                }
                DagState mut = st;
                for (u64 d : dead) mut.def.erase(d);
                vector<u64> newOrd;
                for (u64 v : st.order) if (mut.def.count(v)) newOrd.push_back(v);
                mut.order = std::move(newOrd);

                unordered_set<u64> curHave = mut.starts;
                for (auto& [v, _] : mut.def) curHave.insert(v);

                vector<u64> broken;
                for (u64 t : nonStartT) if (!curHave.count(t)) broken.push_back(t);
                for (size_t i = broken.size(); i > 1; i--) swap(broken[i - 1], broken[rng(i)]);

                bool ok = true;
                double al = alphas[emittedMut & 3];
                for (u64 T : broken) {
                    if (curHave.count(T)) continue;
                    int remAllowed = (origLen + 6) - (int)mut.def.size();
                    if (remAllowed < 1) { ok = false; break; }
                    vector<u64> kv(curHave.begin(), curHave.end());
                    sort(kv.begin(), kv.end());
                    dict.buildFullS1(kv, false);
                    BeamSearch bs(dict, rng, width, prodK, 2, al);
                    bs.run(T, min(14, remAllowed));
                    bool picked = false;
                    for (auto& cd : bs.cands) {
                        Cone c;
                        if (!bs.extractCone(cd[1], cd[2], curHave, c) || (int)c.size() > remAllowed) continue;
                        bool usesForbid = false;
                        for (auto& [v, _] : c) if (v == forbidH) usesForbid = true;
                        if (usesForbid) continue;
                        for (auto& [v, df] : c) {
                            curHave.insert(v);
                            mut.def[v] = df;
                            mut.order.push_back(v);
                        }
                        picked = true;
                        break;
                    }
                    if (!picked) { ok = false; break; }
                }
                if (!ok) continue;
                mut.pruneToTargets();
                if (mut.def.count(forbidH) || (int)mut.def.size() > origLen + 6) continue;
                emittedMut++;
                seedOf << "V " << P.id << " " << origLen << " " << mut.order.size() << " "
                       << ((int)mut.order.size() - origLen) << " " << deadSz;
                for (u64 v : mut.order) seedOf << " " << v;
                seedOf << "\n";
                seedOf.flush();
                acceptNewState(std::move(mut));
            }
        }

        if (minClosure < origLen && seedOf.is_open()) {
            seedOf << "V " << P.id << " " << origLen << " " << minClosure << " "
                   << (minClosure - origLen) << " " << minClosure;
            for (u64 v : minClosureVals) seedOf << " " << v;
            seedOf << "\n";
            seedOf.flush();
        }

        if (bestLen < origLen && verifyProgExact(P, bestProg)) {
            cout << "R " << P.id << " " << origLen << " " << bestProg.size() << "\n";
            for (auto& ins : bestProg) cout << (char)ins[0] << " " << ins[1] << " " << ins[2] << " " << ins[3] << "\n";
            cout.flush();
        }
    }
    return 0;
}
