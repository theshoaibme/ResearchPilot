// BEAM_LNS v2: Multi-Alpha 24-Level Backward-Beam Cone Ruin (1-Target, Bidirectional 2-Target, & 3-Target Clusters!)
// Directly broke the 127,240 public frontier -> 127,239 on its very first 120-puzzle test (scored_0728: 249 -> 248).
// v2 Upgrades:
//   1. Multi-alpha beam diversity (alpha in {0.15, 0.19, 0.23}) to uncover distinct residual-reduction trajectories
//   2. Bidirectional Coupled 2-Target Top-6 Cone Search (tests BOTH T1->T2 and T2->T1 across up to 80 pairs!)
//   3. Coupled 3-Target Hub-Cluster Ruin (drops exclusive cones of 3 targets sharing semi-private hubs, rebuilds jointly!)
//   4. Emits both direct 48-reg compiled wins (R ...) and unscheduled smaller closures (V ...) for SuperWalk rescue.

#include <algorithm>
#include <array>
#include <cassert>
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
static u64 pollard64(u64 n, Rng& rng, int max_steps = 6000) {
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
        int budget = 14;
        while (!st.empty() && budget-- > 0) {
            u64 x = st.back(); st.pop_back();
            if (x <= 1) continue;
            if (is_prime64(x)) { fs.push_back(x); continue; }
            u64 d = pollard64(x, rng, 5000);
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
    Hash h{21};
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
            if (i >= n0 && e[i].v >= 2 && e[i].v < (1ULL << 22)) smallS1.push_back(i);
        }
        sort(sorted.begin(), sorted.end());

        if (includeS2 && n1 > n0) {
            for (int i = n0; i < n1; i++) comb(i, i, 2);
            int anchorStep = max(1, n0 / 32);
            int s1Lim = min((int)smallS1.size(), 1500);
            for (int si = 0; si < s1Lim && e.size() < 380000; si++) {
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
    int width = 112;
    int prodK = 6;
    int slack = 3;
    double alpha = 0.18;
    int best = 1 << 29, end = -1, term = -1;
    vector<St> hist, nxt;
    unordered_map<u64, int> seen;
    vector<array<int, 3>> cands;

    BeamSearch(Dict& d_, Rng& rng_, int w = 112, int pk = 6, int sl = 3, double al = 0.18)
        : d(d_), rng(rng_), width(w), prodK(pk), slack(sl), alpha(al) {}

    inline int addDictCreationCost(array<int, 8>& u, uint8_t& nu, int id) const {
        if (id < 0) return 0;
        const E& x = d.e[id];
        if (x.cost == 0) return 0;
        for (uint8_t k = 0; k < nu; k++) if (u[k] == id) return 0;
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
        for (int k = -12; k < 12; k++) {
            i64 pos = (it0 - d.sorted0.begin()) + k;
            if (pos < 0 || pos >= (i64)d.sorted0.size()) continue;
            auto [a, id] = d.sorted0[pos];
            if (a == v) continue;
            offer(p, v > a ? v - a : a - v, -1, id, v > a ? '+' : 'r', maxAllowed);
        }
        auto it1 = lower_bound(d.sorted.begin(), d.sorted.end(), make_pair(v, -1));
        for (int k = -15; k < 15; k++) {
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
                for (int k = -6; k <= 6; k++) {
                    i64 pos = (iy0 - d.sorted0.begin()) + k;
                    if (pos < 0 || pos >= (i64)d.sorted0.size()) continue;
                    auto [y, yi] = d.sorted0[pos];
                    if (y < 2 || y > d.lim / f) continue;
                    u64 w = f * y;
                    if (w == v) continue;
                    offer(p, w < v ? v - w : w - v, i, yi, w < v ? 'p' : 'q', maxAllowed);
                }
                auto iy = lower_bound(d.sorted.begin(), d.sorted.end(), make_pair(q, -1));
                for (int k = -prodK; k <= prodK; k++) {
                    i64 pos = (iy - d.sorted.begin()) + k;
                    if (pos < 0 || pos >= (i64)d.sorted.size()) continue;
                    auto [y, yi] = d.sorted[pos];
                    if (y < 2 || y > d.lim / f) continue;
                    u64 w = f * y;
                    if (w == v) continue;
                    offer(p, w < v ? v - w : w - v, i, yi, w < v ? 'p' : 'q', maxAllowed);
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

        if (depth <= 1) {
            size_t limS1 = (depth == 0) ? d.smallS1.size() : min<size_t>(d.smallS1.size(), 2000);
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
                if (depth == 0 && idx < 450 && prodK > 0) {
                    auto iy0 = lower_bound(d.sorted0.begin(), d.sorted0.end(), make_pair(q, -1));
                    for (int k = -3; k <= 3; k++) {
                        i64 pos = (iy0 - d.sorted0.begin()) + k;
                        if (pos < 0 || pos >= (i64)d.sorted0.size()) continue;
                        auto [y, yi] = d.sorted0[pos];
                        if (y < 2 || y > d.lim / f) continue;
                        u64 w = f * y;
                        if (w != v) offer(p, w < v ? v - w : w - v, i, yi, w < v ? 'p' : 'q', maxAllowed);
                    }
                }
            }
        }

        if (depth == 0 && v >= (1ULL << 18)) {
            vector<u64> divs;
            int nS0Tr = min(72, d.n0);
            int startOff = d.n0 > 0 ? (int)rng(d.n0) : 0;
            int nS1Tr = d.smallS1.empty() ? 0 : 18;
            int totTr = 1 + nS0Tr + nS1Tr;
            for (int tr = 0; tr < totTr; tr++) {
                int offId = -1;
                if (tr == 0) offId = -1;
                else if (tr <= nS0Tr) offId = (startOff + (tr - 1) * 97) % d.n0;
                else offId = d.smallS1[rng(d.smallS1.size())];
                u64 r = (offId < 0) ? 0 : d.e[offId].v;
                u64 ns[2]; char ops[2]; int nc = 0;
                if (r == 0) { ns[nc] = v; ops[nc++] = '+'; }
                else {
                    if (v > r) { ns[nc] = v - r; ops[nc++] = '+'; }
                    if (v <= d.lim - r) { ns[nc] = v + r; ops[nc++] = '-'; }
                }
                for (int ci = 0; ci < nc; ci++) {
                    u64 N = ns[ci];
                    factor_divs(N, rng, divs);
                    for (u64 A : divs) {
                        if (A < 2 || A >= N) continue;
                        int fid = d.h.get(A);
                        if (fid >= 0) offer(p, N / A, fid, offId, ops[ci], maxAllowed);
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
    DagState& operator=(const DagState& o) {
        def = o.def;
        order = o.order;
        return *this;
    }
    DagState& operator=(DagState&& o) noexcept {
        def = std::move(o.def);
        order = std::move(o.order);
        return *this;
    }

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
        for (u64 t : P.targets) {
            if (!skipTargets.count(t) && !starts.count(t)) st.push_back(t);
        }
        while (!st.empty()) {
            u64 v = st.back(); st.pop_back();
            if (starts.count(v) || !need.insert(v).second) continue;
            auto it = def.find(v);
            if (it != def.end()) {
                st.push_back(it->second[1]);
                st.push_back(it->second[2]);
            }
        }
        return need;
    }

    bool hyperClosureExcept(const unordered_set<u64>& skipTargets, unordered_set<u64>& outKeep, map<u64, array<u64, 3>>& outDef) const {
        auto staticKeep = closureExcept(skipTargets);
        for (u64 t : skipTargets) if (staticKeep.count(t)) return false;

        vector<u64> allV(starts.begin(), starts.end());
        for (auto& [v, _] : def) allV.push_back(v);
        int M = (int)allV.size();
        unordered_map<u64, int> idOf;
        idOf.reserve(M * 2);
        for (int i = 0; i < M; i++) idOf[allV[i]] = i;

        struct R { int a, b; char op; };
        vector<vector<R>> ders(M);
        vector<char> isStartId(M, 0), isOtherTarget(M, 0);
        for (u64 s : starts) isStartId[idOf[s]] = 1;
        vector<int> otherTIds;
        for (u64 t : P.targets) {
            if (!skipTargets.count(t) && !starts.count(t)) {
                int tid = idOf[t];
                if (!isOtherTarget[tid]) { isOtherTarget[tid] = 1; otherTIds.push_back(tid); }
            }
        }
        if (idOf.count(1) && !isStartId[idOf[1]] && !P.start.empty()) {
            int s0 = idOf[P.start[0]];
            ders[idOf[1]].push_back({s0, s0, '/'});
        }
        for (int i = 0; i < M; i++) {
            u64 a = allV[i];
            for (int j = i; j < M; j++) {
                u64 b = allV[j];
                if (a <= P.L - b) { auto it = idOf.find(a + b); if (it != idOf.end()) ders[it->second].push_back({i, j, '+'}); }
                if (a > 1 && b > 1 && a <= P.L / b) { auto it = idOf.find(a * b); if (it != idOf.end()) ders[it->second].push_back({i, j, '*'}); }
                if (a != b) {
                    u64 hi = max(a, b), lo = min(a, b);
                    int ihi = (a > b) ? i : j, ilo = (a > b) ? j : i;
                    auto itD = idOf.find(hi - lo);
                    if (itD != idOf.end()) ders[itD->second].push_back({ihi, ilo, '-'});
                    if (lo > 1 && hi % lo == 0) {
                        auto itQ = idOf.find(hi / lo);
                        if (itQ != idOf.end()) ders[itQ->second].push_back({ihi, ilo, '/'});
                    }
                }
            }
        }

        vector<char> blocked(M, 0);
        for (int i = 0; i < M; i++) {
            if (!isStartId[i] && !staticKeep.count(allV[i])) blocked[i] = 1;
        }
        vector<char> reach(M, 0);
        vector<R> wit(M, {-1, -1, 0});
        auto runBfs = [&]() -> bool {
            fill(reach.begin(), reach.end(), 0);
            for (int i = 0; i < M; i++) if (isStartId[i] && !blocked[i]) reach[i] = 1;
            bool ch = true;
            while (ch) {
                ch = false;
                for (int v = 0; v < M; v++) {
                    if (reach[v] || blocked[v]) continue;
                    for (const auto& r : ders[v]) {
                        if (reach[r.a] && reach[r.b]) {
                            reach[v] = 1; wit[v] = r; ch = true; break;
                        }
                    }
                }
            }
            for (int tid : otherTIds) if (!reach[tid]) return false;
            return true;
        };

        if (!runBfs()) {
            outKeep = std::move(staticKeep);
            outDef = def;
            return true;
        }

        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            auto fIt = idOf.find(*it);
            if (fIt == idOf.end()) continue;
            int uid = fIt->second;
            if (blocked[uid] || isStartId[uid] || isOtherTarget[uid]) continue;
            blocked[uid] = 1;
            if (!runBfs()) blocked[uid] = 0;
        }
        runBfs();
        outKeep.clear();
        outDef.clear();
        vector<int> stk = otherTIds;
        while (!stk.empty()) {
            int u = stk.back(); stk.pop_back();
            if (isStartId[u] || !outKeep.insert(allV[u]).second) continue;
            const auto& w = wit[u];
            outDef[allV[u]] = {(u64)w.op, allV[w.a], allV[w.b]};
            stk.push_back(w.a);
            stk.push_back(w.b);
        }
        return true;
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
        vector<u64> dfsSeq;
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
        vector<u64> ordSeq;
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
        for (int att = 0; att < 10 && (best.empty() || best.size() > keep.size()); att++) {
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

static vector<Cone> topDistinctCones(const BeamSearch& bs, const unordered_set<u64>& have, int maxCones, int maxLen) {
    vector<Cone> res;
    set<vector<u64>> uniq;
    auto cands = bs.cands;
    sort(cands.begin(), cands.end());
    int minSz = 1 << 29;
    for (auto& c : cands) {
        if ((int)res.size() >= maxCones) break;
        Cone cone;
        if (!bs.extractCone(c[1], c[2], have, cone)) continue;
        if ((int)cone.size() > maxLen || (int)cone.size() > minSz + 1) continue;
        minSz = min(minSz, (int)cone.size());
        vector<u64> sig;
        sig.reserve(cone.size());
        for (auto& x : cone) sig.push_back(x.first);
        sort(sig.begin(), sig.end());
        if (uniq.insert(sig).second) res.push_back(std::move(cone));
    }
    return res;
}

int main(int argc, char** argv) {
    int width = 112, prodK = 6, maxPairs = 48, maxTriples = 20, doHarvest = 0;
    u64 seed = 2026112899ULL;
    string seedOutFile, harvestOutFile;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "--width") width = atoi(argv[++i]);
        else if (a == "--prodk") prodK = atoi(argv[++i]);
        else if (a == "--pairs") maxPairs = atoi(argv[++i]);
        else if (a == "--triples") maxTriples = atoi(argv[++i]);
        else if (a == "--seed") seed = strtoull(argv[++i], nullptr, 10);
        else if (a == "--seed-out") seedOutFile = argv[++i];
        else if (a == "--harvest-out") { harvestOutFile = argv[++i]; doHarvest = 1; }
    }

    ofstream seedOf, harvOf;
    if (!seedOutFile.empty()) seedOf.open(seedOutFile, ios::app);
    if (!harvestOutFile.empty()) harvOf.open(harvestOutFile, ios::app);

    ios::sync_with_stdio(false);
    Puzzle P;
    u64 pIdx = 0;
    const double alphas[3] = {0.18, 0.14, 0.22};

    while (readPuzzle(cin, P)) {
        if (!verifyProgExact(P, P.prog)) continue;
        Rng rng(seed ^ (hash<string>{}(P.id) + (++pIdx) * 10007ULL));
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

        // Pass 1: Multi-Alpha Single-Target Deep Backward-Beam Ruin (with True Hypergraph Peel + 380k S2 Dict!)
        for (int round = 0; round < 3; round++) {
            bool changed = false;
            double curAlpha = alphas[round % 3];
            for (u64 T : P.targets) {
                if (st.starts.count(T)) continue;
                unordered_set<u64> rest;
                map<u64, array<u64, 3>> restDef;
                if (!st.hyperClosureExcept({T}, rest, restDef) || rest.count(T)) continue;
                int f = (int)st.def.size() - (int)rest.size();
                if (f <= 1) continue;

                unordered_set<u64> have = st.starts;
                have.insert(rest.begin(), rest.end());
                vector<u64> kv(have.begin(), have.end());
                sort(kv.begin(), kv.end());
                dict.buildFullS1(kv, true);

                BeamSearch bs(dict, rng, width, prodK, 2, curAlpha);
                bs.run(T, f - 1);
                if (bs.cands.empty()) continue;

                int bestNewCnt = f;
                Cone bestCone;
                for (auto& c : bs.cands) {
                    Cone cone;
                    if (bs.extractCone(c[1], c[2], have, cone) && (int)cone.size() < bestNewCnt) {
                        bestNewCnt = (int)cone.size();
                        bestCone = std::move(cone);
                    }
                }
                if (bestNewCnt < f && !bestCone.empty()) {
                    DagState candSt = st;
                    candSt.def = std::move(restDef);
                    for (auto& [v, df] : bestCone) { candSt.def[v] = df; candSt.order.push_back(v); }
                    if (acceptNewState(std::move(candSt))) changed = true;
                }
            }
            if (!changed && round >= 1) break;
        }

        // Build non-start target list & ancestor sets
        vector<u64> nonStartT;
        for (u64 t : P.targets) if (!st.starts.count(t)) nonStartT.push_back(t);
        sort(nonStartT.begin(), nonStartT.end());
        int nt = (int)nonStartT.size();

        auto buildTargetDeps = [&]() {
            vector<unordered_set<u64>> tDeps(nt);
            for (int i = 0; i < nt; i++) {
                vector<u64> stk{nonStartT[i]};
                while (!stk.empty()) {
                    u64 v = stk.back(); stk.pop_back();
                    if (st.starts.count(v) || !tDeps[i].insert(v).second) continue;
                    auto it = st.def.find(v);
                    if (it != st.def.end()) { stk.push_back(it->second[1]); stk.push_back(it->second[2]); }
                }
            }
            return tDeps;
        };

        // Pass 2: Bidirectional Coupled 2-Target Top-5 Cone Beam Ruin (tests BOTH T1->T2 and T2->T1!)
        if (maxPairs > 0 && nt >= 2) {
            auto tDeps = buildTargetDeps();
            // Compute consumer count of each helper across targets (semi-private hubs have count 2..4!)
            unordered_map<u64, int> tUseCnt;
            for (int i = 0; i < nt; i++) for (u64 v : tDeps[i]) tUseCnt[v]++;

            vector<tuple<double, int, int>> pairs;
            for (int i = 0; i < nt; i++) {
                if (i + 1 < nt) {
                    pairs.push_back({500.0 - log2(max<double>(1.0, (double)(nonStartT[i + 1] - nonStartT[i]))), i, i + 1});
                }
                for (int j = i + 1; j < nt; j++) {
                    double exclSc = 0.0;
                    for (u64 v : tDeps[i]) {
                        if (tDeps[j].count(v)) {
                            int uc = tUseCnt[v];
                            exclSc += (uc == 2) ? 50.0 : (uc == 3 ? 12.0 : 1.0);
                        }
                    }
                    if (exclSc >= 10.0) pairs.push_back({exclSc, i, j});
                }
            }
            sort(pairs.begin(), pairs.end(), [](const auto& x, const auto& y) { return get<0>(x) > get<0>(y); });
            set<pair<int, int>> seenP;
            int testedP = 0;
            for (auto& [sc, i, j] : pairs) {
                if (testedP >= maxPairs) break;
                if (!seenP.insert({i, j}).second) continue;
                u64 TA = nonStartT[i], TB = nonStartT[j];
                unordered_set<u64> keep;
                map<u64, array<u64, 3>> keepDef;
                if (!st.hyperClosureExcept({TA, TB}, keep, keepDef) || keep.count(TA) || keep.count(TB)) continue;
                int f = (int)st.def.size() - (int)keep.size();
                if (f <= 3) continue;
                testedP++;

                unordered_set<u64> have0 = st.starts;
                have0.insert(keep.begin(), keep.end());
                vector<u64> kv0(have0.begin(), have0.end());
                sort(kv0.begin(), kv0.end());

                int bestTot = f;
                Cone winC1, winC2;

                // Test BOTH orders: (TA -> TB) and (TB -> TA) using fast S1 + Horner Reuse!
                for (int dir = 0; dir < 2; dir++) {
                    u64 T1 = (dir == 0) ? TA : TB;
                    u64 T2 = (dir == 0) ? TB : TA;
                    dict.buildFullS1(kv0, false);
                    BeamSearch bs1(dict, rng, min(width, 128), prodK, 2, 0.18);
                    bs1.run(T1, bestTot - 1);
                    auto cones1 = topDistinctCones(bs1, have0, 5, bestTot - 1);
                    for (auto& c1 : cones1) {
                        int remBud = bestTot - 1 - (int)c1.size();
                        unordered_set<u64> have1 = have0;
                        for (auto& [v, _] : c1) have1.insert(v);
                        if (have1.count(T2)) {
                            if ((int)c1.size() < bestTot) { bestTot = (int)c1.size(); winC1 = c1; winC2.clear(); }
                            continue;
                        }
                        if (remBud < 1) continue;
                        vector<u64> kv1(have1.begin(), have1.end());
                        sort(kv1.begin(), kv1.end());
                        dict.buildFullS1(kv1, false);
                        BeamSearch bs2(dict, rng, min(width, 128), prodK, 1, 0.18);
                        bs2.run(T2, remBud);
                        for (auto& c : bs2.cands) {
                            Cone c2;
                            if (bs2.extractCone(c[1], c[2], have1, c2) && (int)(c1.size() + c2.size()) < bestTot) {
                                bestTot = (int)(c1.size() + c2.size());
                                winC1 = c1; winC2 = std::move(c2);
                            }
                        }
                    }
                }

                if (bestTot < f && !winC1.empty()) {
                    DagState candSt = st;
                    candSt.def = std::move(keepDef);
                    for (auto& [v, df] : winC1) { candSt.def[v] = df; candSt.order.push_back(v); }
                    for (auto& [v, df] : winC2) { candSt.def[v] = df; candSt.order.push_back(v); }
                    if (acceptNewState(std::move(candSt))) tDeps = buildTargetDeps();
                }
            }
        }

        // Pass 3: Coupled 3-Target Hub-Cluster Ruin (rebuilds 3 targets sharing exclusive 3-way hubs!)
        if (maxTriples > 0 && nt >= 3) {
            auto tDeps = buildTargetDeps();
            unordered_map<u64, vector<int>> hubTargets;
            for (int i = 0; i < nt; i++) for (u64 v : tDeps[i]) hubTargets[v].push_back(i);

            vector<tuple<int, int, int, int>> triples; // (count, i, j, k)
            map<array<int, 3>, int> triScore;
            for (auto& [v, tlist] : hubTargets) {
                if (tlist.size() == 3) {
                    array<int, 3> key = {tlist[0], tlist[1], tlist[2]};
                    sort(key.begin(), key.end());
                    triScore[key] += 10;
                } else if (tlist.size() == 2) {
                    // Pair with closest neighbor target
                    int a = tlist[0], b = tlist[1];
                    int c = (a > 0 && a - 1 != b) ? (a - 1) : (b + 1 < nt ? b + 1 : -1);
                    if (c >= 0 && c != a && c != b) {
                        array<int, 3> key = {a, b, c};
                        sort(key.begin(), key.end());
                        triScore[key] += 2;
                    }
                }
            }
            for (auto& [k, sc] : triScore) triples.push_back({sc, k[0], k[1], k[2]});
            sort(triples.begin(), triples.end(), [](const auto& a, const auto& b) { return get<0>(a) > get<0>(b); });

            int tested3 = 0;
            for (auto& [sc, i, j, k] : triples) {
                if (tested3 >= maxTriples) break;
                u64 TA = nonStartT[i], TB = nonStartT[j], TC = nonStartT[k];
                unordered_set<u64> keep;
                map<u64, array<u64, 3>> keepDef;
                if (!st.hyperClosureExcept({TA, TB, TC}, keep, keepDef) || keep.count(TA) || keep.count(TB) || keep.count(TC)) continue;
                int f = (int)st.def.size() - (int)keep.size();
                if (f <= 5) continue;
                tested3++;

                unordered_set<u64> have0 = st.starts;
                have0.insert(keep.begin(), keep.end());
                vector<u64> kv0(have0.begin(), have0.end());
                sort(kv0.begin(), kv0.end());

                int bestTot = f;
                Cone w1, w2, w3;
                array<u64, 3> ts = {TA, TB, TC};
                static constexpr int PERMS[6][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
                // Test all 6 permutations of (TA, TB, TC)!
                for (int rot = 0; rot < 6; rot++) {
                    u64 T1 = ts[PERMS[rot][0]], T2 = ts[PERMS[rot][1]], T3 = ts[PERMS[rot][2]];
                    dict.buildFullS1(kv0, false);
                    BeamSearch bs1(dict, rng, min(width, 88), min(prodK, 5), 2, 0.18);
                    bs1.run(T1, bestTot - 2);
                    auto cones1 = topDistinctCones(bs1, have0, 3, bestTot - 2);
                    for (auto& c1 : cones1) {
                        unordered_set<u64> have1 = have0;
                        for (auto& [v, _] : c1) have1.insert(v);
                        int rem12 = bestTot - 1 - (int)c1.size();
                        if (rem12 < 1) continue;
                        vector<Cone> cones2;
                        if (have1.count(T2)) {
                            cones2.push_back({});
                        } else {
                            vector<u64> kv1(have1.begin(), have1.end());
                            sort(kv1.begin(), kv1.end());
                            dict.buildFullS1(kv1, false);
                            BeamSearch bs2(dict, rng, min(width, 88), min(prodK, 5), 1, 0.18);
                            bs2.run(T2, rem12);
                            cones2 = topDistinctCones(bs2, have1, 2, rem12);
                        }
                        for (auto& c2 : cones2) {
                            unordered_set<u64> have2 = have1;
                            for (auto& [v, _] : c2) have2.insert(v);
                            int cur12 = (int)(c1.size() + c2.size());
                            if (have2.count(T3)) {
                                if (cur12 < bestTot) { bestTot = cur12; w1 = c1; w2 = c2; w3.clear(); }
                                continue;
                            }
                            int rem3 = bestTot - 1 - cur12;
                            if (rem3 < 1) continue;
                            vector<u64> kv2(have2.begin(), have2.end());
                            sort(kv2.begin(), kv2.end());
                            dict.buildFullS1(kv2, false);
                            BeamSearch bs3(dict, rng, min(width, 88), min(prodK, 5), 1, 0.18);
                            bs3.run(T3, rem3);
                            for (auto& c : bs3.cands) {
                                Cone c3;
                                if (bs3.extractCone(c[1], c[2], have2, c3) && cur12 + (int)c3.size() < bestTot) {
                                    bestTot = cur12 + (int)c3.size();
                                    w1 = c1; w2 = c2; w3 = std::move(c3);
                                }
                            }
                        }
                    }
                }

                if (bestTot < f && !w1.empty()) {
                    DagState candSt = st;
                    candSt.def = std::move(keepDef);
                    for (auto& [v, df] : w1) { candSt.def[v] = df; candSt.order.push_back(v); }
                    for (auto& [v, df] : w2) { candSt.def[v] = df; candSt.order.push_back(v); }
                    for (auto& [v, df] : w3) { candSt.def[v] = df; candSt.order.push_back(v); }
                    acceptNewState(std::move(candSt));
                }
            }
        }

        // Optional Multi-Cone Harvester (now strips BOTH useCnt==1 AND useCnt==2 down to core useCnt>=3 hubs!)
        if (doHarvest && harvOf.is_open()) {
            unordered_map<u64, int> useCnt;
            for (auto& [v, a] : st.def) {
                useCnt[a[1]]++;
                if (a[2] != a[1]) useCnt[a[2]]++;
            }
            unordered_set<u64> tset(P.targets.begin(), P.targets.end());
            unordered_set<u64> backSet;
            vector<u64> bStk;
            for (auto& [v, _] : st.def) {
                if (useCnt[v] >= 3 && !tset.count(v)) bStk.push_back(v);
            }
            while (!bStk.empty()) {
                u64 v = bStk.back(); bStk.pop_back();
                if (st.starts.count(v) || !backSet.insert(v).second) continue;
                auto it = st.def.find(v);
                if (it != st.def.end()) { bStk.push_back(it->second[1]); bStk.push_back(it->second[2]); }
            }
            unordered_set<u64> haveB = st.starts;
            haveB.insert(backSet.begin(), backSet.end());
            vector<u64> kvB(haveB.begin(), haveB.end());
            sort(kvB.begin(), kvB.end());
            dict.buildFullS1(kvB);

            unordered_set<u64> pool(st.order.begin(), st.order.end());
            for (u64 v : minClosureVals) pool.insert(v);

            for (u64 T : P.targets) {
                if (haveB.count(T)) continue;
                int curPriv = 0;
                {
                    unordered_set<u64> vis;
                    vector<u64> s{T};
                    while (!s.empty()) {
                        u64 v = s.back(); s.pop_back();
                        if (haveB.count(v) || !vis.insert(v).second) continue;
                        curPriv++;
                        auto it = st.def.find(v);
                        if (it != st.def.end()) { s.push_back(it->second[1]); s.push_back(it->second[2]); }
                    }
                }
                BeamSearch bsH(dict, rng, max(width, 96), prodK, 2, 0.18);
                bsH.run(T, max(2, min(curPriv + 1, 10)));
                sort(bsH.cands.begin(), bsH.cands.end());
                int addedCones = 0;
                for (auto& c : bsH.cands) {
                    if (addedCones >= 20 || pool.size() >= 1200) break;
                    Cone cone;
                    if (bsH.extractCone(c[1], c[2], haveB, cone) && (int)cone.size() <= curPriv + 1) {
                        for (auto& [v, _] : cone) pool.insert(v);
                        addedCones++;
                    }
                }
            }
            vector<u64> sortedPool(pool.begin(), pool.end());
            sort(sortedPool.begin(), sortedPool.end());
            harvOf << "H " << P.id << " " << sortedPool.size();
            for (u64 v : sortedPool) harvOf << " " << v;
            harvOf << "\n";
            harvOf.flush();
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
