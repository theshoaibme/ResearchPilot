// SUPERWALK: Next-Horizon Algebraic Hypergraph & Register-Schedule Engine for Santa 2026
// Key Algorithmic Advances beyond HyperWalk:
//   1. Coupled Re-Derive + Topological Relocate + Peak-Buster 48-Register Scheduler
//      + Equal-Cost Swap Rescue whenever cost() < best (solves spill-locked states like scored_0774!)
//   2. Deep Cone Ruin (Rm <= 14) + Exact Sorted Two-Pointer S1xS1 Sum/Diff MITM
//      + S2-Remainder Division (X = A*B +- R where R is 2-step from K, up to 6 new nodes)
//   3. S1-Offset Brent Pollard-Rho Factorization (factors X +- r for BOTH r in K AND r in S1(K)!)
//   4. Arbitrary 2-Target / 2-Cone Joint Ruin & Target-Difference / Shared-Cone Synthesis
//   5. Exact Built-in Referee Verifier

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <set>
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
    inline double u() { return (next() >> 11) * (1.0 / (1ULL << 53)); }
};

static inline u64 hsh(u64 x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

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
static u64 pollard64(u64 n, Rng& rng, int max_steps = 8000) {
    if ((n & 1) == 0) return 2;
    for (int attempt = 0; attempt < 4; attempt++) {
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
static void factor_fast(u64 n, Rng& rng, vector<pair<u64,int>>& pf) {
    pf.clear();
    if (n <= 1) return;
    static const u64 SP[] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97};
    vector<u64> fs;
    for (u64 p : SP) {
        while (n % p == 0) { fs.push_back(p); n /= p; }
    }
    if (n > 1) {
        vector<u64> st = {n};
        int budget = 16;
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
    for (u64 p : fs) {
        if (pf.empty() || pf.back().first != p) pf.push_back({p, 1});
        else pf.back().second++;
    }
}

struct Puzzle {
    string id;
    vector<u64> start, targets;
    u64 L;
    int regs;
    vector<array<int, 4>> prog;
};

static bool readPuzzle(istream& in, Puzzle& p) {
    string tag;
    while (in >> tag) {
        if (tag == "P") break;
    }
    if (!in) return false;
    p = Puzzle();
    int ns, nt, np;
    in >> p.id >> ns;
    p.start.resize(ns);
    for (int i = 0; i < ns; i++) in >> p.start[i];
    in >> nt;
    p.targets.resize(nt);
    for (int i = 0; i < nt; i++) in >> p.targets[i];
    in >> p.L >> p.regs >> np;
    p.prog.resize(np);
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

struct U64Map {
    vector<u64> k;
    vector<int> v;
    size_t mask = 0;
    void init(int bits) {
        size_t n = 1ULL << bits;
        k.assign(n, 0); v.assign(n, -1); mask = n - 1;
    }
    void clear() { fill(k.begin(), k.end(), 0); }
    inline int get(u64 x) const {
        if (!x) return -1;
        size_t i = hsh(x) & mask;
        while (true) {
            u64 kk = k[i];
            if (kk == x) return v[i];
            if (!kk) return -1;
            i = (i + 1) & mask;
        }
    }
    inline void set(u64 x, int val) {
        size_t i = hsh(x) & mask;
        while (k[i] && k[i] != x) i = (i + 1) & mask;
        k[i] = x; v[i] = val;
    }
    inline void erase(u64 x) {
        size_t i = hsh(x) & mask;
        while (k[i] && k[i] != x) i = (i + 1) & mask;
        if (!k[i]) return;
        size_t hole = i;
        while (true) {
            i = (i + 1) & mask;
            if (!k[i]) { k[hole] = 0; return; }
            size_t want = hsh(k[i]) & mask;
            if ((i - want) & mask && ((hole - want) & mask) < ((i - want) & mask)) {
                k[hole] = k[i]; v[hole] = v[i]; hole = i;
            }
        }
    }
};

struct PairTable {
    vector<u64> k;
    vector<uint16_t> cnt;
    vector<int> cpos;
    vector<u64> cand;
    size_t mask = 0, used = 0;
    void init(int bits) {
        size_t n = 1ULL << bits;
        k.assign(n, 0); cnt.assign(n, 0); cpos.assign(n, -1); cand.clear();
        mask = n - 1; used = 0;
    }
    inline void inc(u64 x) {
        size_t i = hsh(x) & mask;
        while (k[i] && k[i] != x) i = (i + 1) & mask;
        if (!k[i]) { k[i] = x; cnt[i] = 1; cpos[i] = -1; used++; return; }
        if (++cnt[i] == 2 && cpos[i] < 0) { cpos[i] = (int)cand.size(); cand.push_back(x); }
    }
    inline void dec(u64 x) {
        size_t i = hsh(x) & mask;
        while (k[i] && k[i] != x) i = (i + 1) & mask;
        if (!k[i] || cnt[i] == 0) return;
        if (--cnt[i] == 1 && cpos[i] >= 0) {
            int p = cpos[i]; cpos[i] = -1;
            u64 back = cand.back(); cand.pop_back();
            if (p < (int)cand.size()) {
                cand[p] = back;
                size_t j = hsh(back) & mask;
                while (k[j] != back) j = (j + 1) & mask;
                cpos[j] = p;
            }
        }
    }
};

struct SuperWalk {
    static constexpr int MV = 512;
    static constexpr int MT = 16384;

    const Puzzle& P;
    u64 L;
    Rng rng;

    u64 val[MV];
    bool alive[MV], isS[MV], isT[MV];
    int al[MV], posAl[MV], nal = 0;
    u64 av[MV];
    int nStart = 0, one = -1;
    vector<int> freeS, startSlots, targetSlots;
    U64Map vmap;

    int16_t ta[MT], tb[MT], tc[MT], tx[MT];
    uint8_t tt[MT];
    int tHigh = 0, nTri = 0;
    vector<int> freeT;
    vector<int> inc[MV];
    int deg[MV], crit[MV], sup[MV];
    vector<int> zeroed;

    PairTable pt;

    i64 stProp = 0, stSwap = 0, stUp = 0, stRb = 0, stRbHit = 0, stRbGain = 0;
    i64 stFac = 0, stFacHit = 0, stPair = 0, stPairHit = 0, stPairGain = 0;
    i64 stTargPair = 0, stTargPairHit = 0, stTargPairGain = 0;
    i64 stCross = 0, stSched = 0, stSchedFail = 0, stRemat = 0, stDown = 0;

    int best = 0;
    vector<u64> bestSet, feasSet;
    vector<array<int, 4>> bestProg;
    vector<vector<u64>> variants;

    double posv[MV];

    SuperWalk(const Puzzle& p, u64 seed) : P(p), L(p.L) {
        rng.s = seed ? seed : 1;
        for (int i = 0; i < 8; i++) rng.next();
        vmap.init(11);
        memset(alive, 0, sizeof alive);
        memset(isS, 0, sizeof isS);
        memset(isT, 0, sizeof isT);
        memset(deg, 0, sizeof deg);
        memset(crit, 0, sizeof crit);
        memset(sup, 255, sizeof sup);
        for (int i = MV - 1; i >= 0; i--) freeS.push_back(i);
    }

    inline int look(u64 w) const { return vmap.get(w); }
    inline bool fre(int s) const { return isS[s] || s == one; }
    inline bool removable(int u) const { return !isS[u] && !isT[u]; }
    inline int cost() const { return nal - nStart; }
    inline bool derives(int t, int x) const {
        return !(ta[t] == tb[t] && tc[t] != x);
    }

    struct Rel { int y, z; uint8_t k; };
    Rel rel[1024];
    int nrel = 0;

    void enumerate(u64 w) {
        nrel = 0;
        if (w <= L / 2) { int s = look(w + w); if (s >= 0 && nrel < 1020) rel[nrel++] = {s, s, 4}; }
        if (w > 1 && w <= L / w) { int s = look(w * w); if (s >= 0 && nrel < 1020) rel[nrel++] = {s, s, 5}; }
        u64 Lw = L / w;
        for (int i = 0; i < nal && nrel < 1020; i++) {
            u64 y = av[i]; int sy = al[i];
            if (y < w) {
                u64 d = w - y;
                if (d >= y) { int sz = look(d); if (sz >= 0) rel[nrel++] = {sy, sz, 0}; }
                if (y > 1 && w % y == 0) {
                    u64 q = w / y;
                    if (q >= y) { int sz = look(q); if (sz >= 0) rel[nrel++] = {sy, sz, 1}; }
                }
            }
            if (y <= L - w) { int sz = look(y + w); if (sz >= 0) rel[nrel++] = {sy, sz, 2}; }
            if (w > 1 && y > 1 && y <= Lw) { int sz = look(y * w); if (sz >= 0) rel[nrel++] = {sy, sz, 3}; }
        }
    }

    inline void addInc(int x, int t) {
        inc[x].push_back(t);
        if (!fre(x) && derives(t, x)) {
            int d = ++deg[x];
            if (d == 1) {
                for (int m : {ta[t], tb[t], tc[t]}) if (m != x) crit[m]++;
            } else if (d == 2) {
                for (int ot : inc[x]) if (ot != t && derives(ot, x)) {
                    for (int m : {ta[ot], tb[ot], tc[ot]}) if (m != x) {
                        if (--crit[m] == 0) zeroed.push_back(m);
                    }
                    break;
                }
            }
        }
    }
    inline void delInc(int x, int t) {
        auto& v = inc[x];
        for (size_t i = 0; i < v.size(); i++) {
            if (v[i] == t) { v[i] = v.back(); v.pop_back(); break; }
        }
        if (!fre(x) && derives(t, x)) {
            int d = --deg[x];
            if (d == 1) {
                for (int ot : inc[x]) if (derives(ot, x)) {
                    for (int m : {ta[ot], tb[ot], tc[ot]}) if (m != x) crit[m]++;
                    break;
                }
            } else if (d == 0) {
                for (int m : {ta[t], tb[t], tc[t]}) if (m != x) {
                    if (--crit[m] == 0) zeroed.push_back(m);
                }
            }
        }
    }
    void addTri(int a, int b, int c, uint8_t type) {
        if (a > b) swap(a, b);
        int t;
        if (!freeT.empty()) { t = freeT.back(); freeT.pop_back(); }
        else { if (tHigh >= MT) return; t = tHigh++; }
        ta[t] = a; tb[t] = b; tc[t] = c; tx[t] = a ^ b ^ c; tt[t] = type;
        nTri++;
        addInc(a, t);
        if (b != a) addInc(b, t);
        addInc(c, t);
    }
    void delTri(int t) {
        int a = ta[t], b = tb[t], c = tc[t];
        delInc(a, t);
        if (b != a) delInc(b, t);
        delInc(c, t);
        nTri--;
        freeT.push_back(t);
    }

    int addLight(u64 w) {
        int s = freeS.back(); freeS.pop_back();
        val[s] = w; alive[s] = true; deg[s] = 0; crit[s] = 0; sup[s] = -1; inc[s].clear();
        posAl[s] = nal; al[nal] = s; av[nal] = w; nal++;
        vmap.set(w, s);
        if (w == 1) one = s;
        for (int i = 0; i < nrel; i++) {
            auto& r = rel[i];
            if (r.k == 0) addTri(r.y, r.z, s, 0);
            else if (r.k == 1) addTri(r.y, r.z, s, 1);
            else if (r.k == 2) addTri(r.y, s, r.z, 0);
            else if (r.k == 3) addTri(r.y, s, r.z, 1);
            else if (r.k == 4) addTri(s, s, r.z, 0);
            else addTri(s, s, r.z, 1);
        }
        return s;
    }
    void delLight(int s) {
        while (!inc[s].empty()) delTri(inc[s].back());
        if (one == s) one = -1;
        vmap.erase(val[s]);
        alive[s] = false; isS[s] = false; isT[s] = false; sup[s] = -1;
        int p = posAl[s];
        nal--;
        if (p < nal) {
            int last = al[nal]; al[p] = last; av[p] = av[nal]; posAl[last] = p;
        }
        freeS.push_back(s);
    }

    template <bool ADD>
    void pairs(u64 w) {
        auto fn = [&](u64 r) { if constexpr (ADD) pt.inc(r); else pt.dec(r); };
        if (w <= L / 2) fn(w + w);
        if (w > 1 && w <= L / w) fn(w * w);
        u64 Lw = L / w;
        for (int i = 0; i < nal; i++) {
            u64 y = av[i];
            if (y == w) continue;
            if (y <= L - w) fn(y + w);
            if (y > w) { fn(y - w); if (w > 1 && y % w == 0) fn(y / w); }
            else { fn(w - y); if (y > 1 && w % y == 0) fn(w / y); }
            if (w > 1 && y > 1 && y <= Lw) fn(y * w);
        }
    }

    void rebuildPairs() {
        int b = 15;
        while ((1ULL << b) < (size_t)nal * nal * 5) b++;
        pt.init(min(b, 20));
        int n = nal;
        for (nal = 0; nal < n;) { u64 w = av[nal]; nal++; pairs<true>(w); }
    }

    int16_t rem_[MT]; uint8_t tk[MT]; bool vis[MV]; int que[MV];
    bool closureOK(int u) {
        memcpy(rem_, tx, sizeof(int16_t) * tHigh); memset(tk, 0, tHigh); memset(vis, 0, sizeof vis);
        if (u >= 0) vis[u] = true;
        int qh = 0, qt = 0;
        for (int s : startSlots) { vis[s] = true; que[qt++] = s; }
        if (one >= 0 && one != u && !vis[one]) { vis[one] = true; que[qt++] = one; }
        while (qh < qt) {
            int v = que[qh++];
            for (int t : inc[v]) {
                int th;
                if (ta[t] == tb[t]) { if (ta[t] != v) continue; th = tc[t]; }
                else { rem_[t] ^= v; if (++tk[t] != 2) continue; th = rem_[t]; }
                if (!vis[th]) { vis[th] = true; que[qt++] = th; }
            }
        }
        return qt == nal - (u >= 0 ? 1 : 0);
    }

    void buildSupport() {
        for (int i = 0; i < nal; i++) sup[al[i]] = -1;
        vector<int> q; memset(vis, 0, sizeof vis); memcpy(rem_, tx, sizeof(int16_t) * tHigh); memset(tk, 0, tHigh);
        for (int s : startSlots) { vis[s] = true; q.push_back(s); }
        if (one >= 0 && !vis[one]) { vis[one] = true; q.push_back(one); }
        for (size_t h = 0; h < q.size(); h++) {
            int v = q[h];
            for (int t : inc[v]) {
                int th;
                if (ta[t] == tb[t]) { if (ta[t] != v) continue; th = tc[t]; }
                else { rem_[t] ^= v; if (++tk[t] != 2) continue; th = rem_[t]; }
                if (!vis[th]) { vis[th] = true; sup[th] = t; q.push_back(th); }
            }
        }
    }

    bool lost[MV], gone[MV] = {};
    vector<int> L1, Q1;
    vector<pair<int, int>> ns;
    int freeX = -1;
    bool freeMask[MV] = {};

    inline bool partnersOK(int t, int x) const {
        int a = ta[t], b = tb[t], c = tc[t];
        if (a == b) return c == x && !lost[a];
        return (a == x || !lost[a]) && (b == x || !lost[b]) && (c == x || !lost[c]);
    }

    bool tryRemove(int u) {
        L1.clear(); ns.clear(); L1.push_back(u); lost[u] = true;
        for (size_t h = 0; h < L1.size(); h++) {
            int v = L1[h];
            for (int t : inc[v]) {
                int a = ta[t], b = tb[t], c = tc[t];
                if (a != v && sup[a] == t && !lost[a] && a != freeX && !freeMask[a]) { lost[a] = true; L1.push_back(a); }
                if (b != v && sup[b] == t && !lost[b] && b != freeX && !freeMask[b]) { lost[b] = true; L1.push_back(b); }
                if (c != v && sup[c] == t && !lost[c] && c != freeX && !freeMask[c]) { lost[c] = true; L1.push_back(c); }
            }
        }
        size_t need = L1.size() - 1; Q1.clear();
        for (size_t i = 1; i < L1.size(); i++) {
            int x = L1[i]; if (!lost[x]) continue;
            for (int t : inc[x]) if (partnersOK(t, x)) { lost[x] = false; ns.push_back({x, t}); Q1.push_back(x); break; }
            while (!Q1.empty()) {
                int v = Q1.back(); Q1.pop_back();
                for (int t : inc[v]) {
                    int a = ta[t], b = tb[t], c = tc[t];
                    for (int y : {a, b, c}) if (y != v && y != u && lost[y] && !gone[y] && partnersOK(t, y)) {
                        lost[y] = false; ns.push_back({y, t}); Q1.push_back(y);
                    }
                }
            }
        }
        bool ok = ns.size() == need;
        for (int x : L1) lost[x] = false;
        return ok;
    }

    void commitRemove(int u) {
        auto touch = [&](int t) { if (t < 0) return; for (int y : {ta[t], tb[t], tc[t]}) zeroed.push_back(y); };
        for (auto& [x, t] : ns) { touch(sup[x]); sup[x] = t; }
        touch(sup[u]);
        pairs<false>(val[u]);
        delLight(u);
    }

    int pruneZeroed(int keep) {
        int removed = 0;
        while (!zeroed.empty()) {
            int j = rng(zeroed.size()); int u = zeroed[j]; zeroed[j] = zeroed.back(); zeroed.pop_back();
            if (u == keep || !alive[u] || !removable(u) || crit[u] != 0) continue;
            if (tryRemove(u)) { commitRemove(u); removed++; }
        }
        return removed;
    }

    int pruneAll(int keep) {
        int removed = 0; bool ch = true;
        while (ch) {
            ch = false;
            vector<int> o;
            for (int i = 0; i < nal; i++) if (removable(al[i]) && crit[al[i]] == 0 && al[i] != keep) o.push_back(al[i]);
            for (size_t i = o.size(); i > 1; i--) swap(o[i - 1], o[rng(i)]);
            for (int u : o) if (alive[u] && crit[u] == 0 && tryRemove(u)) { commitRemove(u); removed++; ch = true; }
        }
        zeroed.clear();
        return removed;
    }

    // ==================== UPGRADE 1: PEAK-BUSTER + COUPLED RE-DERIVE SCHEDULER ====================
    struct Sch { vector<int> seq; int da[MV], db[MV]; uint8_t op[MV]; } sc;
    int idxOf[MV], lastU[MV];
    vector<int> die_;
    static inline int opOf(int t2, int kind) { return t2 == 0 ? (kind == 0 ? '+' : '-') : (kind == 0 ? '*' : '/'); }

    void tightenDerivations() {
        int n = (int)sc.seq.size();
        for (int s : startSlots) idxOf[s] = -1;
        for (int i = 0; i < n; i++) idxOf[sc.seq[i]] = i;
        for (int pass = 0; pass < 2; pass++) {
            int cur = overflow(P.regs);
            if (!cur) return;
            for (int i = 0; i < n; i++) {
                int x = sc.seq[i];
                if (x == one || inc[x].size() < 2) continue;
                int bestP = sc.da[x], bestQ = sc.db[x], bestO = sc.op[x], bestC = cur;
                for (int t : inc[x]) {
                    int a = ta[t], b = tb[t], c = tc[t], p, q, op;
                    if (a == b) { if (c != x) continue; p = q = a; op = opOf(tt[t], 0); }
                    else if (c == x) { p = a; q = b; op = opOf(tt[t], 0); }
                    else { p = c; q = (a == x) ? b : a; op = opOf(tt[t], 1); }
                    if (idxOf[p] >= i || idxOf[q] >= i || (p == bestP && q == bestQ)) continue;
                    sc.da[x] = p; sc.db[x] = q; sc.op[x] = op;
                    int c2 = overflow(P.regs);
                    if (c2 < bestC) { bestC = c2; bestP = p; bestQ = q; bestO = op; }
                }
                sc.da[x] = bestP; sc.db[x] = bestQ; sc.op[x] = bestO;
                cur = bestC;
                if (!cur) return;
            }
        }
    }

    bool initSchedule(int mode) {
        sc.seq.clear();
        vector<char> inq(MV, 0), done(MV, 0);
        int anyS = startSlots[0];
        vector<int> ready;

        auto pushReady = [&](int v) {
            for (int t : inc[v]) {
                int a = ta[t], b = tb[t], c = tc[t];
                if (a == b) { if (a == v && !inq[c]) { inq[c] = 1; ready.push_back(c); } continue; }
                int cnt = done[a] + done[b] + done[c];
                if (cnt != 2) continue;
                int th = !done[a] ? a : !done[b] ? b : c;
                if (!inq[th]) { inq[th] = 1; ready.push_back(th); }
            }
        };

        if (mode == 3 || mode == 4) {
            if (!initSchedule(mode == 3 ? 0 : 2)) return false;
            int n = (int)sc.seq.size();
            vector<vector<int>> adj(MV);
            int indeg[MV] = {}, remU[MV] = {};
            for (int x : sc.seq) {
                int a = sc.da[x], b = sc.db[x];
                remU[a]++; if (b != a) remU[b]++;
                if (!isS[a]) { adj[a].push_back(x); indeg[x]++; }
                if (!isS[b] && b != a) { adj[b].push_back(x); indeg[x]++; }
            }
            vector<int> rdy;
            for (int x : sc.seq) if (indeg[x] == 0) rdy.push_back(x);
            vector<int> newSeq; newSeq.reserve(n);
            while (!rdy.empty()) {
                int pick = 0; double bestSc = -1e18;
                for (int i = 0; i < (int)rdy.size(); i++) {
                    int x = rdy[i], a = sc.da[x], b = sc.db[x];
                    int freed = (remU[a] == 1 ? 1 : 0) + ((b != a && remU[b] == 1) ? 1 : 0);
                    int creates = (remU[x] > 0) ? 1 : 0;
                    int unlocks = 0;
                    for (int ch : adj[x]) if (indeg[ch] == 1) unlocks++;
                    double scv = (freed - creates) * 100.0 + (remU[x] == 0 ? 60.0 : -remU[x] * 3.0)
                               + unlocks * 4.0 - posv[x] * 0.05 + rng.u() * 14.0;
                    if (scv > bestSc) { bestSc = scv; pick = i; }
                }
                int x = rdy[pick];
                rdy[pick] = rdy.back(); rdy.pop_back();
                newSeq.push_back(x);
                int a = sc.da[x], b = sc.db[x];
                remU[a]--; if (b != a) remU[b]--;
                for (int ch : adj[x]) if (--indeg[ch] == 0) rdy.push_back(ch);
            }
            if ((int)newSeq.size() == n) {
                sc.seq = std::move(newSeq);
                tightenDerivations();
                return true;
            }
            return false;
        }

        int stepIdx[MV]; memset(stepIdx, 255, sizeof stepIdx);
        for (int s : startSlots) { done[s] = 1; inq[s] = 1; stepIdx[s] = 0; }
        for (int s : startSlots) pushReady(s);
        if (one >= 0 && !isS[one] && !inq[one]) { inq[one] = 1; ready.push_back(one); }

        while (!ready.empty()) {
            int pick = 0;
            if (mode == 0) {
                double bestP = 1e18;
                for (int i = 0; i < (int)ready.size(); i++) {
                    if (posv[ready[i]] < bestP) { bestP = posv[ready[i]]; pick = i; }
                }
            } else if (mode == 1) {
                double bestSc = 1e18;
                for (int i = 0; i < (int)ready.size(); i++) {
                    int x = ready[i];
                    double scv = posv[x] * 0.4 + (isT[x] ? -12.0 : 0.0) + (double)inc[x].size() * 0.6 + rng.u() * 5.0;
                    if (scv < bestSc) { bestSc = scv; pick = i; }
                }
            } else {
                double bestP = 1e18;
                for (int i = 0; i < (int)ready.size(); i++) {
                    double pv = posv[ready[i]] + rng.u() * 24.0;
                    if (pv < bestP) { bestP = pv; pick = i; }
                }
            }
            int x = ready[pick];
            ready[pick] = ready.back(); ready.pop_back();

            bool ok = false;
            int bestSpan = -1e9;
            if (x == one) { sc.da[x] = sc.db[x] = anyS; sc.op[x] = '/'; ok = true; bestSpan = 0; }
            int nd = 0;
            for (int t : inc[x]) {
                int a = ta[t], b = tb[t], c = tc[t], p = -1, q = -1, op = 0;
                if (a == b) { if (c != x || !done[a]) continue; p = q = a; op = opOf(tt[t], 0); }
                else if (c == x) { if (!done[a] || !done[b]) continue; p = a; q = b; op = opOf(tt[t], 0); }
                else { int o = (a == x) ? b : a; if (!done[c] || !done[o]) continue; p = c; q = o; op = opOf(tt[t], 1); }
                if (mode <= 1) {
                    int span = stepIdx[p] + stepIdx[q] + (int)(rng(3));
                    if (!ok || span > bestSpan) { bestSpan = span; sc.da[x] = p; sc.db[x] = q; sc.op[x] = op; ok = true; }
                } else {
                    if (rng(++nd) == 0) { sc.da[x] = p; sc.db[x] = q; sc.op[x] = op; ok = true; }
                }
            }
            if (!ok) return false;
            done[x] = 1;
            stepIdx[x] = (int)sc.seq.size() + 1;
            sc.seq.push_back(x);
            pushReady(x);
        }
        return (int)sc.seq.size() == nal - nStart;
    }

    int overflow(int limit, int* maxPeak = nullptr, int* peakPos = nullptr) {
        int n = (int)sc.seq.size();
        if ((int)die_.size() < n + 1) die_.assign(n + 1, 0);
        memset(lastU, 255, sizeof lastU);
        for (int i = 0; i < n; i++) { int x = sc.seq[i]; lastU[sc.da[x]] = i; lastU[sc.db[x]] = i; }
        std::fill(die_.begin(), die_.begin() + n + 1, 0);
        int cur = 0, ov = 0, pk = 0, ppos = -1;
        for (int s : startSlots) if (lastU[s] >= 0) { cur++; die_[lastU[s]]++; }
        for (int i = 0; i < n; i++) {
            int x = sc.seq[i];
            int need = cur - die_[i] + 1;
            if (need > pk) { pk = need; ppos = i; }
            if (need > limit) ov += (need - limit);
            cur -= die_[i];
            if (lastU[x] > i) { cur++; die_[lastU[x]]++; }
        }
        if (maxPeak) *maxPeak = pk;
        if (peakPos) *peakPos = ppos;
        return ov == 0 ? 0 : (ov * 2 + (pk - limit) * 8);
    }

    bool annealOrder(int iters) {
        int n = (int)sc.seq.size();
        int pk = 0, ppos = -1;
        int cur = overflow(P.regs, &pk, &ppos);
        if (!cur) return true;
        for (int i = 0; i < n; i++) idxOf[sc.seq[i]] = i;
        for (int s : startSlots) idxOf[s] = -1;
        vector<vector<int>> users(MV);
        for (int x : sc.seq) {
            users[sc.da[x]].push_back(x);
            if (sc.db[x] != sc.da[x]) users[sc.db[x]].push_back(x);
        }
        auto rmUser = [&](int p, int x) {
            auto& v = users[p];
            for (size_t i = 0; i < v.size(); i++) if (v[i] == x) { v[i] = v.back(); v.pop_back(); return; }
        };
        auto mv = [&](int f, int t) {
            if (f < t) rotate(sc.seq.begin() + f, sc.seq.begin() + f + 1, sc.seq.begin() + t + 1);
            else if (t < f) rotate(sc.seq.begin() + t, sc.seq.begin() + f, sc.seq.begin() + f + 1);
        };

        int waveLen = max(1, iters / 3);
        for (int it = 0; it < iters; it++) {
            double phase = (double)(it % waveLen) / waveLen;
            double T = 3.2 * pow(0.02, phase) + 0.04;

            int roll = (int)rng(20);
            // Move A (20%): Coupled Re-Derive + Topological Relocate!
            if (roll < 4) {
                int i = rng(n), x = sc.seq[i];
                if (x == one || inc[x].size() < 2) continue;
                const auto& I = inc[x];
                int t = I[rng(I.size())];
                int a = ta[t], b = tb[t], c = tc[t], p, q, op;
                if (a == b) { if (c != x) continue; p = q = a; op = opOf(tt[t], 0); }
                else if (c == x) { p = a; q = b; op = opOf(tt[t], 0); }
                else { p = c; q = (a == x) ? b : a; op = opOf(tt[t], 1); }
                if (p == sc.da[x] && q == sc.db[x]) continue;

                int hi = n - 1;
                for (int u : users[x]) hi = min(hi, idxOf[u] - 1);
                int maxP = max(idxOf[p], idxOf[q]);
                if (maxP > hi) continue; // would create cycle or invalid order

                int j = i;
                if (maxP >= i) {
                    int span = min(hi - maxP + 1, 12);
                    j = maxP + (span > 1 ? (int)rng(span) : 0);
                }
                int oa = sc.da[x], ob = sc.db[x], oo = sc.op[x];
                sc.da[x] = p; sc.db[x] = q; sc.op[x] = op;
                if (j != i) mv(i, j);
                int pk2, ppos2;
                int c2 = overflow(P.regs, &pk2, &ppos2);
                if (c2 <= cur || rng.u() < exp((cur - c2) / T)) {
                    cur = c2; ppos = ppos2;
                    rmUser(oa, x); if (ob != oa) rmUser(ob, x);
                    users[p].push_back(x); if (q != p) users[q].push_back(x);
                    if (j != i) for (int k = min(i, j); k <= max(i, j); k++) idxOf[sc.seq[k]] = k;
                    if (!cur) return true;
                } else {
                    if (j != i) mv(j, i);
                    sc.da[x] = oa; sc.db[x] = ob; sc.op[x] = oo;
                }
                continue;
            }

            // Move B (35%): Peak-Buster Targeted Relocation (push birth past peak or pull last-use before peak)
            int i = -1, preferDir = 0; // +1 push right, -1 pull left
            if (roll < 11 && ppos >= 0 && ppos < n) {
                // Collect live values at ppos
                int probe = (int)rng(ppos + 1);
                int v = sc.seq[probe];
                int lu = -1;
                for (int u : users[v]) lu = max(lu, idxOf[u]);
                if (lu >= ppos) {
                    if (rng(2) == 0) { i = probe; preferDir = 1; }
                    else { i = lu; preferDir = -1; }
                }
            }
            if (i < 0) i = rng(n);
            int x = sc.seq[i];
            int lo = max(idxOf[sc.da[x]], idxOf[sc.db[x]]) + 1, hi = n - 1;
            for (int u : users[x]) hi = min(hi, idxOf[u] - 1);
            if (hi <= lo) continue;
            int j;
            if (preferDir == 1 && hi > i) {
                j = (hi >= ppos && rng(2) == 0) ? (ppos + (int)rng(hi - ppos + 1)) : (i + 1 + (int)rng(hi - i));
            } else if (preferDir == -1 && lo < i) {
                j = (lo <= ppos && rng(2) == 0) ? (lo + (int)rng(ppos - lo + 1)) : (lo + (int)rng(i - lo));
            } else {
                j = lo + rng(hi - lo + 1);
            }
            if (j == i) continue;
            mv(i, j);
            int pk2, ppos2;
            int c2 = overflow(P.regs, &pk2, &ppos2);
            if (c2 <= cur || rng.u() < exp((cur - c2) / T)) {
                cur = c2; ppos = ppos2;
                for (int k = min(i, j); k <= max(i, j); k++) idxOf[sc.seq[k]] = k;
                if (!cur) return true;
            } else mv(j, i);
        }
        return false;
    }

    bool emitSchedule(vector<array<int, 4>>& out) {
        int n = (int)sc.seq.size();
        memset(lastU, 255, sizeof lastU);
        for (int i = 0; i < n; i++) { int x = sc.seq[i]; lastU[sc.da[x]] = i; lastU[sc.db[x]] = i; }
        vector<int> reg(MV, -1), occ(P.regs, -1);
        for (int k = 0; k < (int)P.start.size(); k++) {
            int s = look(P.start[k]);
            if (s >= 0 && reg[s] < 0 && lastU[s] >= 0) { reg[s] = k; occ[k] = s; }
        }
        out.clear();
        for (int i = 0; i < n; i++) {
            int x = sc.seq[i], a = sc.da[x], b = sc.db[x], r1 = reg[a], r2 = reg[b];
            if (r1 < 0 || r2 < 0) return false;
            if (lastU[a] == i) occ[r1] = -1;
            if (lastU[b] == i) occ[r2] = -1;
            int d = -1; for (int r = 0; r < P.regs; r++) if (occ[r] < 0) { d = r; break; }
            if (d < 0) return false;
            out.push_back({sc.op[x], r1, r2, d});
            reg[x] = d;
            if (lastU[x] > i) occ[d] = x;
        }
        return true;
    }

    struct RematPlan { int p = -1, q = -1, op = 0, cost = 0; };

    bool emitWithRemat(vector<array<int, 4>>& out, int maxExtra) {
        if (maxExtra <= 0) return false;
        int n = (int)sc.seq.size();
        vector<vector<int>> usePos(MV);
        auto addUse = [&](int v, int pos) {
            auto& u = usePos[v];
            auto it = lower_bound(u.begin(), u.end(), pos);
            if (it == u.end() || *it != pos) u.insert(it, pos);
        };
        for (int i = 0; i < n; i++) {
            int x = sc.seq[i];
            addUse(sc.da[x], i);
            if (sc.db[x] != sc.da[x]) addUse(sc.db[x], i);
        }
        vector<int> reg(MV, -1), occ(P.regs, -1);
        for (int k = 0; k < (int)P.start.size(); k++) {
            int s = look(P.start[k]);
            if (s >= 0 && reg[s] < 0 && !usePos[s].empty()) { reg[s] = k; occ[k] = s; }
        }
        auto nextUse = [&](int v, int after) {
            auto& u = usePos[v];
            auto it = upper_bound(u.begin(), u.end(), after);
            return (it == u.end()) ? -1 : *it;
        };

        vector<RematPlan> plan(MV);
        int extra = 0;

        auto pickEvict = [&](int step, const array<int, 4>& prot, int& outR) -> bool {
            int remBud = maxExtra - extra;
            int bestR = -1, bestScore = -1;
            RematPlan bestPlan;
            for (int r = 0; r < P.regs; r++) {
                int v = occ[r];
                if (v < 0) { outR = r; return true; }
                if (v == prot[0] || v == prot[1] || v == prot[2] || v == prot[3]) continue;
                int nu = nextUse(v, step);
                if (nu < 0) {
                    reg[v] = -1; occ[r] = -1; outR = r;
                    return true;
                }
                if (remBud <= 0) continue;
                if (v == one) {
                    int scv = 600000 + nu;
                    if (scv > bestScore) { bestScore = scv; bestR = r; bestPlan = {-2, -2, '/', 1}; }
                    continue;
                }
                for (int t : inc[v]) {
                    int a = ta[t], b = tb[t], c = tc[t], p = -1, q = -1, o = 0;
                    if (a == b) { if (c != v) continue; p = q = a; o = opOf(tt[t], 0); }
                    else if (c == v) { p = a; q = b; o = opOf(tt[t], 0); }
                    else { int other = (a == v) ? b : a; p = c; q = other; o = opOf(tt[t], 1); }
                    if (p == v || q == v || reg[p] < 0 || reg[q] < 0) continue;
                    int np = nextUse(p, step), nq = nextUse(q, step);
                    // Avoid extending a parent that has no remaining uses at all unless necessary
                    int ext = (np < nu ? (np < 0 ? 2 : 1) : 0) + (q != p && nq < nu ? (nq < 0 ? 2 : 1) : 0);
                    int scv = (500000 - ext * 90000) + nu;
                    if (scv > bestScore) { bestScore = scv; bestR = r; bestPlan = {p, q, o, 1}; }
                }
                if (val[v] == 2 && remBud >= 2) {
                    int scv = 110000 + nu;
                    if (scv > bestScore) { bestScore = scv; bestR = r; bestPlan = {-3, -3, '+', 2}; }
                }
            }
            if (bestR < 0) return false;
            int v = occ[bestR];
            int nu = nextUse(v, step);
            plan[v] = bestPlan;
            if (bestPlan.p >= 0 && nu >= 0) {
                addUse(bestPlan.p, nu);
                if (bestPlan.q != bestPlan.p) addUse(bestPlan.q, nu);
            }
            reg[v] = -1; occ[bestR] = -1;
            outR = bestR;
            return true;
        };

        out.clear();
        auto ensureInReg = [&](auto& self, int need, int i, int mainA, int mainB, int depth) -> bool {
            if (reg[need] >= 0) return true;
            if (depth > 6) return false;
            RematPlan rp = plan[need];
            if (rp.cost == 0) {
                if (need == one) rp = {-2, -2, '/', 1};
                else {
                    for (int t : inc[need]) {
                        int a = ta[t], b = tb[t], c = tc[t], p = -1, q = -1, o = 0;
                        if (a == b) { if (c != need) continue; p = q = a; o = opOf(tt[t], 0); }
                        else if (c == need) { p = a; q = b; o = opOf(tt[t], 0); }
                        else { int other = (a == need) ? b : a; p = c; q = other; o = opOf(tt[t], 1); }
                        if (reg[p] >= 0 && reg[q] >= 0) { rp = {p, q, o, 1}; break; }
                    }
                    if (rp.cost == 0 && val[need] == 2) rp = {-3, -3, '+', 2};
                }
            }
            if (rp.cost == 0) return false;
            if (rp.p >= 0) {
                if (!self(self, rp.p, i, mainA, mainB, depth + 1)) return false;
                if (!self(self, rp.q, i, rp.p, mainA, depth + 1)) return false;
            }
            if (extra + rp.cost > maxExtra) return false;
            extra += rp.cost;

            int p1 = -1, p2 = -1;
            if (rp.p == -2 || rp.p == -3) {
                for (int r = 0; r < P.regs; r++) if (occ[r] >= 0) { p1 = r; break; }
                if (p1 < 0) return false;
                p2 = p1;
            } else {
                p1 = reg[rp.p]; p2 = reg[rp.q];
                if (p1 < 0 || p2 < 0) return false;
            }

            // Free rp.p or rp.q right away if step i was their final use and they aren't mainA/mainB!
            if (rp.p >= 0 && rp.p != mainA && rp.p != mainB && nextUse(rp.p, i) < 0) {
                occ[p1] = -1; reg[rp.p] = -1;
            }
            if (rp.q >= 0 && rp.q != rp.p && rp.q != mainA && rp.q != mainB && nextUse(rp.q, i) < 0) {
                occ[p2] = -1; reg[rp.q] = -1;
            }

            int d = -1;
            for (int r = 0; r < P.regs; r++) if (occ[r] < 0) { d = r; break; }
            if (d < 0) {
                array<int, 4> prot = {mainA, mainB, rp.p >= 0 ? rp.p : -1, rp.q >= 0 ? rp.q : -1};
                if (!pickEvict(i, prot, d)) return false;
            }
            if (rp.p == -3) {
                out.push_back({'/', p1, p1, d});
                out.push_back({'+', d, d, d});
            } else {
                out.push_back({rp.op, p1, p2, d});
            }
            reg[need] = d; occ[d] = need;
            return true;
        };

        for (int i = 0; i < n; i++) {
            int x = sc.seq[i], a = sc.da[x], b = sc.db[x];
            if (!ensureInReg(ensureInReg, a, i, a, b, 0)) return false;
            if (!ensureInReg(ensureInReg, b, i, a, b, 0)) return false;
            int r1 = reg[a], r2 = reg[b];
            if (r1 < 0 || r2 < 0) return false;
            if (nextUse(a, i) < 0) { occ[r1] = -1; reg[a] = -1; }
            if (b != a && nextUse(b, i) < 0) { occ[r2] = -1; reg[b] = -1; }
            int d = -1;
            for (int r = 0; r < P.regs; r++) if (occ[r] < 0) { d = r; break; }
            if (d < 0) {
                array<int, 4> prot = {-1, -1, -1, -1};
                if (!pickEvict(i, prot, d)) return false;
            }
            out.push_back({sc.op[x], r1, r2, d});
            if (nextUse(x, i) >= 0) { reg[x] = d; occ[d] = x; }
        }
        return verifyProgExact(P, out);
    }

    bool schedule(vector<array<int, 4>>& out, int tries, int iters, bool allowRemat) {
        stSched++;
        vector<pair<int, Sch>> topSc;
        for (int t = 0; t < tries; t++) {
            int mode = (t == 0) ? 3 : (t == 1) ? 0 : (t == 2) ? 1 : (t % 2 == 1 ? 4 : 2);
            if (!initSchedule(mode)) continue;
            int ov = overflow(P.regs);
            if (ov == 0 || annealOrder(iters)) {
                if (emitSchedule(out) && verifyProgExact(P, out)) return true;
            }
            ov = overflow(P.regs);
            if (allowRemat && cost() + 1 < best) {
                topSc.push_back({ov, sc});
                sort(topSc.begin(), topSc.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
                if (topSc.size() > 4) topSc.pop_back();
            }
        }
        if (allowRemat && !topSc.empty() && cost() + 1 < best) {
            int maxExtra = min(8, best - cost() - 1);
            vector<array<int, 4>> bestOut;
            for (auto& [ov, candSc] : topSc) {
                sc = candSc;
                vector<array<int, 4>> candOut;
                if (emitWithRemat(candOut, maxExtra) && (int)candOut.size() < best) {
                    if (bestOut.empty() || candOut.size() < bestOut.size()) {
                        bestOut = std::move(candOut);
                        maxExtra = (int)bestOut.size() - cost() - 1;
                        if (maxExtra <= 0) break;
                    }
                }
            }
            if (!bestOut.empty()) {
                out = std::move(bestOut);
                stRemat++;
                return true;
            }
        }
        stSchedFail++;
        return false;
    }

    vector<u64> orderedSet() const {
        vector<u64> r;
        for (int x : sc.seq) r.push_back(val[x]);
        return r;
    }
    vector<u64> curSet() const {
        vector<u64> r;
        for (int i = 0; i < nal; i++) if (!isS[al[i]]) r.push_back(av[i]);
        return r;
    }

    void loadSet(const vector<u64>& S) {
        while (nal) delLight(al[nal - 1]);
        memset(isS, 0, sizeof isS); memset(isT, 0, sizeof isT);
        nStart = 0; startSlots.clear(); targetSlots.clear(); one = -1;
        auto put = [&](u64 v) { int s = look(v); if (s >= 0) return s; enumerate(v); return addLight(v); };
        vector<u64> st;
        for (u64 v : P.start) if (find(st.begin(), st.end(), v) == st.end()) st.push_back(v);
        for (u64 v : st) {
            enumerate(v);
            int s = addLight(v);
            isS[s] = true; startSlots.push_back(s); nStart++;
        }
        for (int i = 0; i < nal; i++) posv[al[i]] = -1;
        int ix = 0;
        for (u64 v : S) if (look(v) < 0) { int sl = put(v); posv[sl] = ix++; }
        for (u64 t : P.targets) {
            int s = look(t); assert(s >= 0);
            if (!isT[s]) { isT[s] = true; if (!isS[s]) targetSlots.push_back(s); }
        }
        rebuildPairs(); zeroed.clear(); memset(lost, 0, sizeof lost); memset(gone, 0, sizeof gone); buildSupport();
    }

    // ==================== UPGRADE 2 & 3: DEEP CONE MITM + S2-REMAINDER + S1-OFFSET POLLARD-RHO ====================
    static constexpr int BB = 21;
    static constexpr int SB = 18;
    vector<u64> bits_, s1, sorted_;
    vector<u64> s1k = vector<u64>(1 << SB, 0);
    vector<uint32_t> s1pos;
    u64 kv[MV]; int nk = 0;

    inline bool bit(u64 v) const { u64 h = hsh(v) >> (64 - BB); return (bits_[h >> 6] >> (h & 63)) & 1; }
    inline bool s1has(u64 v) const {
        size_t i = hsh(v) >> (64 - SB);
        while (s1k[i]) { if (s1k[i] == v) return true; i = (i + 1) & ((1 << SB) - 1); }
        return false;
    }
    inline bool s1put(u64 v) {
        size_t i = hsh(v) >> (64 - SB);
        while (s1k[i]) { if (s1k[i] == v) return false; i = (i + 1) & ((1 << SB) - 1); }
        if (s1pos.size() > (1u << SB) * 3 / 4) return false;
        s1k[i] = v; s1pos.push_back((uint32_t)i); return true;
    }
    inline bool inS1K(u64 w) const { return bit(w) && s1has(w); }
    inline bool inKVal(u64 v) const { int z = look(v); return z >= 0 && vis[z]; }

    template <class F>
    static inline void inv(u64 x, u64 a, u64 L, F f) {
        if (x > a) f(x - a); else if (a > x) f(a - x);
        if (x <= L - a) f(x + a);
        if (a > 1) {
            u64 q = x / a;
            if (q * a == x) f(q);
            if (a > x && a % x == 0) f(a / x);
            if ((u128)x * a <= (u128)L) f(x * a);
        }
    }

    void buildKandS1(const vector<int>& blocked) {
        memcpy(rem_, tx, sizeof(int16_t) * tHigh); memset(tk, 0, tHigh); memset(vis, 0, sizeof vis);
        for (int b : blocked) vis[b] = true;
        int qh = 0, qt = 0;
        for (int s : startSlots) if (!vis[s]) { vis[s] = true; que[qt++] = s; }
        if (one >= 0 && !vis[one] && !gone[one]) { vis[one] = true; que[qt++] = one; }
        while (qh < qt) {
            int v = que[qh++];
            for (int t : inc[v]) {
                int th;
                if (ta[t] == tb[t]) { if (ta[t] != v) continue; th = tc[t]; }
                else { rem_[t] ^= v; if (++tk[t] != 2) continue; th = rem_[t]; }
                if (!vis[th] && !gone[th]) { vis[th] = true; que[qt++] = th; }
            }
        }
        for (int b : blocked) vis[b] = false;
        nk = qt;
        for (int i = 0; i < qt; i++) kv[i] = val[que[i]];
        bits_.assign((1ULL << BB) >> 6, 0);
        s1.clear();
        for (uint32_t i : s1pos) s1k[i] = 0;
        s1pos.clear();
        auto put = [&](u64 r) {
            if (r == 0 || r > L) return;
            u64 h = hsh(r) >> (64 - BB);
            bits_[h >> 6] |= 1ULL << (h & 63);
            if (s1put(r)) s1.push_back(r);
        };
        for (int i = 0; i < nk; i++) {
            u64 y = kv[i], Ly = L / y;
            for (int j = 0; j <= i; j++) {
                u64 z = kv[j];
                if (z <= L - y) put(y + z);
                if (z <= Ly && z > 1 && y > 1) put(y * z);
                if (y > z) { put(y - z); if (z > 1 && y % z == 0) put(y / z); }
                else if (z > y) { put(z - y); if (y > 1 && z % y == 0) put(z / y); }
            }
        }
    }

    int cap = 3;
    double deepP = 0.80, facP = 0.85;

    bool applyReplacement(const vector<u64>& Wn, const vector<int>& Rm, double refPos, int maxNet) {
        if (Wn.empty() || nal + (int)Wn.size() >= MV - 4) return false;
        vector<int> added;
        zeroed.clear();
        auto addReal = [&](u64 w) {
            if (look(w) >= 0) return;
            enumerate(w);
            int s = addLight(w);
            sup[s] = -1;
            int nd = 0;
            for (int t : inc[s]) if (derives(t, s) && rng(++nd) == 0) sup[s] = t;
            posv[s] = refPos - 1e-3 * (1.0 + rng.u());
            if (sup[s] < 0) { delLight(s); return; }
            pairs<true>(w);
            added.push_back(s);
        };
        for (int pass = 0; pass < 4; pass++) {
            bool any = false;
            for (u64 w : Wn) if (look(w) < 0) {
                size_t prev = added.size();
                addReal(w);
                if (added.size() > prev) any = true;
            }
            if (!any) break;
        }
        int removed = 0;
        for (int r : Rm) {
            if (alive[r] && removable(r) && find(added.begin(), added.end(), r) == added.end() && crit[r] == 0 && tryRemove(r)) {
                commitRemove(r);
                removed++;
            }
        }
        int net = (int)added.size() - removed;
        if (net > maxNet) {
            for (size_t i = added.size(); i-- > 0;) {
                if (alive[added[i]] && tryRemove(added[i])) commitRemove(added[i]);
            }
            zeroed.clear();
            return false;
        }
        int r2 = pruneZeroed(-1);
        onState();
        return (net - r2) <= maxNet;
    }

    // Fast helper: find w in S1(K) such that R = w op k (k in K), returning w (or 0 if none)
    inline u64 findS1StepTo(u64 R, u64 forbid1, u64 forbid2) const {
        u64 hit = 0;
        for (int j = 0; j < nk && !hit; j++) {
            inv(R, kv[j], L, [&](u64 w) {
                if (!hit && w != forbid1 && w != forbid2 && inS1K(w)) {
                    int z = look(w);
                    if (z < 0 || gone[z]) hit = w;
                }
            });
        }
        return hit;
    }

    bool rebuildMove() {
        stRb++;
        int x;
        if (!targetSlots.empty() && rng(5) < 2) {
            x = targetSlots[rng(targetSlots.size())];
        } else {
            do x = al[rng(nal)]; while (fre(x));
        }
        vector<int> Rm; vector<pair<int, int>> undo; vector<int> qu;
        freeX = x;
        auto pushT = [&](int t) { if (t < 0) return; for (int y : {ta[t], tb[t], tc[t]}) if (y != x) qu.push_back(y); };
        pushT(sup[x]);
        while (!qu.empty() && Rm.size() < 14) {
            int j = rng(qu.size()), r = qu[j]; qu[j] = qu.back(); qu.pop_back();
            if (gone[r] || !removable(r)) continue;
            if (tryRemove(r)) {
                for (auto& [y, t] : ns) { undo.push_back({y, sup[y]}); pushT(sup[y]); sup[y] = t; }
                pushT(sup[r]); gone[r] = lost[r] = true; Rm.push_back(r);
            }
        }
        int f = (int)Rm.size();
        bool up1 = cost() + 1 <= best + cap;
        int lim = f + (up1 ? 1 : 0);
        vector<u64> Wn;
        int bestN = lim + 1, ties = 0;

        if (f >= 2 || (f == 1 && up1 && rng(4) == 0)) {
            buildKandS1({x});
            u64 X = val[x];
            auto inK = [&](u64 v) { int z = look(v); return z >= 0 && vis[z]; };

            // Offer an ordered dependency sequence `seq` whose elements derive X
            // checkAllS1 = true means every non-K element must directly be in S1(K);
            // checkAllS1 = false allows later elements in `seq` to depend on earlier elements in `seq`!
            auto offerSeq = [&](const vector<u64>& seq, bool checkAllS1) {
                vector<u64> nv;
                for (size_t i = 0; i < seq.size(); i++) {
                    u64 v = seq[i];
                    if (v == 0 || v > L || v == X) return;
                    if (inK(v)) continue;
                    int z = look(v);
                    if (z >= 0 && !gone[z]) return;
                    if (find(nv.begin(), nv.end(), v) != nv.end()) continue;
                    if (checkAllS1 && !inS1K(v)) return;
                    nv.push_back(v);
                }
                int n = (int)nv.size();
                if (n == 0 || n > lim || n > bestN) return;
                bool allOld = true;
                for (u64 v : nv) if (look(v) < 0) allOld = false;
                if (allOld) return;
                if (n < bestN) { bestN = n; ties = 0; }
                if (rng(++ties) == 0) Wn = nv;
            };

            // 1-new exact
            for (int i = 0; i < nk; i++) inv(X, kv[i], L, [&](u64 w) { if (bit(w)) offerSeq({w}, true); });

            // Build sorted U = K U S1(K) once for exact two-pointer & quotient search!
            bool builtSorted = false;
            auto ensureSorted = [&]() {
                if (builtSorted) return;
                builtSorted = true;
                sorted_ = s1;
                for (int i = 0; i < nk; i++) sorted_.push_back(kv[i]);
                sort(sorted_.begin(), sorted_.end());
                sorted_.erase(unique(sorted_.begin(), sorted_.end()), sorted_.end());
            };

            // 2-new exact: Two-Pointer Sum/Diff on S1(K) + K-step + Product/Div
            if (bestN > 1 && lim >= 2) {
                ensureSorted();
                size_t nb = sorted_.size();
                // Exact two-pointer w1 + w2 = X (covers 100% of S1+S1 in O(nb)!)
                if (nb >= 2 && sorted_[0] < X) {
                    size_t l = 0, r = lower_bound(sorted_.begin(), sorted_.end(), X) - sorted_.begin();
                    if (r >= nb) r = nb - 1;
                    int hits = 0;
                    while (l < r && hits < 128) {
                        u64 s = sorted_[l] + sorted_[r];
                        if (s == X) {
                            offerSeq({sorted_[l], sorted_[r]}, true);
                            hits++; l++; r--;
                        } else if (s < X) l++;
                        else r--;
                    }
                }
                // Exact two-pointer w2 - w1 = X (covers 100% of S1-S1 in O(nb)!)
                if (nb >= 2 && X < L) {
                    size_t l = 0, r = lower_bound(sorted_.begin(), sorted_.end(), X + sorted_[0]) - sorted_.begin();
                    int hits = 0;
                    while (r < nb && l < r && hits < 128) {
                        u64 d = sorted_[r] - sorted_[l];
                        if (d == X) {
                            offerSeq({sorted_[l], sorted_[r]}, true);
                            hits++; l++; r++;
                        } else if (d < X) r++;
                        else { l++; if (l == r) r++; }
                    }
                }
                // 2-step chain w1 -> w2 -> X via K
                int nh = 0, off = rng(nk);
                for (int ii = 0; ii < nk && nh < 500; ii++) {
                    int i = ii + off; if (i >= nk) i -= nk;
                    inv(X, kv[i], L, [&](u64 w2) {
                        if (w2 == X || look(w2) >= 0) return;
                        for (int j = 0; j < nk && nh < 500; j++) {
                            inv(w2, kv[j], L, [&](u64 w1) {
                                if (inS1K(w1)) { nh++; offerSeq({w1, w2}, false); }
                            });
                        }
                    });
                }
                // Sampled w1 * w2 = X or w1 / w2 = X from S1(K)
                size_t ns1 = s1.size(), o2 = ns1 ? rng(ns1) : 0;
                for (size_t q = 0; q < ns1 && q < 3000 && nh < 800; q++) {
                    u64 w1 = s1[q + o2 < ns1 ? q + o2 : q + o2 - ns1];
                    if (w1 > 1 && X % w1 == 0) {
                        u64 w2 = X / w1;
                        if (w2 != w1 && inS1K(w2)) { nh++; offerSeq({w1, w2}, true); }
                    }
                    if (w1 > X && w1 % X == 0) {
                        u64 w2 = w1 / X;
                        if (w2 != w1 && inS1K(w2)) { nh++; offerSeq({w1, w2}, true); }
                    }
                }
            }

            // Deep 3..6-new: S1 & S2-Remainder Division X = A * B +- R!
            if (bestN > 2 && lim >= 3 && f >= 2 && rng.u() < deepP) {
                ensureSorted();
                size_t nb = sorted_.size(), st = nb > 8000 ? nb / 8000 : 1, o = rng(st);
                int checks = 0, s2checks = 0;
                auto tryB = [&](u64 B, bool bInK) {
                    if (B < 2 || B > X) return;
                    u64 q = X / B;
                    size_t c = lower_bound(sorted_.begin(), sorted_.end(), q) - sorted_.begin();
                    int rad = bInK ? 6 : 3;
                    for (int d = -rad; d <= rad; d++) {
                        i64 ix = (i64)c + d;
                        if (ix < 0 || ix >= (i64)nb) continue;
                        u64 A = sorted_[ix];
                        if (A < 2 || (u128)A * B > (u128)L) continue;
                        u64 Pp = A * B;
                        if (Pp == X || Pp == 0) continue;
                        int zP = look(Pp);
                        if (zP >= 0 && !vis[zP] && !gone[zP]) continue;
                        u64 R = Pp > X ? Pp - X : X - Pp;
                        if (R == X || R == 0) continue;

                        bool aInK = inK(A), pInK = inK(Pp);
                        int baseCost = (!aInK ? 1 : 0) + (!bInK ? 1 : 0) + (!pInK ? 1 : 0);
                        if (baseCost >= bestN || baseCost > lim) continue;

                        if (inK(R)) {
                            vector<u64> seq;
                            if (!aInK) seq.push_back(A);
                            if (!bInK) seq.push_back(B);
                            if (!pInK) seq.push_back(Pp);
                            offerSeq(seq, false);
                        } else if (inS1K(R)) {
                            if (baseCost + 1 <= lim && baseCost + 1 <= bestN && checks < 25000) {
                                checks++;
                                vector<u64> seq;
                                if (!aInK) seq.push_back(A);
                                if (!bInK) seq.push_back(B);
                                seq.push_back(R);
                                if (!pInK) seq.push_back(Pp);
                                offerSeq(seq, false);
                            }
                        } else if (baseCost + 2 <= lim && baseCost + 2 <= bestN && s2checks < 1200) {
                            // NEW: S2-Remainder! R = w op k (w in S1(K), k in K) -> 2 nodes {w, R}!
                            int zR = look(R);
                            if (zR < 0 || gone[zR]) {
                                s2checks++;
                                u64 w = findS1StepTo(R, X, Pp);
                                if (w) {
                                    vector<u64> seq;
                                    if (!aInK) seq.push_back(A);
                                    if (!bInK) seq.push_back(B);
                                    seq.push_back(w);
                                    seq.push_back(R);
                                    if (!pInK) seq.push_back(Pp);
                                    offerSeq(seq, false);
                                }
                            }
                        }
                    }
                };
                for (int i = 0; i < nk; i++) tryB(kv[i], true);
                for (size_t i = o; i < nb; i += st) tryB(sorted_[i], inK(sorted_[i]));

                // Unshackled 60-bit 3-step chain (w1 op w2) op a = X
                for (int rep = 0; rep < 6 && checks < 25000 && bestN > 3; rep++) {
                    u64 a = kv[rng(nk)];
                    inv(X, a, L, [&](u64 y) {
                        if (look(y) >= 0) return;
                        if (rep < 3) {
                            // Two-pointer / inv over s1 for y
                            for (u64 w1 : s1) {
                                if (checks >= 25000) break;
                                inv(y, w1, L, [&](u64 w2) {
                                    if (w2 != w1 && inS1K(w2) && checks < 25000) { checks++; offerSeq({w1, w2, y}, false); }
                                });
                            }
                        } else {
                            for (int r2 = 0; r2 < 12 && checks < 25000; r2++) {
                                u64 b = kv[rng(nk)];
                                inv(y, b, L, [&](u64 z) {
                                    if (look(z) >= 0) return;
                                    u64 w1 = findS1StepTo(z, X, y);
                                    if (w1) { checks++; offerSeq({w1, z, y}, false); }
                                });
                            }
                        }
                    });
                }

                // UPGRADE 3: Multi-Offset Brent Pollard-Rho with BOTH K-offsets AND S1-offsets!
                if (bestN > 2 && f >= 3 && X >= (1ULL << 20) && rng.u() < facP) {
                    stFac++;
                    vector<pair<u64, int>> pf;
                    vector<u64> divs;
                    int nKTrials = 28;
                    int nS1Trials = (lim >= 3 && !s1.empty()) ? 16 : 0;
                    int totalTr = 1 + nKTrials + nS1Trials;
                    for (int tr = 0; tr < totalTr && bestN > 2; tr++) {
                        u64 r = 0;
                        bool rInS1 = false;
                        if (tr == 0) r = 0;
                        else if (tr <= nKTrials) r = kv[rng(nk)];
                        else {
                            r = s1[rng(s1.size())];
                            int zr = look(r);
                            if (zr >= 0 && !gone[zr]) continue;
                            rInS1 = true;
                        }
                        u64 cands[2] = {0, 0};
                        int nc = 0;
                        if (r == 0) cands[nc++] = X;
                        else {
                            if (X > r) cands[nc++] = X - r;
                            if (X <= L - r) cands[nc++] = X + r;
                        }
                        for (int ci = 0; ci < nc && bestN > 2; ci++) {
                            u64 N = cands[ci];
                            if (N < 4) continue;
                            int zN = look(N);
                            if (r > 0 && zN >= 0 && !vis[zN] && !gone[zN]) continue;
                            factor_fast(N, rng, pf);
                            divs.clear(); divs.push_back(1);
                            for (auto& [p, e] : pf) {
                                size_t cur_sz = divs.size();
                                u64 mult = 1;
                                for (int k = 1; k <= e && divs.size() < 256; k++) {
                                    mult *= p;
                                    for (size_t j = 0; j < cur_sz && divs.size() < 256; j++) {
                                        divs.push_back(divs[j] * mult);
                                    }
                                }
                            }
                            for (u64 A : divs) {
                                if (A < 2 || A >= N) continue;
                                u64 B = N / A;
                                if (B < A) continue;
                                bool aK = inK(A), bK = inK(B);
                                bool a1 = aK || inS1K(A);
                                bool b1 = bK || inS1K(B);
                                int extraR = (r == 0) ? 0 : (rInS1 ? 2 : 1); // {r, N} or {N}
                                if (a1 && b1) {
                                    int cEst = (!aK) + (!bK) + extraR;
                                    if (cEst <= lim && cEst <= bestN) {
                                        vector<u64> seq;
                                        if (!aK) seq.push_back(A);
                                        if (!bK) seq.push_back(B);
                                        if (rInS1) seq.push_back(r);
                                        if (r > 0) seq.push_back(N);
                                        size_t prevTies = ties;
                                        int prevBest = bestN;
                                        offerSeq(seq, false);
                                        if (bestN < prevBest || ties > prevTies) stFacHit++;
                                    }
                                } else if ((a1 || b1) && lim >= 3) {
                                    u64 good = a1 ? A : B;
                                    u64 hard = a1 ? B : A;
                                    bool goodK = inK(good);
                                    int cEst = (!goodK ? 1 : 0) + 2 + extraR;
                                    if (cEst <= lim && cEst <= bestN) {
                                        int zH = look(hard);
                                        if (zH < 0 || gone[zH]) {
                                            u64 w1 = findS1StepTo(hard, X, N);
                                            if (w1 && w1 != good && w1 != r) {
                                                vector<u64> seq;
                                                if (!goodK) seq.push_back(good);
                                                seq.push_back(w1);
                                                seq.push_back(hard);
                                                if (rInS1) seq.push_back(r);
                                                if (r > 0) seq.push_back(N);
                                                offerSeq(seq, false);
                                                if (!Wn.empty()) stFacHit++;
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

        for (size_t i = undo.size(); i-- > 0;) sup[undo[i].first] = undo[i].second;
        for (int r : Rm) gone[r] = lost[r] = false;
        freeX = -1;
        if (Wn.empty()) return false;
        stRbHit++;
        int c0 = cost();
        bool ok = applyReplacement(Wn, Rm, posv[x], up1 ? 1 : 0);
        if (ok && cost() < c0) stRbGain++;
        return ok;
    }

    // ==================== PILLAR 3A: 2-SINK HUB-KNOCKOUT & BRIDGE INTERSECTION ====================
    bool pairHubRebuild() {
        stPair++;
        int h = al[rng(nal)];
        if (!removable(h)) return false;
        vector<int> sinks;
        for (int t : inc[h]) {
            for (int y : {ta[t], tb[t], tc[t]}) {
                if (y != h && !fre(y) && sup[y] == t && find(sinks.begin(), sinks.end(), y) == sinks.end()) {
                    sinks.push_back(y);
                }
            }
        }
        if (sinks.size() < 2 || sinks.size() > 3) return false;
        vector<int> Rm; vector<pair<int, int>> undo;
        for (int y : sinks) freeMask[y] = true;
        vector<int> qu = {h};
        for (int y : sinks) {
            int t = sup[y];
            if (t >= 0) for (int z : {ta[t], tb[t], tc[t]}) if (z != y && !freeMask[z]) qu.push_back(z);
        }
        while (!qu.empty() && Rm.size() < 12) {
            int j = rng(qu.size()), r = qu[j]; qu[j] = qu.back(); qu.pop_back();
            if (gone[r] || !removable(r) || freeMask[r]) continue;
            if (tryRemove(r)) {
                for (auto& [y, t] : ns) { undo.push_back({y, sup[y]}); sup[y] = t; }
                int t = sup[r];
                if (t >= 0) for (int z : {ta[t], tb[t], tc[t]}) if (z != r && !freeMask[z]) qu.push_back(z);
                gone[r] = lost[r] = true; Rm.push_back(r);
            }
        }
        int f = (int)Rm.size();
        vector<u64> Wn;
        bool up1 = cost() + 1 <= best + cap;

        if (f >= 2 && sinks.size() == 2) {
            int y1 = sinks[0], y2 = sinks[1];
            u64 Y1 = val[y1], Y2 = val[y2];
            buildKandS1({y1, y2});

            vector<u64> I1, I2;
            for (int i = 0; i < nk; i++) {
                inv(Y1, kv[i], L, [&](u64 w) {
                    if (w != Y1 && w != Y2 && inS1K(w) && (look(w) < 0 || gone[look(w)])) I1.push_back(w);
                });
                inv(Y2, kv[i], L, [&](u64 w) {
                    if (w != Y1 && w != Y2 && inS1K(w) && (look(w) < 0 || gone[look(w)])) I2.push_back(w);
                });
            }
            sort(I1.begin(), I1.end()); I1.erase(unique(I1.begin(), I1.end()), I1.end());
            sort(I2.begin(), I2.end()); I2.erase(unique(I2.begin(), I2.end()), I2.end());

            if (!I1.empty() && !I2.empty()) {
                for (u64 w : I2) {
                    if (binary_search(I1.begin(), I1.end(), w) && look(w) < 0) { Wn = {w}; break; }
                }
            }
            if (Wn.empty() && (f >= 3 || (f == 2 && up1)) && !I1.empty() && !I2.empty()) {
                int tries = 0;
                for (size_t a = 0; a < I1.size() && Wn.empty() && tries < 64; a++) {
                    u64 w1 = I1[(a + rng(I1.size())) % I1.size()];
                    for (size_t b = 0; b < I2.size() && Wn.empty() && tries < 64; b++) {
                        u64 w2 = I2[(b + rng(I2.size())) % I2.size()];
                        tries++;
                        if (w1 != w2 && (look(w1) < 0 || look(w2) < 0)) { Wn = {w1, w2}; break; }
                    }
                }
            }
            if (Wn.empty() && f >= 3 && !I1.empty()) {
                for (u64 w1 : I1) {
                    inv(Y2, w1, L, [&](u64 w2) {
                        if (!Wn.empty()) return;
                        if (w2 != w1 && w2 != Y1 && w2 != Y2 && inS1K(w2) && (look(w2) < 0 || gone[look(w2)])) {
                            if (look(w1) < 0 || look(w2) < 0) Wn = {w1, w2};
                        }
                    });
                    if (!Wn.empty()) break;
                }
            }
            if (Wn.empty() && f >= 3 && !I1.empty()) {
                inv(Y2, Y1, L, [&](u64 w2) {
                    if (!Wn.empty()) return;
                    if (w2 != Y1 && w2 != Y2 && inS1K(w2) && (look(w2) < 0 || gone[look(w2)])) {
                        for (u64 w1 : I1) {
                            if (w1 != w2 && (look(w1) < 0 || look(w2) < 0)) { Wn = {w1, w2}; break; }
                        }
                    }
                });
            }
            if (Wn.empty() && f >= 3 && !I2.empty()) {
                inv(Y1, Y2, L, [&](u64 w1) {
                    if (!Wn.empty()) return;
                    if (w1 != Y1 && w1 != Y2 && inS1K(w1) && (look(w1) < 0 || gone[look(w1)])) {
                        for (u64 w2 : I2) {
                            if (w1 != w2 && (look(w1) < 0 || look(w2) < 0)) { Wn = {w2, w1}; break; }
                        }
                    }
                });
            }
        }

        for (size_t i = undo.size(); i-- > 0;) sup[undo[i].first] = undo[i].second;
        for (int r : Rm) gone[r] = lost[r] = false;
        for (int y : sinks) freeMask[y] = false;
        if (Wn.empty()) return false;
        stPairHit++;
        int c0 = cost();
        bool ok = applyReplacement(Wn, Rm, posv[h], up1 ? 1 : 0);
        if (ok && cost() < c0) stPairGain++;
        return ok;
    }

    // ==================== UPGRADE 4: ARBITRARY 2-TARGET JOINT RUIN & TARGET-BRIDGE SYNTHESIS ====================
    bool pairTargetRebuild() {
        if (targetSlots.size() < 2) return false;
        stTargPair++;
        int y1 = targetSlots[rng(targetSlots.size())];
        int y2 = -1;
        if (rng(2) == 0) {
            // Pick closest target in value to Y1 (maximizes chance of small |Y1 - Y2| or shared quotient!)
            u64 bestDiff = ~0ULL;
            for (int t : targetSlots) {
                if (t == y1) continue;
                u64 d = val[t] > val[y1] ? val[t] - val[y1] : val[y1] - val[t];
                if (d < bestDiff && rng(3) != 0) { bestDiff = d; y2 = t; }
            }
        }
        if (y2 < 0 || y2 == y1) {
            do y2 = targetSlots[rng(targetSlots.size())]; while (y2 == y1);
        }

        vector<int> Rm; vector<pair<int, int>> undo;
        freeMask[y1] = freeMask[y2] = true;
        vector<int> qu;
        for (int y : {y1, y2}) {
            int t = sup[y];
            if (t >= 0) for (int z : {ta[t], tb[t], tc[t]}) if (z != y && !freeMask[z]) qu.push_back(z);
        }
        while (!qu.empty() && Rm.size() < 16) {
            int j = rng(qu.size()), r = qu[j]; qu[j] = qu.back(); qu.pop_back();
            if (gone[r] || !removable(r) || freeMask[r]) continue;
            if (tryRemove(r)) {
                for (auto& [y, t] : ns) { undo.push_back({y, sup[y]}); sup[y] = t; }
                int t = sup[r];
                if (t >= 0) for (int z : {ta[t], tb[t], tc[t]}) if (z != r && !freeMask[z]) qu.push_back(z);
                gone[r] = lost[r] = true; Rm.push_back(r);
            }
        }
        int f = (int)Rm.size();
        bool up1 = cost() + 1 <= best + cap;
        int lim = f + (up1 ? 1 : 0);
        vector<u64> Wn;
        int bestN = lim + 1;

        if (f >= 4) {
            u64 Y1 = val[y1], Y2 = val[y2];
            buildKandS1({y1, y2});
            auto inK = [&](u64 v) { int z = look(v); return z >= 0 && vis[z]; };

            // Quick cone finder of size <= maxC for a target Z over current K & S1(K)
            auto findSmallCone = [&](u64 Z, int maxC, vector<u64>& out) -> bool {
                out.clear();
                if (maxC <= 0) return false;
                // 1 node
                for (int i = 0; i < nk; i++) {
                    u64 hit = 0;
                    inv(Z, kv[i], L, [&](u64 w) {
                        if (!hit && w != Y1 && w != Y2 && inS1K(w) && (look(w) < 0 || gone[look(w)])) hit = w;
                    });
                    if (hit) { out = {hit}; return true; }
                }
                if (maxC < 2) return false;
                // 2 nodes via K
                for (int i = 0; i < nk; i++) {
                    bool ok = false;
                    inv(Z, kv[i], L, [&](u64 w2) {
                        if (ok || w2 == Y1 || w2 == Y2 || (look(w2) >= 0 && !gone[look(w2)])) return;
                        u64 w1 = findS1StepTo(w2, Y1, Y2);
                        if (w1 && w1 != w2) { out = {w1, w2}; ok = true; }
                    });
                    if (ok) return true;
                }
                if (maxC < 3) return false;
                // 2..3 nodes via quotient Z = A*B +- R (B in K)
                if (sorted_.empty()) {
                    sorted_ = s1;
                    for (int i = 0; i < nk; i++) sorted_.push_back(kv[i]);
                    sort(sorted_.begin(), sorted_.end());
                    sorted_.erase(unique(sorted_.begin(), sorted_.end()), sorted_.end());
                }
                size_t nb = sorted_.size();
                for (int i = 0; i < nk; i++) {
                    u64 B = kv[i];
                    if (B < 2 || B > Z) continue;
                    u64 q = Z / B;
                    size_t c = lower_bound(sorted_.begin(), sorted_.end(), q) - sorted_.begin();
                    for (int d = -2; d <= 2; d++) {
                        i64 ix = (i64)c + d;
                        if (ix < 0 || ix >= (i64)nb) continue;
                        u64 A = sorted_[ix];
                        if (A < 2 || (u128)A * B > (u128)L) continue;
                        u64 Pp = A * B;
                        if (Pp == Z || Pp == Y1 || Pp == Y2 || (look(Pp) >= 0 && !inK(Pp) && !gone[look(Pp)])) continue;
                        u64 R = Pp > Z ? Pp - Z : Z - Pp;
                        int base = (!inK(A) ? 1 : 0) + (!inK(Pp) ? 1 : 0);
                        if (inK(R) && base <= maxC) {
                            if (!inK(A)) out.push_back(A);
                            if (!inK(Pp)) out.push_back(Pp);
                            return true;
                        }
                        if (inS1K(R) && (look(R) < 0 || gone[look(R)]) && base + 1 <= maxC) {
                            if (!inK(A)) out.push_back(A);
                            out.push_back(R);
                            if (!inK(Pp)) out.push_back(Pp);
                            return true;
                        }
                    }
                }
                return false;
            };

            sorted_.clear();
            // Strategy 1: Build Y1 in c1 <= 3 nodes, AND build D = |Y1 - Y2| (or Y2 / Y1) in c2 nodes so Y2 = Y1 +- D!
            u64 D = Y1 > Y2 ? Y1 - Y2 : Y2 - Y1;
            if (D > 0 && (look(D) < 0 || gone[look(D)] || inK(D))) {
                vector<u64> cY1, cD;
                int dSelf = inK(D) ? 0 : 1; // D itself is 1 helper node unless already in K!
                if (inK(D)) {
                    if (findSmallCone(Y1, min(3, f - 1), cY1)) { Wn = cY1; bestN = (int)Wn.size(); }
                } else if (inS1K(D)) {
                    if (findSmallCone(Y1, min(3, f - 2), cY1)) {
                        cY1.push_back(D);
                        Wn = cY1; bestN = (int)Wn.size();
                    }
                } else if (findSmallCone(Y1, min(3, f - 3), cY1)) {
                    int remBud = (f - 1) - (int)cY1.size() - dSelf;
                    if (remBud >= 1 && findSmallCone(D, min(3, remBud), cD)) {
                        vector<u64> comb = cY1;
                        for (u64 v : cD) if (find(comb.begin(), comb.end(), v) == comb.end()) comb.push_back(v);
                        if (find(comb.begin(), comb.end(), D) == comb.end()) comb.push_back(D);
                        if ((int)comb.size() < f) { Wn = comb; bestN = (int)Wn.size(); }
                    }
                }
            }

            // Strategy 2: Shared Quotient Base P = A * B serving BOTH Y1 = P +- R1 and Y2 = P +- R2!
            if (Wn.empty()) {
                if (sorted_.empty()) {
                    sorted_ = s1;
                    for (int i = 0; i < nk; i++) sorted_.push_back(kv[i]);
                    sort(sorted_.begin(), sorted_.end());
                    sorted_.erase(unique(sorted_.begin(), sorted_.end()), sorted_.end());
                }
                size_t nb = sorted_.size();
                for (int i = 0; i < nk && Wn.empty(); i++) {
                    u64 B = kv[i];
                    if (B < 2 || B > min(Y1, Y2)) continue;
                    u64 q1 = Y1 / B;
                    size_t c = lower_bound(sorted_.begin(), sorted_.end(), q1) - sorted_.begin();
                    for (int d = -2; d <= 2 && Wn.empty(); d++) {
                        i64 ix = (i64)c + d;
                        if (ix < 0 || ix >= (i64)nb) continue;
                        u64 A = sorted_[ix];
                        if (A < 2 || (u128)A * B > (u128)L) continue;
                        u64 Pp = A * B;
                        if (Pp == Y1 || Pp == Y2 || (look(Pp) >= 0 && !inK(Pp) && !gone[look(Pp)])) continue;
                        u64 R1 = Pp > Y1 ? Pp - Y1 : Y1 - Pp;
                        u64 R2 = Pp > Y2 ? Pp - Y2 : Y2 - Pp;
                        if (R1 == 0 || R2 == 0) continue;
                        int base = (!inK(A) ? 1 : 0) + (!inK(Pp) ? 1 : 0);
                        vector<u64> r1v, r2v;
                        bool ok1 = false, ok2 = false;
                        if (inK(R1)) ok1 = true;
                        else if (inS1K(R1) && (look(R1) < 0 || gone[look(R1)])) { r1v = {R1}; ok1 = true; }
                        else if (look(R1) < 0 || gone[look(R1)]) {
                            u64 w = findS1StepTo(R1, Y1, Y2);
                            if (w) { r1v = {w, R1}; ok1 = true; }
                        }
                        if (!ok1) continue;
                        if (inK(R2)) ok2 = true;
                        else if (inS1K(R2) && (look(R2) < 0 || gone[look(R2)])) { r2v = {R2}; ok2 = true; }
                        else if (look(R2) < 0 || gone[look(R2)]) {
                            u64 w = findS1StepTo(R2, Y1, Y2);
                            if (w) { r2v = {w, R2}; ok2 = true; }
                        }
                        if (!ok2) continue;
                        vector<u64> comb;
                        auto addU = [&](u64 v) {
                            if (!inK(v) && v != Y1 && v != Y2 && find(comb.begin(), comb.end(), v) == comb.end()) comb.push_back(v);
                        };
                        addU(A); addU(Pp);
                        for (u64 v : r1v) addU(v);
                        for (u64 v : r2v) addU(v);
                        if (!comb.empty() && (int)comb.size() < f && (int)comb.size() < bestN) {
                            bool bad = false;
                            for (u64 v : comb) if (look(v) >= 0 && !gone[look(v)]) bad = true;
                            if (!bad) { Wn = comb; bestN = (int)Wn.size(); }
                        }
                    }
                }
            }
        }

        for (size_t i = undo.size(); i-- > 0;) sup[undo[i].first] = undo[i].second;
        for (int r : Rm) gone[r] = lost[r] = false;
        freeMask[y1] = freeMask[y2] = false;
        if (Wn.empty()) return false;
        stTargPairHit++;
        int c0 = cost();
        bool ok = applyReplacement(Wn, Rm, min(posv[y1], posv[y2]), up1 ? 1 : 0);
        if (ok && cost() < c0) stTargPairGain++;
        return ok;
    }

    vector<u64> minCostSet, seedBasin;
    int minCostSeen = 1e9;
    bool inBurst = false;

    // Inline Cap=0 Equal-Cost Swap Burst: when we discover a c < best value set that overflows 48 regs by 1,
    // immediately explore neighboring equal-cost value sets (swapping long-lived hubs) to find one with peak <= 48!
    void relaxRegisterBurst(int targetCost) {
        if (inBurst) return;
        inBurst = true;
        vector<u64> savedS = curSet();
        int swapsDone = 0;
        for (int att = 0; att < 3500 && cost() <= targetCost && best > targetCost; att++) {
            if (pt.cand.empty()) break;
            if (pt.used > pt.mask * 0.6) rebuildPairs();
            u64 w = pt.cand[rng(pt.cand.size())];
            if (w == 1 || look(w) >= 0) continue;
            enumerate(w);
            bool der = false;
            for (int i = 0; i < nrel; i++) if (rel[i].k < 4) { der = true; break; }
            if (!der || nrel < 2 || nal >= MV - 4 || nTri + nrel > MT - 16) continue;

            lst.clear(); U.clear(); ep++;
            auto gain = [&](int x) {
                if (fre(x) || mark[x] == ep) return;
                mark[x] = ep;
                int t = sup[x]; if (t < 0) return;
                for (int p : {ta[t], tb[t], tc[t]}) if (p != x && removable(p) && mark2[p] != ep) {
                    mark2[p] = ep; lst.push_back(p);
                }
            };
            for (int i = 0; i < nrel; i++) {
                auto& r = rel[i];
                if (r.k < 4) { gain(r.y); gain(r.z); } else gain(r.z);
            }
            for (int p : lst) {
                int c = crit[p];
                if (c) {
                    for (int t : inc[p]) for (int x : {ta[t], tb[t], tc[t]}) {
                        if (x != p && !fre(x) && deg[x] == 1 && mark[x] == ep && derives(t, x) && mark3[x] != p) {
                            mark3[x] = p; c--;
                        }
                    }
                    for (int t : inc[p]) for (int x : {ta[t], tb[t], tc[t]}) mark3[x] = -1;
                }
                if (c <= 0) U.push_back(p);
            }
            if (U.empty()) continue;
            zeroed.clear();
            int s = addLight(w);
            sup[s] = -1;
            int nd = 0;
            for (int t : inc[s]) if (derives(t, s) && rng(++nd) == 0) sup[s] = t;
            if (sup[s] < 0) { delLight(s); zeroed.clear(); continue; }
            posv[s] = rng.u() * 100.0;
            int u = -1;
            for (size_t i = U.size(); i > 1; i--) swap(U[i - 1], U[rng(i)]);
            for (int x : U) if (crit[x] == 0 && tryRemove(x)) { u = x; break; }
            if (u < 0) { delLight(s); zeroed.clear(); continue; }
            pairs<true>(w);
            commitRemove(u);
            pruneZeroed(s);
            swapsDone++;
            if (cost() < minCostSeen && closureOK(-1)) { minCostSeen = cost(); minCostSet = curSet(); }
            if (swapsDone % 3 == 0) {
                vector<array<int, 4>> out;
                if (schedule(out, 12, 90000, true) && (int)out.size() < best) {
                    best = (int)out.size();
                    bestProg = out;
                    bestSet = orderedSet();
                    feasSet = bestSet;
                    seedBasin.clear();
                    stDown++;
                    inBurst = false;
                    return;
                }
            }
        }
        inBurst = false;
    }

    void onState(int tries = 24, int iters = 160000) {
        int c = cost();
        if (c < minCostSeen && closureOK(-1)) {
            minCostSeen = c;
            minCostSet = curSet();
        }
        if (c < best) {
            vector<array<int, 4>> out;
            if (schedule(out, tries, iters, true) && (int)out.size() < best) {
                best = (int)out.size();
                bestProg = out;
                bestSet = orderedSet();
                feasSet = bestSet;
                seedBasin.clear();
                stDown++;
            } else if (!inBurst && c < best) {
                relaxRegisterBurst(c);
            }
        }
    }

    int mark[MV], mark2[MV], mark3[MV], ep = 0;
    vector<int> lst, U;

    void run(double secs, double T_init, int cap_, double rbFrac, double pairFrac, int crossEvery) {
        cap = cap_;
        double t0 = now_sec(), tReset = t0;
        double resetPeriod = (cap == 0) ? max(25.0, secs * 0.75) : max(8.0, secs * 0.25);
        memset(mark, 0, sizeof mark); memset(mark2, 0, sizeof mark2); memset(mark3, 255, sizeof mark3);
        feasSet = bestSet;
        if (minCostSet.empty()) { minCostSet = bestSet; minCostSeen = (int)bestSet.size(); }
        i64 accSince = 0;
        double rbTime = 0, pairTime = 0;

        for (i64 it = 0;; it++) {
            if ((it & 63) == 0) {
                double t = now_sec();
                if (t - t0 > secs) break;
                if (t - tReset > resetPeriod) {
                    tReset = t;
                    const vector<u64>& baseS = !seedBasin.empty()
                        ? seedBasin
                        : ((!minCostSet.empty() && minCostSeen < (int)bestSet.size() && rng(2) == 0) ? minCostSet : bestSet);
                    if (!variants.empty() && rng(2) == 0) {
                        stCross++;
                        vector<u64> u = baseS;
                        const auto& vv = variants[rng(variants.size())];
                        u.insert(u.end(), vv.begin(), vv.end());
                        if ((int)u.size() + nStart < MV - 8) {
                            loadSet(u);
                            pruneAll(-1);
                            onState();
                        } else loadSet(baseS);
                    } else {
                        loadSet(baseS);
                    }
                }
            }
            if (pt.cand.empty()) { loadSet(bestSet); if (pt.cand.empty()) break; }
            if (pt.used > pt.mask * 0.6) rebuildPairs();

            double elapsed = now_sec() - t0;
            double frac = min(1.0, elapsed / max(0.1, secs));
            double curT = T_init * pow(0.15, frac);
            double pUp = exp(-1.0 / max(0.05, curT));

            // 2-Sink Hub-Knockout OR Arbitrary 2-Target Joint Cone Synthesis!
            if (pairFrac > 0 && (it & 31) == 0 && pairTime < pairFrac * elapsed) {
                double tp0 = now_sec();
                if (rng(5) < 3) pairHubRebuild();
                else pairTargetRebuild();
                pairTime += now_sec() - tp0;
                if (pt.used > pt.mask * 0.6) rebuildPairs();
            }

            // Deep Cone MITM + S2-Remainder + S1-Offset Pollard-Rho Rebuild
            if (rbFrac > 0 && (it & 15) == 0 && rbTime < rbFrac * elapsed) {
                double tb0 = now_sec();
                rebuildMove();
                rbTime += now_sec() - tb0;
                if (pt.used > pt.mask * 0.6) rebuildPairs();
            }

            if (crossEvery > 0 && !variants.empty() && it > 0 && it % crossEvery == 0 && cost() >= best) {
                stCross++;
                vector<u64> u = curSet();
                const auto& vv = variants[rng(variants.size())];
                u.insert(u.end(), vv.begin(), vv.end());
                if ((int)u.size() + nStart < MV - 8) {
                    loadSet(u);
                    pruneAll(-1);
                    onState();
                }
            }

            u64 w = pt.cand[rng(pt.cand.size())];
            if (w == 1 || look(w) >= 0) continue;
            stProp++;
            enumerate(w);
            bool der = false;
            for (int i = 0; i < nrel; i++) if (rel[i].k < 4) { der = true; break; }
            if (!der || nrel < 2) continue;
            if (nal >= MV - 4 || nTri + nrel > MT - 16) continue;

            lst.clear(); U.clear(); ep++;
            auto gain = [&](int x) {
                if (fre(x) || mark[x] == ep) return;
                mark[x] = ep;
                int t = sup[x]; if (t < 0) return;
                for (int p : {ta[t], tb[t], tc[t]}) if (p != x && removable(p) && mark2[p] != ep) {
                    mark2[p] = ep; lst.push_back(p);
                }
            };
            for (int i = 0; i < nrel; i++) {
                auto& r = rel[i];
                if (r.k < 4) { gain(r.y); gain(r.z); } else gain(r.z);
            }
            for (int p : lst) {
                int c = crit[p];
                if (c) {
                    for (int t : inc[p]) for (int x : {ta[t], tb[t], tc[t]}) {
                        if (x != p && !fre(x) && deg[x] == 1 && mark[x] == ep && derives(t, x) && mark3[x] != p) {
                            mark3[x] = p; c--;
                        }
                    }
                    for (int t : inc[p]) for (int x : {ta[t], tb[t], tc[t]}) mark3[x] = -1;
                }
                if (c <= 0) U.push_back(p);
            }
            bool upOK = (cost() + 1 <= best + cap) && (rng.u() < pUp);
            if (U.empty() && !upOK) continue;

            zeroed.clear();
            int s = addLight(w);
            int u = -1; sup[s] = -1;
            {
                int nd = 0;
                for (int t : inc[s]) if (derives(t, s) && rng(++nd) == 0) sup[s] = t;
                if (sup[s] < 0) { delLight(s); zeroed.clear(); continue; }
                double pm = -1, um = 1e18;
                for (int y : {ta[sup[s]], tb[sup[s]], tc[sup[s]]}) if (y != s) pm = max(pm, posv[y]);
                for (int t : inc[s]) for (int y : {ta[t], tb[t], tc[t]}) if (y != s && posv[y] > pm) um = min(um, posv[y]);
                posv[s] = (um < 1e17) ? (rng(2) ? um - 1e-3 * rng.u() : pm + (um - pm) * rng.u()) : pm + 1e-3 * rng.u();
            }
            if (!U.empty()) {
                for (size_t i = U.size(); i > 1; i--) swap(U[i - 1], U[rng(i)]);
                for (int x : U) if (crit[x] == 0 && tryRemove(x)) { u = x; break; }
            }
            if (u < 0 && !upOK) { delLight(s); zeroed.clear(); continue; }
            pairs<true>(w);
            if (u >= 0) {
                stSwap++; commitRemove(u); accSince++;
                int r = pruneZeroed(s);
                if (accSince % 64 == 0) r += pruneAll(s);
                if (r) {
                    onState(28, 180000);
                } else if (cost() < best && (cap == 0 ? (stSwap & 3) == 0 : (stSwap & 15) == 0)) {
                    onState(8, 60000);
                }
            } else {
                stUp++; zeroed.clear();
            }
        }
    }
};

static vector<u64> progToValueSet(const Puzzle& p, const vector<array<int, 4>>& prog) {
    vector<u64> r(p.regs, 0), S;
    set<u64> seen(p.start.begin(), p.start.end());
    for (size_t i = 0; i < p.start.size(); i++) r[i] = p.start[i];
    for (auto& ins : prog) {
        u64 a = r[ins[1]], b = r[ins[2]];
        u64 v = ins[0] == '+' ? a + b : ins[0] == '-' ? a - b : ins[0] == '*' ? a * b : (b ? a / b : 0);
        r[ins[3]] = v;
        if (v && seen.insert(v).second) S.push_back(v);
    }
    return S;
}

int main(int argc, char** argv) {
    double secs = 20.0, T_init = 0.25, rbFrac = 0.55, pairFrac = 0.25;
    int cap = 3, verbose = 0, crossEvery = 80000;
    u64 seed = 20261128;
    string varFile, seedSetFile, dumpMinVFile;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "--time") secs = atof(argv[++i]);
        else if (a == "--seed") seed = strtoull(argv[++i], nullptr, 10);
        else if (a == "--T") T_init = atof(argv[++i]);
        else if (a == "--cap") cap = atoi(argv[++i]);
        else if (a == "--rb") rbFrac = atof(argv[++i]);
        else if (a == "--pair") pairFrac = atof(argv[++i]);
        else if (a == "--cross") crossEvery = atoi(argv[++i]);
        else if (a == "--variants") varFile = argv[++i];
        else if (a == "--seed-sets") seedSetFile = argv[++i];
        else if (a == "--dump-minv") dumpMinVFile = argv[++i];
        else if (a == "--verbose") verbose = 1;
    }

    unordered_map<string, vector<vector<u64>>> varSets;
    if (!varFile.empty()) {
        ifstream vf(varFile);
        Puzzle vp;
        while (readPuzzle(vf, vp)) {
            if (verifyProgExact(vp, vp.prog)) varSets[vp.id].push_back(progToValueSet(vp, vp.prog));
        }
    }
    unordered_map<string, vector<vector<u64>>> seedSets;
    if (!seedSetFile.empty()) {
        ifstream sf(seedSetFile);
        string tag;
        while (sf >> tag) {
            if (tag != "V") { string dummy; getline(sf, dummy); continue; }
            string pid; int origO, newO, dlt, chg;
            sf >> pid >> origO >> newO >> dlt >> chg;
            vector<u64> vs(newO);
            for (int i = 0; i < newO; i++) sf >> vs[i];
            seedSets[pid].push_back(std::move(vs));
        }
    }

    ios::sync_with_stdio(false);
    Puzzle p;
    u64 idx = 0;
    while (readPuzzle(cin, p)) {
        if (!verifyProgExact(p, p.prog)) { cerr << p.id << " invalid input\n"; continue; }
        double t0 = now_sec();
        u64 pseed = (seed * 0x9e3779b97f4a7c15ULL) ^ (hash<string>{}(p.id) + idx++);
        auto E = make_unique<SuperWalk>(p, pseed);
        vector<u64> S = progToValueSet(p, p.prog);
        if ((int)S.size() + (int)p.start.size() >= SuperWalk::MV - 8) { cerr << p.id << " too big\n"; continue; }
        E->best = (int)p.prog.size();
        E->bestProg = p.prog;
        E->bestSet = S;
        if (varSets.count(p.id)) E->variants = varSets[p.id];
        if (seedSets.count(p.id) && !seedSets[p.id].empty()) {
            const auto& ss = seedSets[p.id];
            E->variants.push_back(S);
            for (const auto& sv : ss) {
                if ((int)sv.size() + (int)p.start.size() >= SuperWalk::MV - 8) continue;
                E->loadSet(sv);
                if (!E->closureOK(-1)) continue;
                E->pruneAll(-1);
                E->onState(48, 300000);
            }
            double per = secs / (double)ss.size();
            for (const auto& sv : ss) {
                if ((int)sv.size() + (int)p.start.size() >= SuperWalk::MV - 8) continue;
                E->loadSet(sv);
                if (!E->closureOK(-1)) continue;
                E->pruneAll(-1);
                E->onState(32, 200000);
                if (E->cost() >= E->best) E->seedBasin = E->curSet();
                else E->seedBasin.clear();
                E->run(per, T_init, max(cap, max(0, E->cost() - E->best) + 2), rbFrac, pairFrac, crossEvery);
            }
        } else {
            E->loadSet(S);
            if (!E->closureOK(-1)) { cerr << p.id << " closure fail\n"; continue; }
            E->pruneAll(-1);
            E->onState(16, 120000);
            E->run(secs, T_init, cap, rbFrac, pairFrac, crossEvery);
        }
        double dt = now_sec() - t0;
        if (verbose) {
            cerr << p.id << " " << p.prog.size() << "->" << E->best
                 << " (minV=" << E->minCostSeen << ")"
                 << " | dt=" << fixed << setprecision(1) << dt << "s"
                 << " prop=" << E->stProp << " swap=" << E->stSwap
                 << " rb=" << E->stRb << "/" << E->stRbHit << "/" << E->stRbGain
                 << " fac=" << E->stFac << "/" << E->stFacHit
                 << " pair=" << E->stPair << "/" << E->stPairHit << "/" << E->stPairGain
                 << " tpair=" << E->stTargPair << "/" << E->stTargPairHit << "/" << E->stTargPairGain
                 << " cross=" << E->stCross
                 << " sched=" << E->stSched << "(fail=" << E->stSchedFail << ",remat=" << E->stRemat << ")\n";
        }
        if (!dumpMinVFile.empty() && E->minCostSeen < (int)p.prog.size() && !E->minCostSet.empty()) {
            ofstream dmf(dumpMinVFile, ios::app);
            dmf << "V " << p.id << " " << p.prog.size() << " " << E->minCostSet.size() << " "
                << ((int)E->minCostSet.size() - (int)p.prog.size()) << " " << E->minCostSet.size();
            for (u64 v : E->minCostSet) dmf << " " << v;
            dmf << "\n";
        }
        if (E->best < (int)p.prog.size() && verifyProgExact(p, E->bestProg)) {
            cout << "R " << p.id << " " << p.prog.size() << " " << E->bestProg.size() << "\n";
            for (auto& x : E->bestProg) cout << (char)x[0] << " " << x[1] << " " << x[2] << " " << x[3] << "\n";
            cout.flush();
        }
    }
    return 0;
}
