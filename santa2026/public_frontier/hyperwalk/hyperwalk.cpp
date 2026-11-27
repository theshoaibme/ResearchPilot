// HYPERWALK: Next-Generation Value-Set Hypergraph Engine for Santa 2026.
// Built from the ground up to smash all 4 bottlenecks of triwalk / setwalk / elfopt:
//   1. Min-Pressure Greedy + Multi-Derivation + Rematerialization-Rescue 48-register Scheduler
//   2. Unshackled 60-bit Deep Rebuild + Exact Brent Pollard-Rho Factor-Offset Rebuild
//   3. 2-Sink Joint Hub-Knockout & Bridge-Intersection Rebuild (breaks single-cone deadlocks!)
//   4. High-Slack Cooling + Multi-Bank Value-Set Crossover + Built-in Exact Referee Verifier
//
// Input/Output format: pfmt (P id nStart starts... nTarget targets... L regs nProg / op i j d)
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

// Fast 64-bit Miller-Rabin + Brent Pollard-Rho for Pillar 2 Factor-Offset Rebuild
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
    return n; // bail out cleanly if stubborn semiprime
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
    vector<array<int, 4>> prog; // [op_char, i, j, d]
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

// Fast open-addressing map u64 -> int (empty key = 0)
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

// Candidate pair-result table: tracks values reachable in 1 op from V with count >= 2
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

struct HyperWalk {
    static constexpr int MV = 512;
    static constexpr int MT = 16384;

    const Puzzle& P;
    u64 L;
    Rng rng;

    // Value slots
    u64 val[MV];
    bool alive[MV], isS[MV], isT[MV];
    int al[MV], posAl[MV], nal = 0;
    u64 av[MV];
    int nStart = 0, one = -1;
    vector<int> freeS, startSlots;
    U64Map vmap;

    // Triples (a + b = c when tt==0; a * b = c when tt==1; a <= b)
    int16_t ta[MT], tb[MT], tc[MT], tx[MT];
    uint8_t tt[MT];
    int tHigh = 0, nTri = 0;
    vector<int> freeT;
    vector<int> inc[MV];
    int deg[MV], crit[MV], sup[MV];
    vector<int> zeroed;

    PairTable pt;

    // Stats
    i64 stProp = 0, stSwap = 0, stUp = 0, stRb = 0, stRbHit = 0, stRbGain = 0;
    i64 stFac = 0, stFacHit = 0, stPair = 0, stPairHit = 0, stPairGain = 0;
    i64 stCross = 0, stSched = 0, stSchedFail = 0, stRemat = 0, stDown = 0;

    // Best tracking
    int best = 0;
    vector<u64> bestSet, feasSet;
    vector<array<int, 4>> bestProg;
    vector<vector<u64>> variants; // alternative value sets for crossover

    double posv[MV];

    HyperWalk(const Puzzle& p, u64 seed) : P(p), L(p.L) {
        rng.s = seed ? seed : 1;
        for (int i = 0; i < 8; i++) rng.next();
        vmap.init(11); // 2048 slots for <= 512 values
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

    // Rel enumeration for new value w against currently alive values
    struct Rel { int y, z; uint8_t k; }; // k: 0:y+z=w, 1:y*z=w, 2:y+w=z, 3:y*w=z, 4:w+w=z, 5:w*w=z
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

    // ==================== PILLAR 1: MIN-PRESSURE + REMATERIALIZATION SCHEDULER ====================
    struct Sch { vector<int> seq; int da[MV], db[MV]; uint8_t op[MV]; } sc;
    int idxOf[MV], lastU[MV];
    vector<int> die_;
    static inline int opOf(int t2, int kind) { return t2 == 0 ? (kind == 0 ? '+' : '-') : (kind == 0 ? '*' : '/'); }

    // Multi-mode initSchedule:
    //   mode 0 = shortest-span posv guided
    //   mode 1 = min-pressure greedy (targets/low-deg first)
    //   mode 2 = randomized posv jitter
    //   mode 3 = true Sethi-Ullman / min-live-delta greedy pebbling!
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

        // For mode 3/4: pick a valid derivation DAG (mode 0 on first try, randomized mode 2 on later tries),
        // then re-order greedily by exact Sethi-Ullman min-live-delta pebbling!
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
                    // Also reward nodes whose consumers are almost ready (indeg==1)!
                    int unlocks = 0;
                    for (int ch : adj[x]) if (indeg[ch] == 1) unlocks++;
                    double scv = (freed - creates) * 100.0 + (remU[x] == 0 ? 60.0 : -remU[x] * 3.0)
                               + unlocks * 4.0 - posv[x] * 0.05 + rng.u() * 12.0;
                    if (scv > bestSc) { bestSc = scv; pick = i; }
                }
                int x = rdy[pick];
                rdy[pick] = rdy.back(); rdy.pop_back();
                newSeq.push_back(x);
                int a = sc.da[x], b = sc.db[x];
                remU[a]--; if (b != a) remU[b]--;
                for (int ch : adj[x]) if (--indeg[ch] == 0) rdy.push_back(ch);
            }
            if ((int)newSeq.size() == n) { sc.seq = std::move(newSeq); return true; }
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
                    double scv = posv[x] * 0.4 + (isT[x] ? -12.0 : 0.0) + (double)inc[x].size() * 0.6 + rng.u() * 4.0;
                    if (scv < bestSc) { bestSc = scv; pick = i; }
                }
            } else {
                double bestP = 1e18;
                for (int i = 0; i < (int)ready.size(); i++) {
                    double pv = posv[ready[i]] + rng.u() * 18.0;
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

    int overflow(int limit, int* maxPeak = nullptr) {
        int n = (int)sc.seq.size();
        if ((int)die_.size() < n + 1) die_.assign(n + 1, 0);
        memset(lastU, 255, sizeof lastU);
        for (int i = 0; i < n; i++) { int x = sc.seq[i]; lastU[sc.da[x]] = i; lastU[sc.db[x]] = i; }
        std::fill(die_.begin(), die_.begin() + n + 1, 0);
        int cur = 0, ov = 0, pk = 0;
        for (int s : startSlots) if (lastU[s] >= 0) { cur++; die_[lastU[s]]++; }
        for (int i = 0; i < n; i++) {
            int x = sc.seq[i];
            int need = cur - die_[i] + 1;
            if (need > pk) pk = need;
            if (need > limit) ov += (need - limit);
            cur -= die_[i];
            if (lastU[x] > i) { cur++; die_[lastU[x]]++; }
        }
        if (maxPeak) *maxPeak = pk;
        // Strongly penalize peak overflow as well as total overflow area!
        return ov == 0 ? 0 : (ov * 2 + (pk - limit) * 16);
    }

    bool annealOrder(int iters) {
        int n = (int)sc.seq.size();
        int cur = overflow(P.regs);
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
        double Ts = 2.0;
        for (int it = 0; it < iters; it++) {
            double T = Ts * (1.0 - (double)it / iters) + 0.03;
            int i = rng(n), x = sc.seq[i];
            if (rng(6) == 0 && x != one) {
                auto& I = inc[x]; if (I.size() < 2) continue;
                int t = I[rng(I.size())]; int a = ta[t], b = tb[t], c = tc[t], p, q, op;
                if (a == b) { if (c != x) continue; p = q = a; op = opOf(tt[t], 0); }
                else if (c == x) { p = a; q = b; op = opOf(tt[t], 0); }
                else { p = c; q = (a == x) ? b : a; op = opOf(tt[t], 1); }
                if (idxOf[p] >= i || idxOf[q] >= i || (p == sc.da[x] && q == sc.db[x])) continue;
                int oa = sc.da[x], ob = sc.db[x], oo = sc.op[x];
                sc.da[x] = p; sc.db[x] = q; sc.op[x] = op;
                int c2 = overflow(P.regs);
                if (c2 <= cur || rng.u() < exp((cur - c2) / T)) {
                    cur = c2; rmUser(oa, x); if (ob != oa) rmUser(ob, x);
                    users[p].push_back(x); if (q != p) users[q].push_back(x);
                    if (!cur) return true;
                } else { sc.da[x] = oa; sc.db[x] = ob; sc.op[x] = oo; }
                continue;
            }
            int lo = max(idxOf[sc.da[x]], idxOf[sc.db[x]]) + 1, hi = n - 1;
            for (int u : users[x]) hi = min(hi, idxOf[u] - 1);
            if (hi <= lo) continue;
            int j = lo + rng(hi - lo + 1);
            if (j == i) continue;
            auto mv = [&](int f, int t) {
                if (f < t) rotate(sc.seq.begin() + f, sc.seq.begin() + f + 1, sc.seq.begin() + t + 1);
                else rotate(sc.seq.begin() + t, sc.seq.begin() + f, sc.seq.begin() + f + 1);
            };
            mv(i, j);
            int c2 = overflow(P.regs);
            if (c2 <= cur || rng.u() < exp((cur - c2) / T)) {
                cur = c2;
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

    // Rematerialization Rescue: if schedule overflows by 1-2 regs AND cost() + remat < best,
    // recompute a cheap value whose parents are live instead of holding it across the peak!
    bool emitWithRemat(vector<array<int, 4>>& out, int maxExtra) {
        if (maxExtra <= 0) return false;
        int n = (int)sc.seq.size();
        // Track remaining future uses for every value in sc.seq
        vector<vector<int>> usePos(MV);
        for (int i = 0; i < n; i++) {
            int x = sc.seq[i];
            usePos[sc.da[x]].push_back(i);
            if (sc.db[x] != sc.da[x]) usePos[sc.db[x]].push_back(i);
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
        auto findLiveDer = [&](int x, int& r1, int& r2, int& op) -> bool {
            if (x == one) {
                for (int r = 0; r < P.regs; r++) if (occ[r] >= 0) { r1 = r2 = r; op = '/'; return true; }
            }
            for (int t : inc[x]) {
                int a = ta[t], b = tb[t], c = tc[t], p = -1, q = -1, o = 0;
                if (a == b) { if (c != x) continue; p = q = a; o = opOf(tt[t], 0); }
                else if (c == x) { p = a; q = b; o = opOf(tt[t], 0); }
                else { int other = (a == x) ? b : a; p = c; q = other; o = opOf(tt[t], 1); }
                if (reg[p] >= 0 && reg[q] >= 0) { r1 = reg[p]; r2 = reg[q]; op = o; return true; }
            }
            return false;
        };

        out.clear();
        int extra = 0;
        for (int i = 0; i < n; i++) {
            int x = sc.seq[i], a = sc.da[x], b = sc.db[x];
            // Ensure a and b are in registers (rematerialize if evicted earlier!)
            for (int need : {a, b}) {
                if (reg[need] >= 0) continue;
                if (++extra > maxExtra) return false;
                int p1, p2, op;
                if (!findLiveDer(need, p1, p2, op)) return false;
                // Find free reg (or evict farthest-next-use rematerializable reg!)
                int d = -1;
                for (int r = 0; r < P.regs; r++) if (occ[r] < 0) { d = r; break; }
                if (d < 0) {
                    int bestR = -1, far = -1;
                    for (int r = 0; r < P.regs; r++) {
                        int v = occ[r];
                        if (v == a || v == b || v == occ[p1] || v == occ[p2]) continue;
                        int nu = nextUse(v, i);
                        if (nu < 0) { bestR = r; break; }
                        if (nu > far) { far = nu; bestR = r; }
                    }
                    if (bestR < 0) return false;
                    reg[occ[bestR]] = -1; occ[bestR] = -1; d = bestR;
                }
                out.push_back({op, p1, p2, d});
                reg[need] = d; occ[d] = need;
            }
            int r1 = reg[a], r2 = reg[b];
            if (nextUse(a, i) < 0) { occ[r1] = -1; reg[a] = -1; }
            if (b != a && nextUse(b, i) < 0) { occ[r2] = -1; reg[b] = -1; }
            int d = -1;
            for (int r = 0; r < P.regs; r++) if (occ[r] < 0) { d = r; break; }
            if (d < 0) {
                // Evict the value with farthest next use (and prefer one that IS rematerializable!)
                int bestR = -1, bestScore = -1;
                for (int r = 0; r < P.regs; r++) {
                    int v = occ[r];
                    int nu = nextUse(v, i);
                    if (nu < 0) { bestR = r; break; }
                    int p1, p2, op;
                    int remat_bonus = findLiveDer(v, p1, p2, op) ? 100000 : 0;
                    int scv = remat_bonus + nu;
                    if (scv > bestScore) { bestScore = scv; bestR = r; }
                }
                if (bestR < 0) return false;
                reg[occ[bestR]] = -1; occ[bestR] = -1; d = bestR;
            }
            out.push_back({sc.op[x], r1, r2, d});
            if (nextUse(x, i) >= 0) { reg[x] = d; occ[d] = x; }
        }
        return verifyProgExact(P, out);
    }

    bool schedule(vector<array<int, 4>>& out, int tries, int iters, bool allowRemat) {
        stSched++;
        int bestOv = 1e9;
        Sch bestSc;
        for (int t = 0; t < tries; t++) {
            int mode = (t == 0) ? 3 : (t == 1) ? 0 : (t == 2) ? 1 : (t % 2 == 1 ? 4 : 2);
            if (!initSchedule(mode)) continue;
            int ov = overflow(P.regs);
            if (ov == 0 || annealOrder(iters)) {
                if (emitSchedule(out) && verifyProgExact(P, out)) return true;
            }
            ov = overflow(P.regs);
            if (ov < bestOv) { bestOv = ov; bestSc = sc; }
        }
        // Rematerialization rescue if strictly better than `best` even with +1..+2 extra ops!
        if (allowRemat && bestOv < 1e9 && cost() + 1 < best) {
            sc = bestSc;
            int maxExtra = min(2, best - cost() - 1);
            if (emitWithRemat(out, maxExtra) && (int)out.size() < best) {
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
        nStart = 0; startSlots.clear(); one = -1;
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
        for (u64 t : P.targets) { int s = look(t); assert(s >= 0); isT[s] = true; }
        rebuildPairs(); zeroed.clear(); memset(lost, 0, sizeof lost); memset(gone, 0, sizeof gone); buildSupport();
    }

    // ==================== PILLAR 2 & 3: UNSHACKLED 60-BIT + POLLARD-RHO + 2-SINK BRIDGE REBUILD ====================
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
        // Clear blocked flags in vis so vis[] strictly means "in K"
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
    double deepP = 0.65, facP = 0.70;

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
        // Two passes in case Wn order needs dependency resolution
        for (int pass = 0; pass < 3; pass++) {
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

    // Pillar 2: Single-cone unshackled 60-bit + Pollard-Rho Factor-Offset Rebuild
    bool rebuildMove() {
        stRb++;
        int x; do x = al[rng(nal)]; while (fre(x));
        vector<int> Rm; vector<pair<int, int>> undo; vector<int> qu;
        freeX = x;
        auto pushT = [&](int t) { if (t < 0) return; for (int y : {ta[t], tb[t], tc[t]}) if (y != x) qu.push_back(y); };
        pushT(sup[x]);
        while (!qu.empty() && Rm.size() < 8) {
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

            auto offerVec = [&](const vector<u64>& seq, bool checkLast) {
                vector<u64> nv;
                for (size_t i = 0; i < seq.size(); i++) {
                    u64 v = seq[i];
                    if (v == 0 || v > L || v == X) return;
                    if (inK(v)) continue;
                    int z = look(v);
                    if (z >= 0 && !gone[z]) return;
                    if (find(nv.begin(), nv.end(), v) != nv.end()) continue;
                    if ((checkLast || i + 1 < seq.size()) && !inS1K(v)) return;
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

            // 1-new and 2-new exact
            for (int i = 0; i < nk; i++) inv(X, kv[i], L, [&](u64 w) { if (bit(w)) offerVec({w}, true); });
            if (bestN > 1 && lim >= 2) {
                int nh = 0, off = rng(nk);
                for (int ii = 0; ii < nk && nh < 400; ii++) {
                    int i = ii + off; if (i >= nk) i -= nk;
                    inv(X, kv[i], L, [&](u64 w2) {
                        if (w2 == X || look(w2) >= 0) return;
                        for (int j = 0; j < nk && nh < 400; j++) {
                            inv(w2, kv[j], L, [&](u64 w1) {
                                if (bit(w1)) { nh++; offerVec({w1, w2}, false); }
                            });
                        }
                    });
                }
                size_t ns1 = s1.size(), o2 = ns1 ? rng(ns1) : 0;
                for (size_t q = 0; q < ns1 && nh < 600; q++) {
                    u64 w1 = s1[q + o2 < ns1 ? q + o2 : q + o2 - ns1];
                    inv(X, w1, L, [&](u64 w2) { if (bit(w2) && w2 != w1) { nh++; offerVec({w1, w2}, true); } });
                }
            }

            // Deep 3-4 new (UNSHACKLED 60-bit!)
            if (Wn.empty() && lim >= 3 && f >= 2 && rng.u() < deepP) {
                sorted_ = s1;
                for (int i = 0; i < nk; i++) sorted_.push_back(kv[i]);
                sort(sorted_.begin(), sorted_.end());
                sorted_.erase(unique(sorted_.begin(), sorted_.end()), sorted_.end());
                int checks = 0;
                size_t nb = sorted_.size(), st = nb > 6000 ? nb / 6000 : 1, o = rng(st);
                auto tryB = [&](u64 B) {
                    if (B < 2 || B > X) return;
                    u64 q = X / B;
                    size_t c = lower_bound(sorted_.begin(), sorted_.end(), q) - sorted_.begin();
                    for (int d = -3; d <= 3; d++) {
                        i64 ix = (i64)c + d;
                        if (ix < 0 || ix >= (i64)nb) continue;
                        u64 A = sorted_[ix];
                        if (A < 2 || (u128)A * B > (u128)L) continue;
                        u64 Pp = A * B;
                        if (Pp == X || Pp == 0) continue;
                        u64 R = Pp > X ? Pp - X : X - Pp;
                        if (R != X && (inK(R) || bit(R)) && checks < 20000) {
                            checks++;
                            offerVec({A, B, R, Pp}, false);
                        }
                    }
                };
                for (int i = 0; i < nk; i++) tryB(kv[i]);
                for (size_t i = o; i < nb; i += st) tryB(sorted_[i]);

                // Unshackled 60-bit (w1 op w2) op a!
                for (int rep = 0; rep < 6 && checks < 25000; rep++) {
                    u64 a = kv[rng(nk)];
                    inv(X, a, L, [&](u64 y) {
                        if (look(y) >= 0) return;
                        if (rep < 3) {
                            for (u64 w1 : s1) inv(y, w1, L, [&](u64 w2) {
                                if (w2 != w1 && bit(w2) && checks < 25000) { checks++; offerVec({w1, w2, y}, false); }
                            });
                        } else {
                            for (int r2 = 0; r2 < 10 && checks < 25000; r2++) {
                                u64 b = kv[rng(nk)];
                                inv(y, b, L, [&](u64 z) {
                                    if (look(z) >= 0) return;
                                    for (int j = 0; j < nk && checks < 25000; j++) {
                                        inv(z, kv[j], L, [&](u64 w1) {
                                            if (bit(w1)) {
                                                checks++;
                                                if (inS1K(w1)) {
                                                    vector<u64> nv = {w1, z, y};
                                                    // Check all 3 valid/new
                                                    bool bad = false;
                                                    for (u64 v : nv) if (v == X || (look(v) >= 0 && !gone[look(v)])) bad = true;
                                                    if (!bad && 3 <= lim && 3 <= bestN) {
                                                        if (3 < bestN) { bestN = 3; ties = 0; }
                                                        if (rng(++ties) == 0) Wn = nv;
                                                    }
                                                }
                                            }
                                        });
                                    }
                                });
                            }
                        }
                    });
                }

                // PILLAR 2B: Exact Brent Pollard-Rho Factor-Offset Rebuild for 24..60-bit X when f >= 3!
                if (Wn.empty() && f >= 3 && X >= (1ULL << 22) && rng.u() < facP) {
                    stFac++;
                    vector<pair<u64, int>> pf;
                    vector<u64> divs;
                    // Test r = 0 plus 16 sampled offsets r in K!
                    for (int tr = 0; tr < 17 && Wn.empty(); tr++) {
                        u64 r = (tr == 0) ? 0 : kv[rng(nk)];
                        u64 cands[2] = {0, 0};
                        int nc = 0;
                        if (r == 0) cands[nc++] = X;
                        else {
                            if (X > r) cands[nc++] = X - r;
                            if (X <= L - r) cands[nc++] = X + r;
                        }
                        for (int ci = 0; ci < nc && Wn.empty(); ci++) {
                            u64 N = cands[ci];
                            factor_fast(N, rng, pf);
                            // Generate up to 128 balanced divisors of N
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
                                bool a_ok = inK(A) || inS1K(A);
                                bool b_ok = inK(B) || inS1K(B);
                                if (a_ok && b_ok) {
                                    if (r == 0) offerVec({A, B}, true);
                                    else offerVec({A, B, N}, false);
                                    if (!Wn.empty()) { stFacHit++; break; }
                                } else if (a_ok && lim >= 3) {
                                    // Check if B is 1-step from K (B = w1 op k -> 2 new: w1, B, plus A if new, plus N if r!=0)
                                    for (int j = 0; j < nk && Wn.empty(); j++) {
                                        inv(B, kv[j], L, [&](u64 w1) {
                                            if (!Wn.empty()) return;
                                            if (inS1K(w1)) {
                                                vector<u64> seq;
                                                if (!inK(A)) seq.push_back(A);
                                                seq.push_back(w1); seq.push_back(B);
                                                if (r > 0) seq.push_back(N);
                                                if ((int)seq.size() <= lim) {
                                                    bool bad = false;
                                                    for (u64 v : seq) if (v == X || (look(v) >= 0 && !gone[look(v)])) bad = true;
                                                    if (!bad && (int)seq.size() <= bestN) {
                                                        bestN = (int)seq.size(); Wn = seq; stFacHit++;
                                                    }
                                                }
                                            }
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Restore temporary removal state
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

    // ==================== PILLAR 3: 2-SINK JOINT HUB-KNOCKOUT & BRIDGE INTERSECTION ====================
    bool pairHubRebuild() {
        stPair++;
        // Pick two distinct non-free values x1, x2 that share a common ancestor in their support (or share a hub!)
        int h = al[rng(nal)];
        if (!removable(h)) return false;
        // Find downstream values supported through h
        vector<int> sinks;
        for (int t : inc[h]) {
            for (int y : {ta[t], tb[t], tc[t]}) {
                if (y != h && !fre(y) && sup[y] == t && find(sinks.begin(), sinks.end(), y) == sinks.end()) {
                    sinks.push_back(y);
                }
            }
        }
        if (sinks.size() < 2 || sinks.size() > 3) return false;
        // Try to free h AND any exclusive ancestors of h & sinks when sinks are temporarily free!
        vector<int> Rm; vector<pair<int, int>> undo;
        for (int y : sinks) freeMask[y] = true;
        vector<int> qu = {h};
        for (int y : sinks) {
            int t = sup[y];
            if (t >= 0) for (int z : {ta[t], tb[t], tc[t]}) if (z != y && !freeMask[z]) qu.push_back(z);
        }
        while (!qu.empty() && Rm.size() < 10) {
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

            // Compute I1 = { w in S1(K) : w derives Y1 with some k in K }
            // and     I2 = { w in S1(K) : w derives Y2 with some k in K }
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

            // 1-node joint bridge: w in I1 ∩ I2 (replaces f >= 2 with 1 node!!)
            if (!I1.empty() && !I2.empty()) {
                for (u64 w : I2) {
                    if (binary_search(I1.begin(), I1.end(), w) && look(w) < 0) {
                        Wn = {w}; break;
                    }
                }
            }
            // 2-node independent replacement: w1 in I1 + w2 in I2 (replaces f >= 3 with 2 nodes -> instant -1..-5!!)
            if (Wn.empty() && (f >= 3 || (f == 2 && up1)) && !I1.empty() && !I2.empty()) {
                // Prefer genuinely new pair
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
            // 2-node coupled bridge: w1 in I1 AND w2 = inv(Y2, w1) in S1(K)!
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
            // 2-node direct Y1<->Y2 chain: w1 in I1 AND Y2 = w2 op Y1!
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
            // Symmetric: w2 in I2 AND Y1 = w1 op Y2!
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

        // Restore
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

    void onState() {
        int c = cost();
        if (c < best) {
            vector<array<int, 4>> out;
            // Deep 18-try x 120k-iter scheduler on every rare cost < best discovery!
            if (schedule(out, 18, 120000, true) && (int)out.size() < best) {
                best = (int)out.size();
                bestProg = out;
                bestSet = orderedSet();
                feasSet = bestSet;
                stDown++;
            }
        }
    }

    // Main loop
    int mark[MV], mark2[MV], mark3[MV], ep = 0;
    vector<int> lst, U;

    void run(double secs, double T_init, int cap_, double rbFrac, double pairFrac, int crossEvery) {
        cap = cap_;
        double t0 = now_sec(), tReset = t0;
        memset(mark, 0, sizeof mark); memset(mark2, 0, sizeof mark2); memset(mark3, 255, sizeof mark3);
        feasSet = bestSet;
        i64 accSince = 0;
        double rbTime = 0, pairTime = 0;

        for (i64 it = 0;; it++) {
            if ((it & 63) == 0) {
                double t = now_sec();
                if (t - t0 > secs) break;
                if (t - tReset > max(8.0, secs * 0.25)) {
                    tReset = t;
                    // Either crossover with a variant or restart from bestSet!
                    if (!variants.empty() && rng(2) == 0) {
                        stCross++;
                        vector<u64> u = bestSet;
                        const auto& vv = variants[rng(variants.size())];
                        u.insert(u.end(), vv.begin(), vv.end());
                        if ((int)u.size() + nStart < MV - 8) {
                            loadSet(u);
                            pruneAll(-1);
                            onState();
                        } else loadSet(bestSet);
                    } else {
                        loadSet(bestSet);
                    }
                }
            }
            if (pt.cand.empty()) { loadSet(bestSet); if (pt.cand.empty()) break; }
            if (pt.used > pt.mask * 0.6) rebuildPairs();

            double elapsed = now_sec() - t0;
            double frac = min(1.0, elapsed / max(0.1, secs));
            double curT = T_init * pow(0.15, frac);
            double pUp = exp(-1.0 / max(0.05, curT));

            // Pillar 3: 2-Sink Joint Hub-Knockout & Bridge Intersection
            if (pairFrac > 0 && (it & 31) == 0 && pairTime < pairFrac * elapsed) {
                double tp0 = now_sec();
                pairHubRebuild();
                pairTime += now_sec() - tp0;
                if (pt.used > pt.mask * 0.6) rebuildPairs();
            }

            // Pillar 2: Unshackled 60-bit + Pollard-Rho Cone Rebuild
            if (rbFrac > 0 && (it & 15) == 0 && rbTime < rbFrac * elapsed) {
                double tb0 = now_sec();
                rebuildMove();
                rbTime += now_sec() - tb0;
                if (pt.used > pt.mask * 0.6) rebuildPairs();
            }

            // Pillar 4: Periodic Value-Set Crossover (only on rare intervals, never wiping active walks!)
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

            // Standard value-set swap / up-move
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
                if (r) onState();
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
    u64 seed = 20261126;
    string varFile, seedSetFile;
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
        else if (a == "--verbose") verbose = 1;
    }

    // Load optional variant pools (same pfmt blocks!)
    unordered_map<string, vector<vector<u64>>> varSets;
    if (!varFile.empty()) {
        ifstream vf(varFile);
        Puzzle vp;
        while (readPuzzle(vf, vp)) {
            if (verifyProgExact(vp, vp.prog)) varSets[vp.id].push_back(progToValueSet(vp, vp.prog));
        }
    }
    // Load optional MEGA-LNS seed value sets (V id orig_ops new_ops delta chg v1 v2 ...)
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
        auto E = make_unique<HyperWalk>(p, pseed);
        vector<u64> S = progToValueSet(p, p.prog);
        if ((int)S.size() + (int)p.start.size() >= HyperWalk::MV - 8) { cerr << p.id << " too big\n"; continue; }
        E->best = (int)p.prog.size();
        E->bestProg = p.prog;
        E->bestSet = S;
        if (varSets.count(p.id)) E->variants = varSets[p.id];
        if (seedSets.count(p.id) && !seedSets[p.id].empty()) {
            const auto& ss = seedSets[p.id];
            E->variants.push_back(S);
            // First pass: test direct scheduling of every seed set (instant win if delta < 0!)
            for (const auto& sv : ss) {
                if ((int)sv.size() + (int)p.start.size() >= HyperWalk::MV - 8) continue;
                E->loadSet(sv);
                if (!E->closureOK(-1)) continue;
                E->pruneAll(-1);
                E->onState();
            }
            // Second pass: spend time budget collapsing each brand-new skeleton basin!
            double per = secs / (double)ss.size();
            for (const auto& sv : ss) {
                if ((int)sv.size() + (int)p.start.size() >= HyperWalk::MV - 8) continue;
                E->loadSet(sv);
                if (!E->closureOK(-1)) continue;
                E->pruneAll(-1);
                E->onState();
                E->run(per, T_init, max(cap, max(0, E->cost() - E->best) + 2), rbFrac, pairFrac, crossEvery);
            }
        } else {
            E->loadSet(S);
            if (!E->closureOK(-1)) { cerr << p.id << " closure fail\n"; continue; }
            E->pruneAll(-1);
            E->onState();
            E->run(secs, T_init, cap, rbFrac, pairFrac, crossEvery);
        }
        double dt = now_sec() - t0;
        if (verbose) {
            cerr << p.id << " " << p.prog.size() << "->" << E->best
                 << " | dt=" << fixed << setprecision(1) << dt << "s"
                 << " prop=" << E->stProp << " swap=" << E->stSwap
                 << " rb=" << E->stRb << "/" << E->stRbHit << "/" << E->stRbGain
                 << " fac=" << E->stFac << "/" << E->stFacHit
                 << " pair=" << E->stPair << "/" << E->stPairHit << "/" << E->stPairGain
                 << " cross=" << E->stCross
                 << " sched=" << E->stSched << "(fail=" << E->stSchedFail << ",remat=" << E->stRemat << ")\n";
        }
        if (E->best < (int)p.prog.size() && verifyProgExact(p, E->bestProg)) {
            cout << "R " << p.id << " " << p.prog.size() << " " << E->bestProg.size() << "\n";
            for (auto& x : E->bestProg) cout << (char)x[0] << " " << x[1] << " " << x[2] << " " << x[3] << "\n";
            cout.flush();
        }
    }
    return 0;
}
