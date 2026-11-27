// Warm-started 10-thread Tabu / Parallel-Tempering Register Pressure Minimizer for scored_0774
// Starts directly from the exact 263-unique-value DAG of santa_best.csv (peak = 49, 1 reg over 48!)
// and searches coupled topological relocations + alternative derivations across all 266 values.

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;

using u64 = uint64_t;
using u128 = __uint128_t;

struct Rng {
    u64 s;
    explicit Rng(u64 seed) : s(seed ? seed : 1) { for (int i = 0; i < 8; i++) next(); }
    inline u64 next() {
        u64 z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    inline u64 operator()(u64 n) { return next() % n; }
    inline double u() { return (next() >> 11) * (1.0 / (1ULL << 53)); }
};

struct Puzzle {
    string id;
    vector<u64> start, targets;
    u64 L;
    int regs;
    vector<array<int, 4>> prog;
};

static bool verifyProgExact(const Puzzle& p, const vector<array<int, 4>>& prog) {
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

struct Der { int p, q, op; };

int main(int argc, char** argv) {
    double secs = argc > 1 ? atof(argv[1]) : 25.0;
    int nThreads = argc > 2 ? atoi(argv[2]) : 10;

    string tag;
    Puzzle P;
    cin >> tag >> P.id;
    int ns, nt, np;
    cin >> ns; P.start.resize(ns);
    for (int i = 0; i < ns; i++) cin >> P.start[i];
    cin >> nt; P.targets.resize(nt);
    for (int i = 0; i < nt; i++) cin >> P.targets[i];
    cin >> P.L >> P.regs >> np;
    P.prog.resize(np);
    for (int i = 0; i < np; i++) {
        string op; int a, b, d;
        cin >> op >> a >> b >> d;
        P.prog[i] = {(int)op[0], a, b, d};
    }

    // Extract exact warm-start 263-value order & derivations from P.prog!
    vector<u64> vals;
    unordered_map<u64, int> idOf;
    for (u64 s : P.start) {
        if (!idOf.count(s)) { idOf[s] = (int)vals.size(); vals.push_back(s); }
    }
    int nStart = (int)vals.size();
    vector<u64> regVal(P.regs, 0);
    for (int i = 0; i < (int)P.start.size(); i++) regVal[i] = P.start[i];

    vector<int> initSeq;
    vector<Der> initDer(512);
    for (auto& ins : P.prog) {
        u64 va = regVal[ins[1]], vb = regVal[ins[2]];
        u64 v = ins[0] == '+' ? va + vb : ins[0] == '-' ? va - vb : ins[0] == '*' ? va * vb : va / vb;
        regVal[ins[3]] = v;
        if (!idOf.count(v)) {
            int id = (int)vals.size();
            idOf[v] = id;
            vals.push_back(v);
            initSeq.push_back(id);
            initDer[id] = {idOf[va], idOf[vb], ins[0]};
        }
    }
    int V = (int)vals.size();
    int N = (int)initSeq.size();
    cerr << "[WarmRescue] V=" << V << " (starts=" << nStart << ", unique_ops=" << N << ", orig_prog=" << np << ")\n";

    // Precompute ALL valid 1-step derivations for every node x in [nStart .. V-1]
    vector<vector<Der>> allDers(V);
    for (int x = nStart; x < V; x++) {
        u64 vx = vals[x];
        if (vx == 1) {
            for (int s = 0; s < nStart; s++) allDers[x].push_back({s, s, '/'});
        }
        for (int a = 0; a < V; a++) {
            if (a == x) continue;
            u64 va = vals[a];
            for (int b = 0; b <= a; b++) {
                if (b == x) continue;
                u64 vb = vals[b];
                if (va <= P.L - vb && va + vb == vx) allDers[x].push_back({a, b, '+'});
                if (va > vb && va - vb == vx) allDers[x].push_back({a, b, '-'});
                if (vb > va && vb - va == vx) allDers[x].push_back({b, a, '-'});
                if (va > 1 && vb > 1 && (u128)va * vb == (u128)vx) allDers[x].push_back({a, b, '*'});
                if (vb > 0 && va % vb == 0 && va / vb == vx) allDers[x].push_back({a, b, '/'});
                if (va > 0 && vb != va && vb % va == 0 && vb / va == vx) allDers[x].push_back({b, a, '/'});
            }
        }
    }
    int multiDerCnt = 0;
    for (int x = nStart; x < V; x++) if (allDers[x].size() >= 2) multiDerCnt++;
    cerr << "[WarmRescue] Nodes with >=2 alternative derivations: " << multiDerCnt << "/" << N << "\n";

    auto evalSeq = [&](const vector<int>& seq, const vector<Der>& der, int& pk, int& ovSteps, int& firstOv, int& lastOv) {
        int lastU[512];
        fill(lastU, lastU + V, -1);
        for (int i = 0; i < N; i++) {
            int x = seq[i];
            lastU[der[x].p] = i;
            lastU[der[x].q] = i;
        }
        int die[512] = {};
        int cur = 0;
        pk = 0; ovSteps = 0; firstOv = -1; lastOv = -1;
        int excessSum = 0;
        for (int s = 0; s < nStart; s++) if (lastU[s] >= 0) { cur++; die[lastU[s]]++; }
        for (int i = 0; i < N; i++) {
            int x = seq[i];
            int need = cur - die[i] + 1;
            if (need > pk) pk = need;
            if (need > P.regs) {
                ovSteps++;
                excessSum += (need - P.regs);
                if (firstOv < 0) firstOv = i;
                lastOv = i;
            }
            cur -= die[i];
            if (lastU[x] > i) { cur++; die[lastU[x]]++; }
        }
        return excessSum == 0 ? 0 : (excessSum * 2 + (pk - P.regs) * 10);
    };

    {
        int pk, ovs, fo, lo;
        int sc0 = evalSeq(initSeq, initDer, pk, ovs, fo, lo);
        cerr << "[WarmRescue] Initial warm-start: score=" << sc0 << " peak=" << pk
             << " ovSteps=" << ovs << " range=[" << fo << ".." << lo << "]\n";
    }

    atomic<bool> solved{false};
    mutex mtx;
    vector<array<int, 4>> winProg;

    auto t0 = chrono::steady_clock::now();
    vector<thread> pool;
    for (int tid = 0; tid < nThreads; tid++) {
        pool.emplace_back([&, tid]() {
            Rng rng(2026112800ULL + tid * 13337ULL);
            vector<int> seq = initSeq;
            vector<Der> der = initDer;
            int idxOf[512];
            for (int s = 0; s < nStart; s++) idxOf[s] = -1;
            for (int i = 0; i < N; i++) idxOf[seq[i]] = i;

            vector<vector<int>> users(V);
            auto rebuildUsers = [&]() {
                for (int v = 0; v < V; v++) users[v].clear();
                for (int x : seq) {
                    users[der[x].p].push_back(x);
                    if (der[x].q != der[x].p) users[der[x].q].push_back(x);
                }
            };
            rebuildUsers();
            auto rmUser = [&](int p, int x) {
                auto& u = users[p];
                for (size_t i = 0; i < u.size(); i++) if (u[i] == x) { u[i] = u.back(); u.pop_back(); return; }
            };
            auto mv = [&](int f, int t) {
                if (f < t) rotate(seq.begin() + f, seq.begin() + f + 1, seq.begin() + t + 1);
                else if (t < f) rotate(seq.begin() + t, seq.begin() + f, seq.begin() + f + 1);
            };

            int pk, ovs, fo, lo_ov;
            int cur = evalSeq(seq, der, pk, ovs, fo, lo_ov);
            int bestLocal = cur;
            vector<int> bestSeq = seq;
            vector<Der> bestDer = der;

            long long iters = 0;
            for (;; iters++) {
                if ((iters & 4095) == 0) {
                    if (solved.load(memory_order_relaxed)) break;
                    double el = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
                    if (el > secs) break;
                    // Periodic restart to bestLocal if drifted too far
                    if (iters > 0 && (iters % 262144) == 0 && cur > bestLocal + 16) {
                        seq = bestSeq; der = bestDer;
                        for (int i = 0; i < N; i++) idxOf[seq[i]] = i;
                        rebuildUsers();
                        cur = evalSeq(seq, der, pk, ovs, fo, lo_ov);
                    }
                }

                // Multi-frequency temperature (each thread runs a distinct base temperature + cosine wave)
                double wave = 0.5 * (1.0 + cos((double)(iters % 120000) * (2.0 * M_PI / 120000.0)));
                double Tmin = 0.08 + 0.04 * (tid % 4);
                double Tmax = 2.2 + 0.6 * (tid % 5);
                double T = Tmin + (Tmax - Tmin) * wave;

                int roll = (int)rng(20);
                // 25%: Coupled Re-Derive + Relocate (including switching `1 = s/s` to a later live start or live value!)
                if (roll < 5) {
                    int i = rng(N), x = seq[i];
                    // Special case for value 1: 1 can ALSO be derived as v / v from ANY already-computed value v!
                    Der cand;
                    if (vals[x] == 1 && rng(2) == 0 && i > 0) {
                        int pick = seq[rng(i)];
                        cand = {pick, pick, '/'};
                    } else {
                        const auto& D = allDers[x];
                        if (D.size() < 2) continue;
                        cand = D[rng(D.size())];
                    }
                    if (cand.p == der[x].p && cand.q == der[x].q && cand.op == der[x].op) continue;
                    int hi = N - 1;
                    for (int u : users[x]) hi = min(hi, idxOf[u] - 1);
                    int maxP = max(idxOf[cand.p], idxOf[cand.q]);
                    if (maxP > hi) continue;
                    int j = i;
                    if (maxP >= i) {
                        int span = min(hi - maxP + 1, 16);
                        j = maxP + (span > 1 ? (int)rng(span) : 0);
                    } else if (rng(3) == 0 && hi > maxP + 1) {
                        int loB = maxP + 1;
                        j = loB + (int)rng(hi - loB + 1);
                    }
                    Der oldD = der[x];
                    der[x] = cand;
                    if (j != i) mv(i, j);
                    int pk2, ovs2, fo2, lo2;
                    int c2 = evalSeq(seq, der, pk2, ovs2, fo2, lo2);
                    if (c2 <= cur || rng.u() < exp((cur - c2) / T)) {
                        rmUser(oldD.p, x); if (oldD.q != oldD.p) rmUser(oldD.q, x);
                        users[cand.p].push_back(x); if (cand.q != cand.p) users[cand.q].push_back(x);
                        if (j != i) for (int k = min(i, j); k <= max(i, j); k++) idxOf[seq[k]] = k;
                        cur = c2; pk = pk2; ovs = ovs2; fo = fo2; lo_ov = lo2;
                        if (cur < bestLocal) { bestLocal = cur; bestSeq = seq; bestDer = der; }
                        if (cur == 0) break;
                    } else {
                        if (j != i) mv(j, i);
                        der[x] = oldD;
                    }
                    continue;
                }

                // 45%: Overflow-Window Targeted Move!
                int i = -1, dir = 0;
                if (roll < 14 && fo >= 0 && lo_ov >= fo) {
                    int targetPos = fo + (lo_ov > fo ? (int)rng(lo_ov - fo + 1) : 0);
                    int probe = (int)rng(targetPos + 1);
                    int v = seq[probe];
                    int lu = -1;
                    for (int u : users[v]) lu = max(lu, idxOf[u]);
                    if (lu >= targetPos) {
                        if (rng(2) == 0) { i = probe; dir = 1; }
                        else { i = lu; dir = -1; }
                    }
                }
                if (i < 0) i = rng(N);
                int x = seq[i];
                int lo = max(idxOf[der[x].p], idxOf[der[x].q]) + 1, hi = N - 1;
                for (int u : users[x]) hi = min(hi, idxOf[u] - 1);
                if (hi <= lo) continue;
                int j;
                if (dir == 1 && hi > i) {
                    j = (lo_ov >= i && hi > lo_ov && rng(2) == 0)
                        ? (lo_ov + 1 + (int)rng(hi - lo_ov))
                        : (i + 1 + (int)rng(hi - i));
                } else if (dir == -1 && lo < i) {
                    j = (fo <= i && lo <= fo && rng(2) == 0)
                        ? (lo + (int)rng(fo - lo + 1))
                        : (lo + (int)rng(i - lo));
                } else {
                    j = lo + (int)rng(hi - lo + 1);
                }
                if (j == i) continue;
                mv(i, j);
                int pk2, ovs2, fo2, lo2;
                int c2 = evalSeq(seq, der, pk2, ovs2, fo2, lo2);
                if (c2 <= cur || rng.u() < exp((cur - c2) / T)) {
                    for (int k = min(i, j); k <= max(i, j); k++) idxOf[seq[k]] = k;
                    cur = c2; pk = pk2; ovs = ovs2; fo = fo2; lo_ov = lo2;
                    if (cur < bestLocal) { bestLocal = cur; bestSeq = seq; bestDer = der; }
                    if (cur == 0) break;
                } else {
                    mv(j, i);
                }
            }

            if (cur == 0 && !solved.exchange(true)) {
                // Emit exact 48-register program!
                int lastU[512]; fill(lastU, lastU + V, -1);
                for (int i = 0; i < N; i++) { lastU[der[seq[i]].p] = i; lastU[der[seq[i]].q] = i; }
                vector<int> reg(V, -1), occ(P.regs, -1);
                for (int k = 0; k < (int)P.start.size(); k++) {
                    int s = idOf[P.start[k]];
                    if (reg[s] < 0 && lastU[s] >= 0) { reg[s] = k; occ[k] = s; }
                }
                vector<array<int, 4>> out;
                bool ok = true;
                for (int i = 0; i < N; i++) {
                    int x = seq[i], a = der[x].p, b = der[x].q;
                    int r1 = reg[a], r2 = reg[b];
                    if (r1 < 0 || r2 < 0) { ok = false; break; }
                    if (lastU[a] == i) occ[r1] = -1;
                    if (lastU[b] == i) occ[r2] = -1;
                    int d = -1;
                    for (int r = 0; r < P.regs; r++) if (occ[r] < 0) { d = r; break; }
                    if (d < 0) { ok = false; break; }
                    out.push_back({der[x].op, r1, r2, d});
                    reg[x] = d;
                    if (lastU[x] > i) occ[d] = x;
                }
                if (ok && verifyProgExact(P, out)) {
                    lock_guard<mutex> lk(mtx);
                    winProg = out;
                    cerr << "[WarmRescue] THREAD " << tid << " SOLVED ZERO-SPILL IN " << iters << " ITERS! len=" << out.size() << "\n";
                } else {
                    solved = false;
                }
            } else {
                lock_guard<mutex> lk(mtx);
                cerr << "[WarmRescue] tid=" << tid << " iters=" << iters << " bestLocal=" << bestLocal << "\n";
            }
        });
    }
    for (auto& th : pool) th.join();

    if (!winProg.empty()) {
        cout << "R " << P.id << " " << P.prog.size() << " " << winProg.size() << "\n";
        for (auto& ins : winProg) cout << (char)ins[0] << " " << ins[1] << " " << ins[2] << " " << ins[3] << "\n";
    }
    return 0;
}
