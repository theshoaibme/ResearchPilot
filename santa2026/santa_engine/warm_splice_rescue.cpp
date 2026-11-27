// WARM_SPLICE_RESCUE: Splices newly discovered smaller value sets (|S_new| < |P.prog|) directly into
// P.prog's working 48-register topological order (preserving 95%+ of the clash-free schedule!) and runs
// 10-thread 250M-iteration Coupled Re-Derive + Overflow-Window Relocation to hit 0 register spills!

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
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
    double secs = argc > 1 ? atof(argv[1]) : 15.0;
    int nThreads = argc > 2 ? atoi(argv[2]) : 10;
    string vsetFile = argc > 3 ? argv[3] : "";

    unordered_map<string, vector<vector<u64>>> candSets;
    if (!vsetFile.empty()) {
        ifstream vf(vsetFile);
        string ln;
        while (getline(vf, ln)) {
            istringstream iss(ln);
            string tag, pid; int o1, n1, dlt, chg;
            if (!(iss >> tag >> pid >> o1 >> n1 >> dlt >> chg) || tag != "V") continue;
            vector<u64> vs(n1);
            for (int i = 0; i < n1; i++) iss >> vs[i];
            candSets[pid].push_back(std::move(vs));
        }
    }

    string tag;
    while (cin >> tag) {
        if (tag != "P") continue;
        Puzzle P;
        int ns, nt, np;
        cin >> P.id >> ns; P.start.resize(ns);
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

        // Map original program step index for every value in P.prog!
        unordered_map<u64, double> origPos;
        {
            vector<u64> r(P.regs, 0);
            for (int i = 0; i < (int)P.start.size(); i++) { r[i] = P.start[i]; origPos[r[i]] = -1.0; }
            for (int i = 0; i < np; i++) {
                u64 va = r[P.prog[i][1]], vb = r[P.prog[i][2]];
                int op = P.prog[i][0];
                u64 v = op == '+' ? va + vb : op == '-' ? va - vb : op == '*' ? va * vb : va / vb;
                r[P.prog[i][3]] = v;
                if (!origPos.count(v)) origPos[v] = (double)i;
            }
        }

        vector<vector<u64>> toTry = candSets[P.id];
        if (toTry.empty()) {
            // Also try P.prog's own unique value set if it has duplicates (like scored_0774)
            vector<u64> ownS;
            unordered_set<u64> seen(P.start.begin(), P.start.end());
            vector<u64> r(P.regs, 0);
            for (int i = 0; i < (int)P.start.size(); i++) r[i] = P.start[i];
            for (auto& ins : P.prog) {
                u64 va = r[ins[1]], vb = r[ins[2]];
                u64 v = ins[0] == '+' ? va + vb : ins[0] == '-' ? va - vb : ins[0] == '*' ? va * vb : va / vb;
                r[ins[3]] = v;
                if (seen.insert(v).second) ownS.push_back(v);
            }
            if ((int)ownS.size() < np) toTry.push_back(ownS);
        }

        vector<array<int, 4>> bestWin;
        int curBestLen = np;

        for (const auto& rawS : toTry) {
            vector<u64> vals;
            unordered_map<u64, int> idOf;
            for (u64 s : P.start) if (!idOf.count(s)) { idOf[s] = (int)vals.size(); vals.push_back(s); }
            int nStart = (int)vals.size();
            for (u64 v : rawS) if (!idOf.count(v)) { idOf[v] = (int)vals.size(); vals.push_back(v); }
            int V = (int)vals.size();
            int N = V - nStart;
            if (N >= curBestLen) continue;

            // Precompute ALL valid 1-step derivations inside `vals`
            vector<vector<Der>> allDers(V);
            for (int x = nStart; x < V; x++) {
                u64 vx = vals[x];
                if (vx == 1) for (int s = 0; s < nStart; s++) allDers[x].push_back({s, s, '/'});
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

            // Build Warm-Spliced Initial Schedule: greedily emit ready nodes in ascending `origPos` order!
            // New nodes (not in origPos) inherit the earliest position right after their parents!
            vector<double> prio(V, 1e9);
            for (int i = 0; i < V; i++) {
                auto it = origPos.find(vals[i]);
                if (it != origPos.end()) prio[i] = it->second;
            }
            vector<char> done(V, 0);
            for (int s = 0; s < nStart; s++) done[s] = 1;
            vector<int> initSeq;
            vector<Der> initDer(V, {-1, -1, 0});
            while ((int)initSeq.size() < N) {
                int bestX = -1; double bestP = 1e18; Der bestD{-1, -1, 0};
                for (int x = nStart; x < V; x++) {
                    if (done[x]) continue;
                    for (const auto& d : allDers[x]) {
                        if (done[d.p] && done[d.q]) {
                            double px = (prio[x] < 1e8) ? prio[x] : (max(prio[d.p], prio[d.q]) + 0.25);
                            if (px < bestP) { bestP = px; bestX = x; bestD = d; }
                            break;
                        }
                    }
                }
                if (bestX < 0) break;
                done[bestX] = 1;
                if (prio[bestX] > 1e8) prio[bestX] = bestP;
                initSeq.push_back(bestX);
                initDer[bestX] = bestD;
            }
            if ((int)initSeq.size() != N) {
                cerr << "[WarmSplice] " << P.id << " N=" << N << " closure incomplete!\n";
                continue;
            }

            auto evalSeq = [&](const vector<int>& seq, const vector<Der>& der, int& pk, int& fo, int& lo_ov) {
                int lastU[512]; fill(lastU, lastU + V, -1);
                for (int i = 0; i < N; i++) { int x = seq[i]; lastU[der[x].p] = i; lastU[der[x].q] = i; }
                int die[512] = {};
                int cur = 0, exSum = 0;
                pk = 0; fo = -1; lo_ov = -1;
                for (int s = 0; s < nStart; s++) if (lastU[s] >= 0) { cur++; die[lastU[s]]++; }
                for (int i = 0; i < N; i++) {
                    int x = seq[i];
                    int need = cur - die[i] + 1;
                    if (need > pk) pk = need;
                    if (need > P.regs) {
                        exSum += (need - P.regs);
                        if (fo < 0) fo = i;
                        lo_ov = i;
                    }
                    cur -= die[i];
                    if (lastU[x] > i) { cur++; die[lastU[x]]++; }
                }
                return exSum == 0 ? 0 : (exSum * 2 + (pk - P.regs) * 10);
            };

            int pk0, fo0, lo0;
            int sc0 = evalSeq(initSeq, initDer, pk0, fo0, lo0);
            cerr << "[WarmSplice] " << P.id << " target_len=" << N << " (was " << curBestLen
                 << ") warm_init_score=" << sc0 << " peak=" << pk0 << " ov=[" << fo0 << ".." << lo0 << "]\n";

            atomic<bool> solved{false};
            mutex mtx;
            vector<array<int, 4>> winProg;
            int globalMinSc = sc0;

            auto tStart = chrono::steady_clock::now();
            vector<thread> pool;
            for (int tid = 0; tid < nThreads; tid++) {
                pool.emplace_back([&, tid]() {
                    Rng rng(987654321ULL + tid * 104729ULL);
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

                    int pk, fo, lo_ov;
                    int cur = evalSeq(seq, der, pk, fo, lo_ov);
                    int bestLocal = cur;
                    vector<int> bestSeq = seq;
                    vector<Der> bestDer = der;

                    for (long long iters = 0; cur > 0; iters++) {
                        if ((iters & 4095) == 0) {
                            if (solved.load(memory_order_relaxed)) break;
                            double el = chrono::duration<double>(chrono::steady_clock::now() - tStart).count();
                            if (el > secs) break;
                            if (iters > 0 && (iters % 262144) == 0 && cur > bestLocal + 16) {
                                seq = bestSeq; der = bestDer;
                                for (int i = 0; i < N; i++) idxOf[seq[i]] = i;
                                rebuildUsers();
                                cur = evalSeq(seq, der, pk, fo, lo_ov);
                            }
                        }
                        double wave = 0.5 * (1.0 + cos((double)(iters % 120000) * (2.0 * M_PI / 120000.0)));
                        double Tmin = 0.08 + 0.04 * (tid % 4);
                        double Tmax = 2.4 + 0.6 * (tid % 5);
                        double T = Tmin + (Tmax - Tmin) * wave;

                        int roll = (int)rng(20);
                        if (roll < 5) {
                            int i = rng(N), x = seq[i];
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
                                j = (maxP + 1) + (int)rng(hi - maxP);
                            }
                            Der oldD = der[x];
                            der[x] = cand;
                            if (j != i) mv(i, j);
                            int pk2, fo2, lo2;
                            int c2 = evalSeq(seq, der, pk2, fo2, lo2);
                            if (c2 <= cur || rng.u() < exp((cur - c2) / T)) {
                                rmUser(oldD.p, x); if (oldD.q != oldD.p) rmUser(oldD.q, x);
                                users[cand.p].push_back(x); if (cand.q != cand.p) users[cand.q].push_back(x);
                                if (j != i) for (int k = min(i, j); k <= max(i, j); k++) idxOf[seq[k]] = k;
                                cur = c2; pk = pk2; fo = fo2; lo_ov = lo2;
                                if (cur < bestLocal) { bestLocal = cur; bestSeq = seq; bestDer = der; }
                            } else {
                                if (j != i) mv(j, i);
                                der[x] = oldD;
                            }
                            continue;
                        }

                        int i = -1, dir = 0;
                        if (roll < 14 && fo >= 0 && lo_ov >= fo) {
                            int tPos = fo + (lo_ov > fo ? (int)rng(lo_ov - fo + 1) : 0);
                            int probe = (int)rng(tPos + 1);
                            int v = seq[probe], lu = -1;
                            for (int u : users[v]) lu = max(lu, idxOf[u]);
                            if (lu >= tPos) {
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
                            j = (lo_ov >= i && hi > lo_ov && rng(2) == 0) ? (lo_ov + 1 + (int)rng(hi - lo_ov)) : (i + 1 + (int)rng(hi - i));
                        } else if (dir == -1 && lo < i) {
                            j = (fo <= i && lo <= fo && rng(2) == 0) ? (lo + (int)rng(fo - lo + 1)) : (lo + (int)rng(i - lo));
                        } else {
                            j = lo + rng(hi - lo + 1);
                        }
                        if (j == i) continue;
                        mv(i, j);
                        int pk2, fo2, lo2;
                        int c2 = evalSeq(seq, der, pk2, fo2, lo2);
                        if (c2 <= cur || rng.u() < exp((cur - c2) / T)) {
                            for (int k = min(i, j); k <= max(i, j); k++) idxOf[seq[k]] = k;
                            cur = c2; pk = pk2; fo = fo2; lo_ov = lo2;
                            if (cur < bestLocal) { bestLocal = cur; bestSeq = seq; bestDer = der; }
                        } else mv(j, i);
                    }

                    if (cur == 0 && !solved.exchange(true)) {
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
                            cerr << "  *** [WarmSplice SOLVED] " << P.id << ": " << curBestLen << " -> " << out.size() << " ***\n";
                        } else solved = false;
                    } else {
                        lock_guard<mutex> lk(mtx);
                        globalMinSc = min(globalMinSc, bestLocal);
                    }
                });
            }
            for (auto& th : pool) th.join();
            if (!winProg.empty() && (int)winProg.size() < curBestLen) {
                curBestLen = (int)winProg.size();
                bestWin = std::move(winProg);
            } else {
                cerr << "[WarmSplice] " << P.id << " min_overflow_score=" << globalMinSc << "\n";
            }
        }

        if (!bestWin.empty()) {
            cout << "R " << P.id << " " << np << " " << bestWin.size() << "\n";
            for (auto& ins : bestWin) cout << (char)ins[0] << " " << ins[1] << " " << ins[2] << " " << ins[3] << "\n";
            cout.flush();
        }
    }
    return 0;
}
