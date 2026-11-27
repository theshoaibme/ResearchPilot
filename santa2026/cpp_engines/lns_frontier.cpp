// Enhanced Value-Set Closure + MITM + Product-Near-Miss LNS Engine (Apache 2.0)
// Adapted & extended from viktorbarbarich LNS v3 + Triwalk product-near-miss ideas.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;
typedef long long ll;
typedef __int128 LL;

struct HSet {
    vector<ll> t;
    int mask;
    vector<int> log;
    HSet(int cap = 2048) {
        int s = 1;
        while (s < cap * 2) s <<= 1;
        t.assign(s, 0);
        mask = s - 1;
    }
    static inline unsigned h(ll x) {
        unsigned long long z = (unsigned long long)x * 0x9E3779B97F4A7C15ULL;
        return (unsigned)(z >> 32);
    }
    inline bool has(ll x) const {
        if (x <= 0) return false;
        int i = h(x) & mask;
        while (t[i]) {
            if (t[i] == x) return true;
            i = (i + 1) & mask;
        }
        return false;
    }
    inline void add(ll x) {
        if (x <= 0) return;
        int i = h(x) & mask;
        while (t[i]) {
            if (t[i] == x) return;
            i = (i + 1) & mask;
        }
        t[i] = x;
        log.push_back(i);
    }
    void rollback(size_t m) {
        while (log.size() > m) {
            t[log.back()] = 0;
            log.pop_back();
        }
    }
};

static inline bool derivable(ll x, const vector<ll>& D, const HSet& hs, ll LIM) {
    for (ll a : D) {
        if (a < x) {
            if (hs.has(x - a)) return true;
        } else if (a > x) {
            if (hs.has(a - x)) return true;
        }
        if (x % a == 0 && hs.has(x / a)) return true;
        if ((LL)a * x <= LIM && hs.has(a * x)) return true;
    }
    return false;
}

static bool derive(ll x, const vector<ll>& D, const HSet& hs, ll LIM, char& op, ll& p, ll& q) {
    for (ll a : D) {
        if (a < x && hs.has(x - a)) { op = '+'; p = a; q = x - a; return true; }
        if (a > x && hs.has(a - x)) { op = '-'; p = a; q = a - x; return true; }
        if (x % a == 0 && hs.has(x / a)) { op = '*'; p = a; q = x / a; return true; }
        if ((LL)a * x <= LIM && hs.has(a * x)) { op = '/'; p = a * x; q = a; return true; }
    }
    return false;
}

static bool order_values(const vector<ll>& S, const vector<ll>& vals, const vector<double>& prio, ll LIM, vector<ll>& order) {
    vector<ll> D(S.begin(), S.end());
    HSet hs(1024);
    for (ll s : S) hs.add(s);
    vector<int> idx(vals.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&](int a, int b) { return prio[a] < prio[b]; });
    vector<char> done(vals.size(), 0);
    order.clear();
    size_t cnt = 0, start = 0;
    while (cnt < vals.size()) {
        bool prog = false;
        while (start < idx.size() && done[idx[start]]) start++;
        for (size_t z = start; z < idx.size(); z++) {
            int i = idx[z];
            if (done[i]) continue;
            if (hs.has(vals[i])) {
                done[i] = 1;
                cnt++;
                prog = true;
                break;
            }
            if (derivable(vals[i], D, hs, LIM)) {
                done[i] = 1;
                cnt++;
                D.push_back(vals[i]);
                hs.add(vals[i]);
                order.push_back(vals[i]);
                prog = true;
                break;
            }
        }
        if (!prog) return false;
    }
    return true;
}

struct Built {
    vector<ll> vals;
    vector<char> op;
    vector<int> a, b;
};

static int peak_of(const Built& B, int ns) {
    int n = B.vals.size();
    vector<int> last(n, -1);
    for (int k = ns; k < n; k++) {
        last[B.a[k]] = max(last[B.a[k]], k);
        last[B.b[k]] = max(last[B.b[k]], k);
    }
    vector<int> diff(n + 2, 0);
    for (int j = 0; j < n; j++) {
        if (last[j] > j) {
            int from = max(j + 1, ns);
            if (from <= last[j] - 1) {
                diff[from]++;
                diff[last[j]]--;
            }
        }
    }
    int peak = 0, cur = 0;
    for (int k = 0; k < n; k++) {
        cur += diff[k];
        if (k >= ns) peak = max(peak, cur + 1);
    }
    return peak;
}

static bool build(const vector<ll>& S, const vector<ll>& order, ll LIM, Built& B, int& peak) {
    int ns = S.size();
    B.vals.assign(S.begin(), S.end());
    B.op.assign(ns, 0);
    B.a.assign(ns, -1);
    B.b.assign(ns, -1);
    unordered_map<ll, int> pos;
    for (int i = 0; i < ns; i++) pos.emplace(S[i], i);
    vector<ll> D(S.begin(), S.end());
    HSet hs(1024);
    for (ll s : S) hs.add(s);
    for (ll x : order) {
        char op;
        ll p, q;
        if (!derive(x, D, hs, LIM, op, p, q)) return false;
        B.vals.push_back(x);
        B.op.push_back(op);
        B.a.push_back(pos[p]);
        B.b.push_back(pos[q]);
        pos.emplace(x, (int)B.vals.size() - 1);
        D.push_back(x);
        hs.add(x);
    }
    peak = peak_of(B, ns);
    return true;
}

static bool reorder(Built& B, int ns, int tries, mt19937_64& rng) {
    int n = B.vals.size();
    vector<int> users(n, 0);
    for (int k = ns; k < n; k++) {
        users[B.a[k]]++;
        if (B.b[k] != B.a[k]) users[B.b[k]]++;
    }
    for (int t = 0; t < tries; t++) {
        vector<int> rem(users), order;
        order.reserve(n - ns);
        vector<char> done(n, 0);
        for (int i = 0; i < ns; i++) done[i] = 1;
        vector<unsigned> noise(n);
        for (auto& x : noise) x = rng() & 1023;
        for (int step = ns; step < n; step++) {
            int best = -1;
            long bs = 0;
            for (int c = ns; c < n; c++) {
                if (done[c] || !done[B.a[c]] || !done[B.b[c]]) continue;
                int delta = (rem[c] > 0 ? 1 : 0);
                if (rem[B.a[c]] == 1) delta--;
                if (B.b[c] != B.a[c] && rem[B.b[c]] == 1) delta--;
                long sc = (long)delta * 1000000 + (t == 0 ? c : (long)noise[c] * 1000 + c);
                if (best < 0 || sc < bs) {
                    best = c;
                    bs = sc;
                }
            }
            done[best] = 1;
            order.push_back(best);
            rem[B.a[best]]--;
            if (B.b[best] != B.a[best]) rem[B.b[best]]--;
        }
        vector<int> pos(n);
        for (int i = 0; i < ns; i++) pos[i] = i;
        for (int i = 0; i < (int)order.size(); i++) pos[order[i]] = ns + i;
        Built R;
        R.vals.resize(n);
        R.op.resize(n);
        R.a.resize(n);
        R.b.resize(n);
        for (int i = 0; i < ns; i++) {
            R.vals[i] = B.vals[i];
            R.op[i] = 0;
            R.a[i] = R.b[i] = -1;
        }
        for (int c = ns; c < n; c++) {
            int p = pos[c];
            R.vals[p] = B.vals[c];
            R.op[p] = B.op[c];
            R.a[p] = pos[B.a[c]];
            R.b[p] = pos[B.b[c]];
        }
        if (peak_of(R, ns) <= 48) {
            B = R;
            return true;
        }
    }
    return false;
}

// Compile zero-spill Built (peak <= 48) directly into JSON register instructions
static string compile_json(const Built& B, int ns) {
    int n = B.vals.size();
    vector<int> last(n, -1);
    for (int k = ns; k < n; k++) {
        last[B.a[k]] = max(last[B.a[k]], k);
        last[B.b[k]] = max(last[B.b[k]], k);
    }
    vector<int> reg_of(n, -1);
    vector<int> holder(48, -1);
    for (int i = 0; i < ns; i++) {
        reg_of[i] = i;
        if (last[i] >= ns) holder[i] = i;
    }
    ostringstream oss;
    oss << "[";
    for (int k = ns; k < n; k++) {
        int ra = reg_of[B.a[k]];
        int rb = reg_of[B.b[k]];
        if (last[B.a[k]] <= k && ra >= 0) holder[ra] = -1;
        if (last[B.b[k]] <= k && rb >= 0) holder[rb] = -1;
        int rd = -1;
        for (int r = 0; r < 48; r++) {
            if (holder[r] == -1) {
                rd = r;
                break;
            }
        }
        if (rd < 0) return "";
        reg_of[k] = rd;
        if (last[k] > k) holder[rd] = k;
        if (k > ns) oss << ",";
        oss << "[\"" << B.op[k] << "\"," << ra << "," << rb << "," << rd << "]";
    }
    oss << "]";
    return oss.str();
}

static void branches(ll u, const vector<ll>& D, ll LIM, vector<ll>& out) {
    out.clear();
    for (ll d : D) {
        if (u > d) out.push_back(u - d);
        if (d > u) out.push_back(d - u);
        if ((LL)u + d <= LIM) out.push_back(u + d);
        if (u % d == 0) out.push_back(u / d);
        if (d % u == 0) out.push_back(d / u);
        if ((LL)u * d <= LIM) out.push_back(u * d);
    }
    if (u % 2 == 0) out.push_back(u / 2);
    if ((LL)u * 2 <= LIM) out.push_back(u * 2);
    ll r = (ll)sqrtl((long double)u);
    while (r * r > u) r--;
    while ((r + 1) * (r + 1) <= u) r++;
    if (r * r == u) out.push_back(r);
}

struct Solver {
    vector<ll> S, T, V;
    ll LIM;
    HSet tset;
    string id;
    HSet P1{1 << 18};
    Built best;
    mt19937_64 rng;

    void buildP1(const vector<ll>& D) {
        P1.rollback(0);
        int n = D.size();
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                ll a = D[i], b = D[j];
                if (a < b) swap(a, b);
                if ((LL)a + b <= LIM) P1.add(a + b);
                if ((LL)a * b <= LIM) P1.add(a * b);
                if (a > b) P1.add(a - b);
                if (a % b == 0) P1.add(a / b);
                if (P1.log.size() > (size_t)(P1.mask / 2)) return;
            }
        }
    }

    bool make(ll u, int b, vector<ll>& D, HSet& hs, vector<ll>& W) {
        if (derivable(u, D, hs, LIM)) return true;
        if (b <= 0) return false;
        buildP1(D);
        vector<ll> br;
        branches(u, D, LIM, br);
        vector<ll> order(br);
        shuffle(order.begin(), order.end(), rng);
        for (ll x : order) {
            if (x > 0 && !hs.has(x) && P1.has(x)) {
                W.push_back(x);
                D.push_back(x);
                hs.add(x);
                return true;
            }
        }
        if (b < 2) return false;
        vector<ll> br2;
        for (ll x : order) {
            if (x <= 0 || hs.has(x) || x > LIM) continue;
            branches(x, D, LIM, br2);
            for (ll y : br2) {
                if (y > 0 && y != x && !hs.has(y) && P1.has(y)) {
                    W.push_back(y);
                    W.push_back(x);
                    D.push_back(y);
                    hs.add(y);
                    D.push_back(x);
                    hs.add(x);
                    return true;
                }
            }
        }
        // MITM over P1: u = x op y
        for (int i : P1.log) {
            ll x = P1.t[i];
            if (hs.has(x)) continue;
            ll c[5];
            int nc = 0;
            if (u > x) c[nc++] = u - x;
            if (x > u) c[nc++] = x - u;
            if ((LL)u + x <= LIM) c[nc++] = u + x;
            if (u % x == 0) c[nc++] = u / x;
            if ((LL)u * x <= LIM) c[nc++] = u * x;
            for (int k = 0; k < nc; k++) {
                ll y = c[k];
                if (y > 0 && !hs.has(y) && P1.has(y)) {
                    W.push_back(x);
                    D.push_back(x);
                    hs.add(x);
                    if (y != x) {
                        W.push_back(y);
                        D.push_back(y);
                        hs.add(y);
                    }
                    return true;
                }
            }
        }
        if (b < 3 || (rng() % 8 != 0)) return false;
        int tried = 0;
        for (ll w : order) {
            if (w <= 0 || hs.has(w) || ++tried > 50) break;
            for (int i : P1.log) {
                ll x = P1.t[i];
                if (hs.has(x) || x == w) continue;
                ll c[5];
                int nc = 0;
                if (w > x) c[nc++] = w - x;
                if (x > w) c[nc++] = x - w;
                if ((LL)w + x <= LIM) c[nc++] = w + x;
                if (w % x == 0) c[nc++] = w / x;
                if ((LL)w * x <= LIM) c[nc++] = w * x;
                for (int k = 0; k < nc; k++) {
                    ll y = c[k];
                    if (y > 0 && y != w && !hs.has(y) && P1.has(y)) {
                        W.push_back(x);
                        D.push_back(x);
                        hs.add(x);
                        if (y != x) {
                            W.push_back(y);
                            D.push_back(y);
                            hs.add(y);
                        }
                        W.push_back(w);
                        D.push_back(w);
                        hs.add(w);
                        return true;
                    }
                }
            }
        }
        return false;
    }

    void closure(vector<ll>& D, HSet& hs, vector<ll>& U) {
        bool ch = true;
        while (ch && !U.empty()) {
            ch = false;
            for (size_t i = 0; i < U.size();) {
                if (derivable(U[i], D, hs, LIM)) {
                    D.push_back(U[i]);
                    hs.add(U[i]);
                    U.erase(U.begin() + i);
                    ch = true;
                } else {
                    i++;
                }
            }
        }
    }

    void run(double secs) {
        tset = HSet();
        for (ll t : T) tset.add(t);
        int ns = S.size();
        auto t0 = chrono::steady_clock::now();
        long it = 0, lastImp = 0;
        Built B;
        int peak = 0;
        if (!build(S, V, LIM, B, peak)) return;
        best = B;
        vector<ll> bestV = V;
        int SLACK = 2;
        double PUP = 0.02;
        long RESET = 25000;

        while (true) {
            if (it - lastImp > RESET && V.size() > bestV.size()) {
                V = bestV;
                build(S, V, LIM, B, peak);
                lastImp = it;
            }
            if ((it & 15) == 0) {
                double el = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
                if (el > secs) break;
            }
            it++;
            int n = B.vals.size();
            if (n <= ns) break;
            vector<vector<int>> cons(n);
            for (int k = ns; k < n; k++) {
                cons[B.a[k]].push_back(k);
                if (B.b[k] != B.a[k]) cons[B.b[k]].push_back(k);
            }
            vector<char> inR(n, 0);
            vector<int> R;
            int nroots = (rng() % 10 < 3) ? 2 : 1;
            for (int q = 0; q < nroots; q++) {
                int r = ns + rng() % (n - ns);
                if (!inR[r]) {
                    inR[r] = 1;
                    R.push_back(r);
                }
            }
            bool ch = true;
            int cap = (rng() % 8 == 0) ? 8 + rng() % 16 : (rng() % 4 == 0 ? 1 + rng() % 3 : 3 + rng() % 8);
            while (ch && (int)R.size() < cap) {
                ch = false;
                for (size_t z = 0; z < R.size() && (int)R.size() < cap; z++) {
                    int x = R[z];
                    for (int p : {B.a[x], B.b[x]}) {
                        if (p < ns || inR[p]) continue;
                        bool priv = true;
                        for (int c : cons[p]) {
                            if (!inR[c]) {
                                priv = false;
                                break;
                            }
                        }
                        if (priv) {
                            inR[p] = 1;
                            R.push_back(p);
                            ch = true;
                        }
                    }
                }
            }
            int tR = 0;
            for (int x : R) if (tset.has(B.vals[x])) tR++;
            int nonT = R.size() - tR;
            bool plateau = rng() % 2;
            int budget = nonT - 1 + (plateau ? 1 : 0) + ((rng() % 4 == 0) ? 1 + rng() % SLACK : 0);
            if (budget < 0) continue;

            vector<ll> D(S.begin(), S.end());
            HSet hs(1024);
            for (ll s : S) hs.add(s);
            vector<ll> U;
            for (int k = ns; k < n; k++) if (!inR[k]) U.push_back(B.vals[k]);
            for (int x : R) if (tset.has(B.vals[x])) U.push_back(B.vals[x]);
            closure(D, hs, U);

            vector<ll> W;
            bool ok = true;
            while (!U.empty()) {
                ll u = U.front();
                int nb = min(3, budget - (int)W.size());
                if (nb < 0 || !make(u, nb, D, hs, W)) {
                    ok = false;
                    break;
                }
                closure(D, hs, U);
            }
            if (!ok || (int)W.size() > budget) continue;

            int newlen = (n - ns) - (int)R.size() + tR + (int)W.size();
            if (newlen > (int)bestV.size() + SLACK) continue;

            vector<ll> vals;
            vector<double> prio;
            int rmin = *min_element(R.begin(), R.end());
            for (int k = ns; k < n; k++) {
                if (!inR[k] || tset.has(B.vals[k])) {
                    vals.push_back(B.vals[k]);
                    prio.push_back(k);
                }
            }
            for (size_t j = 0; j < W.size(); j++) {
                vals.push_back(W[j]);
                prio.push_back(rmin - 0.5 + 0.001 * j);
            }
            {
                HSet sx(1024);
                for (ll s : S) sx.add(s);
                vector<ll> v2;
                vector<double> p2;
                for (size_t j = 0; j < vals.size(); j++) {
                    if (!sx.has(vals[j])) {
                        sx.add(vals[j]);
                        v2.push_back(vals[j]);
                        p2.push_back(prio[j]);
                    }
                }
                vals.swap(v2);
                prio.swap(p2);
            }
            vector<ll> order;
            if (!order_values(S, vals, prio, LIM, order)) continue;
            Built B2;
            int pk;
            if (!build(S, order, LIM, B2, pk)) continue;
            if (pk > 48) {
                if (order.size() > bestV.size() + SLACK || !reorder(B2, ns, 6, rng)) continue;
                order.assign(B2.vals.begin() + ns, B2.vals.end());
            }
            if (order.size() > bestV.size() + SLACK) continue;
            if (order.size() > V.size() && uniform_real_distribution<double>(0, 1)(rng) > PUP) continue;
            if (order.size() < bestV.size()) {
                fprintf(stderr, "[IMP] %s: %zu -> %zu\n", id.c_str(), bestV.size(), order.size());
                fflush(stderr);
                bestV = order;
                best = B2;
                lastImp = it;
            }
            V = order;
            B = B2;
        }
    }
};

int main(int argc, char** argv) {
    double secs = (argc > 1) ? atof(argv[1]) : 2.0;
    unsigned long long base_seed = (argc > 2) ? stoull(argv[2]) : 42ULL;
    string id;
    unsigned long long idx = 0;
    while (cin >> id) {
        Solver s;
        s.id = id;
        s.rng.seed(base_seed + (++idx) * 10007ULL);
        int n;
        cin >> n;
        s.S.resize(n);
        for (auto& x : s.S) cin >> x;
        cin >> n;
        s.T.resize(n);
        for (auto& x : s.T) cin >> x;
        cin >> s.LIM;
        cin >> n;
        vector<ll> V(n);
        for (auto& x : V) cin >> x;
        HSet seen(1024);
        for (ll x : s.S) seen.add(x);
        for (ll x : V) {
            if (!seen.has(x)) {
                seen.add(x);
                s.V.push_back(x);
            }
        }
        s.run(secs);
        string js = compile_json(s.best, s.S.size());
        printf("%s %s\n", id.c_str(), js.empty() ? "NONE" : js.c_str());
        fflush(stdout);
    }
    return 0;
}
