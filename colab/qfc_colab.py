# %% [markdown]
# # 量子適用性健檢：Colab 完整版
# 網頁版（瀏覽器）有規模上限；這份筆記本用同樣的比較方法跑更大的資料。
# 1. 執行第一格安裝套件  2. 上傳 CSV  3. 在「設定」格選任務與欄位  4. 全部執行
#
# 量子部分為無雜訊精確模擬（numpy 量子態），並用 Qiskit Statevector 核對 QAOA 公式。綠燈只代表值得再測，不代表量子優勢。
# 輸贏用統計檢定判斷：分類用 Nadeau–Bengio 修正 t 檢定（5 折 × REPEATS 次），分布學習用逐筆配對檢定，最佳化的隨機方法各跑 SEEDS 次。
# 最佳化可讀三種格式：邊清單、QUBO 方陣（QUBO_MATRIX = True）、0/1 問題規格表（kind,a,b,value,penalty；可含「剛好選 k 個」限制）。

# %%
# !pip -q install qiskit pandas scikit-learn
import numpy as np, pandas as pd, time, itertools, math, sys
from scipy.linalg import cho_factor, cho_solve, eigh

# %% [markdown]
# ## 設定

# %%
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else 'data.csv'  # Colab：上傳後改成檔名
TASK = sys.argv[2] if len(sys.argv) > 2 else 'cls'   # 'opt'（最佳化）, 'cls'（分類）, 'gen'（分布學習）
LABEL = None          # 分類目標欄名；None = 自動（名為 label/is_bot… 或最後一欄）
SRC, DST, WEIGHT = None, None, None   # 最佳化欄位；None = 前三欄
SEED = 7
SEEDS = 3             # 最佳化：每種隨機方法獨立跑幾次
REPEATS = 3           # 分類：5 折交叉驗證重複幾次；分布學習：做幾次不同的資料切分
QUBO_MATRIX = False   # 最佳化：True = 把方陣當 QUBO（對角線是單一變數成本）；有 kind 欄的規格表會自動辨識
MAX_SV_QUBITS = 20    # QAOA 量子態模擬上限（20 約需 16 MB × 數個陣列）
CLS_QUBITS = 8        # 量子核函數量子位元數
CLS_MAX_N = 800
GEN_MAX_BITS = 14
rng = np.random.default_rng(SEED)
df = pd.read_csv(CSV_PATH)
print(df.shape); df.head()

# %% [markdown]
# ## 最佳化：MaxCut／Ising（最大化 Σ w·[z_i≠z_j]）

# %%
def build_graph(df):
    s, d = SRC or df.columns[0], DST or df.columns[1]
    w = WEIGHT or (df.columns[2] if df.shape[1] >= 3 else None)
    labels = pd.Index(pd.unique(pd.concat([df[s].astype(str), df[d].astype(str)])))
    a, b = labels.get_indexer(df[s].astype(str)), labels.get_indexer(df[d].astype(str))
    ww = df[w].astype(float).values if w else np.ones(len(df))
    E = {}
    for i, j, x in zip(a, b, ww):
        if i == j or x == 0: continue
        k = (min(i, j), max(i, j)); E[k] = E.get(k, 0) + x
    edges = np.array([(i, j, x) for (i, j), x in E.items() if x != 0], float)
    return len(labels), edges

def build_qubo(df):
    """0/1 problem → Ising with one reference spin. Spec table rows: kind,a,b,value[,penalty]
       linear: cost of choosing a | pair: cost of choosing a and b | member: variable b in group a | choose: group a picks exactly b"""
    cols = [c.lower().strip() for c in df.columns]; d = df.copy(); d.columns = cols
    names, idx = [], {}
    def vid(s):
        s = str(s).strip()
        if s not in idx: idx[s] = len(names); names.append(s)
        return idx[s]
    lin, pairs, olin, opairs, groups = {}, {}, {}, {}, {}
    def addp(M, i, j, v):
        if i != j: k = (min(i, j), max(i, j)); M[k] = M.get(k, 0) + v
    if 'kind' in cols:
        for _, r in d.iterrows():
            kind = str(r['kind']).strip().lower(); a = r.get('a'); b = r.get('b'); v = float(r['value']) if 'value' in cols and pd.notna(r.get('value')) else 0.0
            if kind in ('linear', 'cost'): i = vid(a); lin[i] = lin.get(i, 0) + v; olin[i] = olin.get(i, 0) + v
            elif kind in ('pair', 'conflict'): i, j = vid(a), vid(b); addp(pairs, i, j, v); addp(opairs, i, j, v)
            elif kind == 'member': groups.setdefault(a, {'k': 1, 'm': [], 'p': None})['m'].append(vid(b))
            elif kind in ('choose', 'exactly'):
                g = groups.setdefault(a, {'k': 1, 'm': [], 'p': None}); g['k'] = int(float(b)) if pd.notna(b) and str(b) != '' else 1
                if 'penalty' in cols and pd.notna(r.get('penalty')) and str(r.get('penalty')) != '': g['p'] = float(r['penalty'])
    else:  # square QUBO matrix
        M = df.values.astype(float); names = [str(c) for c in df.columns]
        for i in range(len(names)):
            lin[i] = olin[i] = M[i, i]
            for j in range(i + 1, len(names)):
                if M[i, j] + M[j, i]: pairs[(i, j)] = opairs[(i, j)] = M[i, j] + M[j, i]
    nv = len(names); sens = max(abs(olin.get(i, 0)) + sum(abs(v) for (a, b), v in opairs.items() if i in (a, b)) for i in range(nv))
    c0 = 0.0
    for g in groups.values():
        P = g['p'] if g['p'] is not None else max(1.0, 2 * sens); g['p'] = P; k = g['k']
        for i in g['m']: lin[i] = lin.get(i, 0) + P * (1 - 2 * k)
        for x, y in itertools.combinations(g['m'], 2): addp(pairs, x, y, 2 * P)
        c0 += P * k * k
    h = np.zeros(nv); J = {}; cst = c0          # x = (1 - s)/2
    for i, a in lin.items(): cst += a / 2; h[i] -= a / 2
    for (i, j), b in pairs.items(): cst += b / 4; h[i] -= b / 4; h[j] -= b / 4; J[(i, j)] = J.get((i, j), 0) + b / 4
    E = [(i, j, v) for (i, j), v in J.items() if abs(v) > 1e-12] + [(i, nv, h[i]) for i in range(nv) if abs(h[i]) > 1e-12]
    edges = np.array(E, float); meta = {'nv': nv, 'names': names, 'const': cst, 'sumW': float(edges[:, 2].sum()), 'olin': olin, 'opairs': opairs, 'groups': groups}
    return nv + 1, edges, meta

def decode_qubo(meta, z):
    x = (z[:meta['nv']] * z[meta['nv']] < 0).astype(int)
    obj = sum(v for i, v in meta['olin'].items() if x[i]) + sum(v for (i, j), v in meta['opairs'].items() if x[i] and x[j])
    viol = [g for g, d in meta['groups'].items() if sum(x[i] for i in d['m']) != d['k']]
    return obj, viol, [meta['names'][i] for i in range(meta['nv']) if x[i]]

def cut(edges, z): i, j, w =edges[:, 0].astype(int), edges[:, 1].astype(int), edges[:, 2]; return float(np.sum(w * (z[i] != z[j])))

def adj_lists(n, edges):
    A = [[] for _ in range(n)]
    for i, j, w in edges: A[int(i)].append((int(j), w)); A[int(j)].append((int(i), w))
    return A

def local_search(n, A, z):
    imp = True
    while imp:
        imp = False
        for i in range(n):
            d = sum(w * z[i] * z[j] for j, w in A[i])
            if d > 1e-12: z[i] = -z[i]; imp = True
    return z

def tabu(n, A, edges, seconds):
    z = local_search(n, A, rng.choice([-1, 1], n)); gain = np.array([sum(w * z[i] * z[j] for j, w in A[i]) for i in range(n)])
    cur = cut(edges, z); best, bz = cur, z.copy(); tl = np.zeros(n, int); it = 0; t0 = time.time()
    while time.time() - t0 < seconds:
        it += 1
        ok = (tl <= it) | (cur + gain > best + 1e-12)
        g = np.where(ok, gain, -np.inf); i = int(np.argmax(g)) if np.isfinite(g).any() else int(rng.integers(n))
        z[i] = -z[i]; cur += gain[i]; gain[i] = -gain[i]
        for j, w in A[i]: gain[j] += 2 * w * z[i] * z[j]
        tl[i] = it + max(3, n // 10) + int(rng.integers(4))
        if cur > best + 1e-12: best, bz = cur, z.copy()
    return bz

def sa(n, A, edges, sweeps):
    z = rng.choice([-1, 1], n); cur = cut(edges, z); best, bz = cur, z.copy()
    mw = np.mean(np.abs(edges[:, 2])); T = 2 * mw; f = (0.005) ** (1 / sweeps)
    for _ in range(sweeps):
        for i in rng.permutation(n):
            d = sum(w * z[i] * z[j] for j, w in A[i])
            if d >= 0 or rng.random() < math.exp(d / T):
                z[i] = -z[i]; cur += d
                if cur > best + 1e-12: best, bz = cur, z.copy()
        T *= f
    return bz

def exact(n, edges):  # vectorised enumeration, n <= 24
    x = np.arange(2 ** (n - 1), dtype=np.int64); val = np.zeros(len(x))
    for i, j, w in edges: val += w * (((x >> int(i)) & 1) != ((x >> int(j)) & 1))
    k = int(np.argmax(val)); return np.array([1 - 2 * ((k >> q) & 1) for q in range(n)])

# p=1 analytic <Z_u Z_v> for C=Σ J Z Z (Ozaeta et al. 2022), same convention as the web engine
def corr(adj, u, v, g, b):
    Au, Av = adj[u], adj[v]; c2 = 2 * g; Juv = Au.get(v, 0.0)
    pu = pv = P1 = P2 = 1.0
    for w, a in Au.items():
        if w == v: continue
        pu *= math.cos(c2 * a); bb = Av.get(w, 0.0); P1 *= math.cos(c2 * (a + bb)); P2 *= math.cos(c2 * (a - bb))
    for w, bb in Av.items():
        if w == u: continue
        pv *= math.cos(c2 * bb)
        if w not in Au: c = math.cos(c2 * bb); P1 *= c; P2 *= c
    return 0.5 * math.sin(4 * b) * math.sin(c2 * Juv) * (pu + pv) - 0.5 * math.sin(2 * b) ** 2 * (P1 - P2)

from scipy.optimize import minimize
def rqaoa(n, edges, ncut=8):
    sc = np.max(np.abs(edges[:, 2])); adj = [dict() for _ in range(n)]
    for i, j, w in edges: i, j = int(i), int(j); adj[i][j] = adj[i].get(j, 0) + w / sc; adj[j][i] = adj[j].get(i, 0) + w / sc
    active, stack, x0 = set(range(n)), [], None
    while len(active) > ncut:
        el = [(u, v, J) for u in active for v, J in adj[u].items() if u < v]
        if not el: break
        E = lambda p: sum(J * corr(adj, u, v, p[0], p[1]) for u, v, J in el)
        if x0 is None: x0 = min(((g, b) for g in np.linspace(0.2, 3.0, 12) for b in np.linspace(0.15, 1.4, 8)), key=E)
        x0 = minimize(E, x0, method='Nelder-Mead', options={'maxiter': 60}).x
        best = (0, None, None)
        for u in active:
            cand = set(adj[u]) | {w for v in adj[u] for w in adj[v]}
            for v in cand:
                if v <= u: continue
                c = corr(adj, u, v, *x0)
                if abs(c) > abs(best[0]) + 1e-12: best = (c, u, v)
        c, u, v = best
        if u is None: break
        s = 1 if c >= 0 else -1
        if len(adj[u]) > len(adj[v]): u, v = v, u
        for w, J in list(adj[u].items()):
            adj[w].pop(u, None)
            if w == v: continue
            nv = adj[v].get(w, 0) + s * J
            if abs(nv) < 1e-12: adj[v].pop(w, None); adj[w].pop(v, None)
            else: adj[v][w] = nv; adj[w][v] = nv
        adj[u].clear(); active.discard(u); stack.append((u, v, s))
    rest = sorted(active); bestE, bz = np.inf, None
    for bits in itertools.product([1, -1], repeat=len(rest)):
        zz = dict(zip(rest, bits)); e = sum(J * zz[a] * zz[w] for a in rest for w, J in adj[a].items() if w in zz and w > a)
        if e < bestE: bestE, bz = e, zz
    z = np.ones(n, int)
    for a, s_ in bz.items(): z[a] = s_
    for u, v, s in reversed(stack): z[u] = s * z[v]
    return z

def qaoa_sv(n, edges, p=1, cvar=None, warm=None, shots=1024):
    sc = np.max(np.abs(edges[:, 2])); x = np.arange(2 ** n)
    C = np.zeros(2 ** n); K = np.zeros(2 ** n)
    for i, j, w in edges:
        diff = ((x >> int(i)) & 1) != ((x >> int(j)) & 1); C += (w / sc) * np.where(diff, -1, 1); K += w * diff
    order = np.argsort(C)
    def run(par):
        g, b = par[:p], par[p:]
        if warm is None: psi = np.full(2 ** n, 2 ** (-n / 2), complex)
        else:
            psi = np.ones(1, complex)
            for q in reversed(range(n)): psi = np.kron(np.array([math.cos(warm[q] / 2), math.sin(warm[q] / 2)]), psi)
        for l in range(p):
            psi = psi * np.exp(-1j * g[l] * C)
            for q in range(n):
                if warm is None: U = np.array([[math.cos(b[l]), -1j * math.sin(b[l])], [-1j * math.sin(b[l]), math.cos(b[l])]])
                else:
                    th = warm[q]; Ry = lambda t: np.array([[math.cos(t / 2), -math.sin(t / 2)], [math.sin(t / 2), math.cos(t / 2)]])
                    U = Ry(th) @ np.diag([np.exp(1j * b[l]), np.exp(-1j * b[l])]) @ Ry(-th)
                psi = psi.reshape(2 ** (n - q - 1), 2, 2 ** q); psi = np.einsum('ab,ibj->iaj', U, psi).reshape(-1)
        return np.abs(psi) ** 2
    def obj(par):
        pr = run(par)
        if cvar is None: return float(pr @ C)
        acc = np.cumsum(pr[order]); m = acc <= cvar; tail = pr[order] * m
        k = np.searchsorted(acc, cvar); extra = cvar - (acc[k - 1] if k > 0 else 0)
        return float((tail @ C[order] + extra * C[order][min(k, len(C) - 1)]) / cvar)
    grid = [np.r_[[g * (l + 1) / p for l in range(p)], [b * (p - l) / p for l in range(p)]] for g in np.linspace(0.25, 2.9, 10) for b in np.linspace(0.15, 1.4, 8)]
    x0 = min(grid, key=obj); res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 80 + 40 * p})
    pr = run(res.x)  # optimisation is deterministic; repeat only the measurement
    return [float(K[rng.choice(2 ** n, size=shots, p=pr / pr.sum())].max()) for _ in range(SEEDS)], float(pr @ K)

def gw_mixing(n, A, edges, rounds=64):  # Goemans–Williamson SDP via low-rank mixing method + hyperplane rounding
    k = min(12, max(2, math.ceil(math.sqrt(2 * n)))); V = rng.standard_normal((n, k)); V /= np.linalg.norm(V, axis=1, keepdims=True)
    for _ in range(400):
        ch = 0.0
        for i in range(n):
            if not A[i]: continue
            g = sum(w * V[j] for j, w in A[i]); s = np.linalg.norm(g)
            if s > 1e-12: nv = -g / s; ch += np.abs(nv - V[i]).sum(); V[i] = nv
        if ch < 1e-7 * n: break
    best, bz = -np.inf, None
    for _ in range(rounds):
        z = np.where(V @ rng.standard_normal(k) >= 0, 1, -1); z = local_search(n, A, z); v = cut(edges, z)
        if v > best: best, bz = v, z.copy()
    return bz

def sv_arrays(n, edges):
    sc = np.max(np.abs(edges[:, 2])); x = np.arange(2 ** n); C = np.zeros(2 ** n); K = np.zeros(2 ** n)
    for i, j, w in edges:
        diff = ((x >> int(i)) & 1) != ((x >> int(j)) & 1); C += (w / sc) * np.where(diff, -1, 1); K += w * diff
    return C, K

def evolve(n, C, gs, bs):  # |+>^n, then layers of e^{-iγC} e^{-iβΣX}
    psi = np.full(2 ** n, 2 ** (-n / 2), complex)
    for g, b in zip(gs, bs):
        psi = psi * np.exp(-1j * g * C); U = np.array([[math.cos(b), -1j * math.sin(b)], [-1j * math.sin(b), math.cos(b)]])
        for q in range(n): psi = np.einsum('ab,ibj->iaj', U, psi.reshape(2 ** (n - q - 1), 2, 2 ** q)).reshape(-1)
    return np.abs(psi) ** 2

def anneal(n, edges, shots=1024):  # simulated quantum annealing, H(s) = -(1-s)ΣX + sC, linear schedule
    C, K = sv_arrays(n, edges); best = None
    for T in (2, 5, 10, 25):
        Kst = min(120, max(10, 4 * T)); dt = T / Kst; s = (np.arange(Kst) + 0.5) / Kst
        pr = evolve(n, C, dt * s, -dt * (1 - s)); e = pr @ C
        if best is None or e < best[0]: best = (e, pr)
    pr = best[1]; return [float(K[rng.choice(2 ** n, size=shots, p=pr / pr.sum())].max()) for _ in range(SEEDS)]

def he_probs(n, th, L):  # real-amplitude hardware-efficient circuit: RY layer + L × (CZ chain + RY layer)
    x = np.arange(2 ** n); cz = np.ones(2 ** n)
    for q in range(n - 1): cz *= np.where(((x >> q) & 1) & ((x >> (q + 1)) & 1), -1, 1)
    psi = np.zeros(2 ** n); psi[0] = 1; k = 0
    def ry(psi, q, t):
        c, s = math.cos(t / 2), math.sin(t / 2); p = psi.reshape(2 ** (n - q - 1), 2, 2 ** q); a0, a1 = p[:, 0, :].copy(), p[:, 1, :].copy()
        p[:, 0, :] = c * a0 - s * a1; p[:, 1, :] = s * a0 + c * a1; return p.reshape(-1)
    for q in range(n): psi = ry(psi, q, th[k]); k += 1
    for _ in range(L):
        psi = psi * cz
        for q in range(n): psi = ry(psi, q, th[k]); k += 1
    return psi ** 2

def train_he(n, L, dLdp, steps, lr=0.05):  # Adam + exact parameter-shift gradients
    P = n * (L + 1); th = np.r_[np.full(n, np.pi / 2), np.zeros(P - n)] + 0.1 * rng.standard_normal(P); m = np.zeros(P); v = np.zeros(P)
    for st in range(steps):
        gp = dLdp(he_probs(n, th, L)); g = np.zeros(P)
        for k in range(P):
            e = np.zeros(P); e[k] = np.pi / 2; g[k] = gp @ (he_probs(n, th + e, L) - he_probs(n, th - e, L)) / 2
        m = .9 * m + .1 * g; v = .999 * v + .001 * g * g; th -= lr * (m / (1 - .9 ** (st + 1))) / (np.sqrt(v / (1 - .999 ** (st + 1))) + 1e-8)
    return he_probs(n, th, L), P

def grover_adaptive(K):  # Dürr–Høyer / Gilliam et al. threshold search with exact amplification statistics
    N = len(K); srt = np.sort(K); y = K[rng.integers(N)]; calls = 0; m = 1.0; since = 0; cap = math.sqrt(N)
    while since < 4 * cap + 8:
        M = N - np.searchsorted(srt, y + 1e-9, side='right'); k = int(rng.integers(max(1, int(m)))); calls += k
        ps = math.sin((2 * k + 1) * math.asin(math.sqrt(M / N))) ** 2 if M else 0
        if rng.random() < ps: y = srt[N - M + rng.integers(M)]; m = 1; since = 0
        else: since += k + 1; m = min(m * 1.2, cap)
    return float(y), calls

def recommend_opt(n, edges, res):
    A = adj_lists(n, edges); col = {}; bip = True
    for s in range(n):
        if s in col: continue
        col[s] = 1; st = [s]
        while st and bip:
            u = st.pop()
            for v_, _ in A[u]:
                if v_ not in col: col[v_] = -col[u]; st.append(v_)
                elif col[v_] == col[u]: bip = False
    pos = (edges[:, 2] > 0).all(); cl = res[res.類型 == 'classical']; top = cl[cl.最好答案 >= cl.最好答案.max() - 1e-9].sort_values('秒').iloc[0]
    if bip and pos: return '二分圖兩色著色（廣度優先搜尋）：一步得到保證最佳解。工具：networkx nx.bipartite.color'
    if n <= 25: return f'精確求解：只有 {n} 個變數，用 Google OR-Tools CP-SAT 或 Gurobi 可保證最佳（「{top.方法}」{top.秒:.3f} 秒已找到最好答案）。'
    return (f'建議用「{top.方法}」（最快拿到最好答案，{top.秒:.3f} 秒）。其他選擇：禁忌搜尋／模擬退火多起點平行；'
            '要品質保證用 OR-Tools 或 Gurobi 整數規劃設時間上限（看 MIP gap）' + ('；GW 半定規劃保證 ≥ 87.8% 最佳（cvxpy）' if pos else '') + '。')

def analyze_opt(df):
    meta = None
    if 'kind' in [c.lower() for c in df.columns] or QUBO_MATRIX: n, edges, meta = build_qubo(df)
    else: n, edges = build_graph(df)
    A = adj_lists(n, edges); out = []; nd = meta['nv'] if meta else n
    def run(name, kind, fn, reps=SEEDS):  # fn returns z (array) or list of values; every random method is run `reps` times
        vals, zs, t0 = [], [], time.time()
        for _ in range(reps):
            o = fn()
            if isinstance(o, list): vals += o; break
            vals.append(cut(edges, o)); zs.append(o)
        z = zs[int(np.argmax(vals))] if zs else None; row = {'方法': name, '類型': kind, '最好答案': max(vals), '平均': float(np.mean(vals)), '次數': len(vals), '每次秒數': (time.time() - t0) / max(1, len(zs) or 1)}
        if meta is not None:
            row['成本（越低越好）'] = meta['const'] + meta['sumW'] - 2 * max(vals)
            if z is not None: obj, viol, chosen = decode_qubo(meta, z); row['滿足限制'] = not viol; row['選了'] = '、'.join(chosen[:12])
        out.append(row); return z
    if n <= 24: run('精確窮舉', 'classical', lambda: exact(n, edges), 1)
    greedy = run('貪婪區域搜尋 1 次', 'classical', lambda: local_search(n, A, rng.choice([-1, 1], n)))
    run('區域搜尋 32 起點', 'classical', lambda: max((local_search(n, A, rng.choice([-1, 1], n)) for _ in range(32)), key=lambda q: cut(edges, q)))
    run('模擬退火', 'classical', lambda: sa(n, A, edges, max(200, min(2000, 40000 // n))))
    run('禁忌搜尋', 'classical', lambda: tabu(n, A, edges, min(5, 0.2 + 0.01 * n)))
    run('GW 半定規劃', 'classical', lambda: gw_mixing(n, A, edges))
    run('RQAOA（確定性）', 'quantum', lambda: rqaoa(n, edges), 1)
    if n <= MAX_SV_QUBITS:
        for name, kw in [('QAOA p=1', {}), ('QAOA p=2', {'p': 2}), ('CVaR-QAOA α=0.1', {'cvar': 0.1}),
                         ('Warm-start QAOA', {'warm': [2 * math.asin(math.sqrt(0.25 if s > 0 else 0.75)) for s in greedy]})]:
            run(name, 'quantum', lambda: qaoa_sv(n, edges, **kw)[0], 1)
        run('量子退火（模擬）', 'quantum', lambda: anneal(n, edges), 1)
        K = sv_arrays(n, edges)[1]; run('Grover 適應性搜尋', 'quantum', lambda: [grover_adaptive(K)[0] for _ in range(SEEDS)], 1)
    if n <= 12:
        C, K = sv_arrays(n, edges)
        run('VQE（硬體高效線路）', 'quantum', lambda: [float(K[rng.choice(2 ** n, size=1024, p=(lambda pr: pr / pr.sum())(train_he(n, 2, lambda p: C, 40, 0.08)[0]))].max()) for _ in range(SEEDS)], 1)
    res = pd.DataFrame(out)
    bc, bq = res[res.類型 == 'classical'].最好答案.max(), res[res.類型 == 'quantum'].最好答案.max()
    lv = '紅燈：古典較好' if bq < bc - 1e-9 else ('紅燈：規模小，古典窮舉即保證最佳' if nd <= 20 else ('綠燈：量子找到更好的解，請把古典時間預算加大再比' if bq > bc + 1e-9 else '黃燈：量子追平最強古典（非優勢）'))
    print(f'{nd} 個決策變數、{len(edges)} 條關係、隨機方法各跑 {SEEDS} 次 → {lv}')
    rec = recommend_opt(n, edges, res.rename(columns={'每次秒數': '秒'}))
    if meta is not None and meta['groups']: rec += ' 有限制條件時，整數規劃（OR-Tools CP-SAT）可直接處理限制，不需要懲罰項。'
    print('建議：' + rec); print('量子方法的時間是模擬時間，不代表量子硬體的速度。'); return res

# Qiskit cross-check of the analytic formula on a tiny random graph
def qiskit_check():
    try:
        from qiskit import QuantumCircuit; from qiskit.quantum_info import Statevector, SparsePauliOp
    except ImportError: print('未安裝 qiskit，略過核對'); return
    n = 5; E = [(0, 1, 1.0), (1, 2, -0.5), (2, 3, 0.8), (3, 4, 1.0), (0, 4, 0.3), (1, 3, 0.6)]; g, b = 0.7, 0.4
    qc = QuantumCircuit(n); qc.h(range(n))
    for i, j, w in E: qc.rzz(2 * g * w, i, j)
    qc.rx(2 * b, range(n)); sv = Statevector(qc)
    adj = [dict() for _ in range(n)]
    for i, j, w in E: adj[i][j] = w; adj[j][i] = w
    err = max(abs(sv.expectation_value(SparsePauliOp.from_sparse_list([('ZZ', [u, v], 1)], n)).real - corr(adj, u, v, g, b)) for u in range(n) for v in range(u + 1, n))
    print('Qiskit 核對 p=1 公式，最大誤差', err)

# %% [markdown]
# ## 分類：量子核函數 vs 古典

# %%
def prep_cls(df):
    lab = LABEL or next((c for c in df.columns if c.lower() in ('label', 'class', 'target', 'y', 'is_bot', 'fraud', 'churn')), df.columns[-1])
    y_raw = df[lab].astype(str); X = df.drop(columns=[lab])
    num = X.select_dtypes('number'); cat = X.select_dtypes(exclude='number'); cat = cat[[c for c in cat if cat[c].nunique() <= 10]]
    X = pd.concat([num.fillna(num.mean()), pd.get_dummies(cat, dtype=float)], axis=1).values.astype(float)
    classes = sorted(y_raw.unique()); y = y_raw.map({c: i for i, c in enumerate(classes)}).values
    if len(y) > CLS_MAX_N: idx = rng.choice(len(y), CLS_MAX_N, replace=False); X, y = X[idx], y[idx]
    X = (X - X.mean(0)) / (X.std(0) + 1e-12); return X, y, classes

def zz_states(Z, lam):
    q = Z.shape[1]; s = np.arange(2 ** q); zb = 1 - 2 * ((s[:, None] >> np.arange(q)) & 1)
    a = lam * Z; ph = a @ zb.T
    for i in range(q):
        for j in range(i + 1, q): ph += 0.5 * lam * np.outer((np.pi - a[:, i]) * (np.pi - a[:, j]), zb[:, i] * zb[:, j])
    psi = np.exp(1j * ph) / 2 ** (q / 2); H = np.array([[1]])
    for _ in range(q): H = np.kron(H, np.array([[1, 1], [1, -1]]) / math.sqrt(2))
    psi = (psi @ H.T) * np.exp(1j * ph); return psi

def krr_acc(K, y, tr, te, alpha):
    Y = -np.ones((len(tr), y.max() + 1)); Y[np.arange(len(tr)), y[tr]] = 1
    A = cho_solve(cho_factor(K[np.ix_(tr, tr)] + alpha * np.eye(len(tr))), Y); pred = (K[np.ix_(te, tr)] @ A).argmax(1)
    return np.mean([np.mean(pred[y[te] == c] == c) for c in np.unique(y[te])])

def make_folds(y, k=5):  # stratified; every model uses the same folds
    folds = [[] for _ in range(k)]; f = 0
    for c in np.unique(y):
        for i in rng.permutation(np.where(y == c)[0]): folds[f % k].append(i); f += 1
    return [np.array(fd) for fd in folds]

def cv_kernel(Ks, y, folds):
    sc = []
    for f in range(len(folds)):
        te = folds[f]; tr = rng.permutation(np.concatenate([folds[g] for g in range(len(folds)) if g != f])); itr, iva = tr[: int(.75 * len(tr))], tr[int(.75 * len(tr)):]
        K, a = max(((K, a) for K in Ks for a in (0.01, 0.1, 1)), key=lambda t: krr_acc(t[0], y, itr, iva, t[1]))
        sc.append(krr_acc(K, y, tr, te, a))
    return sc

def cv_sklearn(est, grid, X, y, folds):
    from sklearn.model_selection import GridSearchCV; from sklearn.metrics import balanced_accuracy_score
    sc = []
    for f in range(len(folds)):
        te = folds[f]; tr = np.concatenate([folds[g] for g in range(len(folds)) if g != f])
        gs = GridSearchCV(est, grid, cv=3, scoring='balanced_accuracy').fit(X[tr], y[tr]); sc.append(balanced_accuracy_score(y[te], gs.predict(X[te])))
    return sc

def projected_features(psi, q):  # <X>,<Y>,<Z> of every qubit (Huang et al. 2021)
    F = []
    for k in range(q):
        p = psi.reshape(len(psi), 2 ** (q - k - 1), 2, 2 ** k); a0, a1 = p[:, :, 0, :], p[:, :, 1, :]
        F += [2 * np.real(np.conj(a0) * a1).sum((1, 2)), 2 * np.imag(np.conj(a0) * a1).sum((1, 2)), (np.abs(a0) ** 2 - np.abs(a1) ** 2).sum((1, 2))]
    return np.array(F).T

def alignment(K, y, idx):  # centred kernel–target alignment (Cortes et al. 2012), training rows only
    Kt = K[np.ix_(idx, idx)]; Y = (y[idx][:, None] == y[idx][None, :]).astype(float); H = np.eye(len(idx)) - 1 / len(idx)
    Kc, Yc = H @ Kt @ H, H @ Y @ H; return float((Kc * Yc).sum() / np.sqrt((Kc * Kc).sum() * (Yc * Yc).sum()))

def vqc_cv(psis, y, folds, q, L=2, steps=150, batch=32, lr=0.08):  # psis: {bandwidth: (states, ZZ kernel)}  # variational quantum classifier, logits = w·<Z_c> + b
    nc = y.max() + 1
    if nc > q: return None
    x = np.arange(2 ** q); cz = np.ones(2 ** q); zc = np.array([1 - 2 * ((x >> c) & 1) for c in range(nc)])
    for a in range(q - 1): cz *= np.where(((x >> a) & 1) & ((x >> (a + 1)) & 1), -1, 1)
    def expz(S, th):
        k = 0
        def ry(S, a, t):
            c, s = math.cos(t / 2), math.sin(t / 2); p = S.reshape(len(S), 2 ** (q - a - 1), 2, 2 ** a); a0, a1 = p[:, :, 0, :].copy(), p[:, :, 1, :].copy()
            p[:, :, 0, :] = c * a0 - s * a1; p[:, :, 1, :] = s * a0 + c * a1; return p.reshape(len(S), -1)
        S = S.copy()
        for a in range(q): S = ry(S, a, th[k]); k += 1
        for _ in range(L):
            S = S * cz
            for a in range(q): S = ry(S, a, th[k]); k += 1
        return (np.abs(S) ** 2) @ zc.T
    sc = []; P = q * (L + 1)
    for f in range(len(folds)):
        te = folds[f]; tr = np.concatenate([folds[g] for g in range(len(folds)) if g != f])
        psi = max(psis.values(), key=lambda v: alignment(v[1], y, tr))[0]
        th = 0.3 * rng.standard_normal(P); w = np.full(nc, 2.0); b = np.zeros(nc); prm = np.r_[th, w, b]; m = np.zeros_like(prm); v = np.zeros_like(prm)
        for st in range(steps):
            bi = rng.choice(tr, min(batch, len(tr)), replace=False); S = psi[bi]; th, w, b = prm[:P], prm[P:P + nc], prm[P + nc:]
            E = expz(S, th); z = E * w + b; z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
            dz = p.copy(); dz[np.arange(len(bi)), y[bi]] -= 1; g = np.zeros_like(prm)
            g[P:P + nc] = (dz * E).mean(0); g[P + nc:] = dz.mean(0)
            for k in range(P):
                e = np.zeros(P); e[k] = np.pi / 2; g[k] = ((dz * w) * (expz(S, th + e) - expz(S, th - e)) / 2).sum(1).mean()
            m = .9 * m + .1 * g; v = .999 * v + .001 * g * g; prm = prm - lr * (m / (1 - .9 ** (st + 1))) / (np.sqrt(v / (1 - .999 ** (st + 1))) + 1e-8)
        th, w, b = prm[:P], prm[P:P + nc], prm[P + nc:]; pred = (expz(psi[te], th) * w + b).argmax(1)
        sc.append(np.mean([np.mean(pred[y[te] == c] == c) for c in np.unique(y[te])]))
    return sc

def recommend_cls(res, y):
    simple = ['邏輯回歸', '線性', 'k 近鄰', 'RBF', '隨機森林', '神經網路']
    tools = {'邏輯回歸': 'scikit-learn LogisticRegression', '線性': 'scikit-learn LogisticRegression／LinearSVC', 'k 近鄰': 'scikit-learn KNeighborsClassifier',
             'RBF': 'scikit-learn SVC(kernel="rbf")', '隨機森林': 'scikit-learn RandomForestClassifier；XGBoost、LightGBM', '神經網路': 'scikit-learn MLPClassifier；PyTorch'}
    cl = res[res.類型 == 'classical'].sort_values('平衡準確率', ascending=False); top = cl.iloc[0]
    near = cl[top.平衡準確率 - cl.平衡準確率 <= max(top.標準差 / math.sqrt(5), 0.005)].copy()
    near['r'] = near.模型.map(lambda s: next((i for i, k in enumerate(simple) if s.startswith(k)), 99)); pick = near.sort_values('r').iloc[0]
    key = next((k for k in simple if pick.模型.startswith(k)), '線性'); cnt = np.bincount(y)
    msg = f'建議用「{pick.模型}」（{pick.平衡準確率:.1%}）' + ('，分數和第一名差距在誤差內且更簡單' if pick.模型 != top.模型 else '') + f'。工具：{tools[key]}。'
    if cnt.min() / cnt.max() < 0.3: msg += ' 類別不平衡：訓練設 class_weight="balanced"，評估看各類別召回率。'
    return msg

def corrected_t(d, test_frac=0.2):  # Nadeau & Bengio (2003) corrected resampled t-test, one-sided p-values
    from scipy.stats import t as tdist
    d = np.asarray(d); J = len(d); se = math.sqrt((1 / J + test_frac / (1 - test_frac)) * d.var(ddof=1)) or 1e-12; tv = d.mean() / se
    return tv, tdist.sf(tv, J - 1), tdist.sf(-tv, J - 1)

def analyze_cls(df):
    from sklearn.linear_model import LogisticRegression; from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier; from sklearn.neighbors import KNeighborsClassifier
    X, y, classes = prep_cls(df); q = min(CLS_QUBITS, X.shape[1]); foldsets = [make_folds(y) for _ in range(REPEATS)]
    w, V = eigh(np.cov(X.T)); Z = X @ V[:, ::-1][:, :q]; Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    D = lambda M: ((M[:, None, :] - M[None, :, :]) ** 2).sum(-1)
    Df, Dz = D(X), D(Z); rows = []
    def add(name, kind, fn):
        sc = [fn(fs) for fs in foldsets]
        if sc[0] is None: return
        sc = sum(sc, []); rows.append({'模型': name, '類型': kind, '平衡準確率': float(np.mean(sc)), '標準差': float(np.std(sc)), 'scores': sc})
    add('線性核（全部特徵）', 'classical', lambda f: cv_kernel([X @ X.T + 1], y, f))
    add('RBF 核函數（全部特徵）', 'classical', lambda f: cv_kernel([np.exp(-s / X.shape[1] * Df) for s in (0.03, 0.1, 0.3, 1, 3)], y, f))
    add(f'RBF 核函數（同 {q} 維）', 'classical', lambda f: cv_kernel([np.exp(-s / q * Dz) for s in (0.03, 0.1, 0.3, 1, 3)], y, f))
    add('邏輯回歸（全部特徵）', 'classical', lambda f: cv_sklearn(LogisticRegression(max_iter=2000), {'C': [0.1, 1, 10]}, X, y, f))
    add('k 近鄰（全部特徵）', 'classical', lambda f: cv_sklearn(KNeighborsClassifier(), {'n_neighbors': [3, 5, 9, 15]}, X, y, f))
    add('隨機森林（全部特徵）', 'classical', lambda f: cv_sklearn(RandomForestClassifier(n_estimators=200, random_state=SEED), {'max_depth': [None, 8]}, X, y, f))
    add('神經網路（全部特徵）', 'classical', lambda f: cv_sklearn(MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=SEED), {'alpha': [1e-4, 1e-2]}, X, y, f))
    QK, PK, states = [], [], {}
    for lam in (0.05, 0.1, 0.2, 0.4, 0.8):
        P = zz_states(Z, lam); Kq = np.abs(P.conj() @ P.T) ** 2; QK.append(Kq)
        if lam <= 0.4: states[lam] = (P, Kq)
        if lam in (0.1, 0.4):
            Dp = D(projected_features(P, q)); PK += [np.exp(-s / (3 * q) * Dp) for s in (0.3, 1, 3)]
    add(f'量子核函數 ZZ（{q} 量子位元）', 'quantum', lambda f: cv_kernel(QK, y, f))
    add(f'投影量子核函數（{q} 量子位元）', 'quantum', lambda f: cv_kernel(PK, y, f))
    add(f'變分量子分類器 VQC（{q} 量子位元）', 'quantum', lambda f: vqc_cv(states, y, f, q))
    res = pd.DataFrame(rows)
    c = res[res.類型 == 'classical'].sort_values('平衡準確率').iloc[-1]; qm = res[res.類型 == 'quantum'].sort_values('平衡準確率').iloc[-1]
    nq = (res.類型 == 'quantum').sum(); tv, pQ, pC = corrected_t(np.array(qm.scores) - np.array(c.scores))
    print(f'最好的量子：{qm.模型} {qm.平衡準確率:.1%}；最好的古典：{c.模型} {c.平衡準確率:.1%}；t = {tv:.2f}，p(量子較好) = {pQ:.3f}，p(古典較好) = {pC:.3f}（5 折 × {REPEATS} 次）')
    print('綠燈：量子顯著較好，請用更多資料再驗證' if pQ < 0.05 / nq and qm.平衡準確率 - c.平衡準確率 >= 0.01 else ('紅燈：古典顯著較準' if pC < 0.05 else '黃燈：差距不顯著，選古典'))
    print(recommend_cls(res, y)); return res.drop(columns='scores')

# %% [markdown]
# ## 分布學習：IQP 量子生成模型 vs 古典

# %%
def analyze_gen(df):
    B = []
    for c in df.columns:
        v = df[c]
        if v.nunique() == 2: b = (v == sorted(v.dropna().unique())[1]).astype(int)   # same rule as the web version
        elif v.dtype.kind in 'if': b = (v > v.median()).astype(int)
        else: b = (v == v.mode()[0]).astype(int)
        p = b.mean()
        if 0.02 < p < 0.98: B.append(b.values)
    B = np.array(B[:GEN_MAX_BITS]).T; nb = B.shape[1]; N = 2 ** nb
    x = (B * (1 << np.arange(nb))).sum(1)
    nll = lambda p, d: float(-np.mean(np.log2(np.maximum(p[d], 1e-12))))
    allx = np.arange(N); bits = (allx[:, None] >> np.arange(nb)) & 1
    terms = [(i,) for i in range(nb)] + list(itertools.combinations(range(nb), 2))
    F = np.array([np.prod(1 - 2 * bits[:, list(t)], 1) for t in terms]).astype(float)
    H = np.array([[1]])
    for _ in range(nb): H = np.kron(H, np.array([[1, 1], [1, -1]]) / math.sqrt(2))
    def models(tr):  # every model is fitted on the training rows only; returns [(name, kind, p over all 2^nb patterns)]
        out = []; inner, val = tr[: int(len(tr) * .8)], tr[int(len(tr) * .8):]
        pi = ((tr[:, None] >> np.arange(nb)) & 1).mean(0) * len(tr) / (len(tr) + 2) + 1 / (len(tr) + 2)
        out.append(('各欄獨立', 'classical', np.prod(np.where(bits, pi, 1 - pi), 1)))
        tb = (tr[:, None] >> np.arange(nb)) & 1
        def J(i, j): c = np.ones((2, 2)); np.add.at(c, (tb[:, i], tb[:, j]), 1); return c / c.sum()
        MI = np.zeros((nb, nb))
        for i, j in itertools.combinations(range(nb), 2):
            P = J(i, j); MI[i, j] = MI[j, i] = np.sum(P * np.log(P / np.outer(P.sum(1), P.sum(0))))
        par, used = {0: None}, {0}
        while len(used) < nb:
            i, j = max(((i, j) for i in used for j in range(nb) if j not in used), key=lambda t: MI[t]); par[j] = i; used.add(j)
        p_cl = np.where(bits[:, 0], pi[0], 1 - pi[0])
        for j, i in par.items():
            if i is None: continue
            P = J(i, j); cond = P / P.sum(1, keepdims=True); p_cl = p_cl * cond[bits[:, i], bits[:, j]]
        out.append(('Chow-Liu 樹', 'classical', p_cl))
        hist = lambda d, a: (np.bincount(d, minlength=N) + a) / (len(d) + a * N)
        a_best = min((0.01, 0.05, 0.2, 0.5, 1, 2), key=lambda a: nll(hist(inner, a), val))   # smoothing chosen without test data
        out.append(('平滑直方圖', 'classical', hist(tr, a_best)))
        def bern_mix(d, K):
            db = (d[:, None] >> np.arange(nb)) & 1; pk = np.full(K, 1 / K); mu = 0.25 + 0.5 * rng.random((K, nb))
            for _ in range(80):
                lg = np.log(pk) + db @ np.log(mu).T + (1 - db) @ np.log(1 - mu).T; R = np.exp(lg - lg.max(1, keepdims=True)); R /= R.sum(1, keepdims=True)
                Nk = R.sum(0); pk = (Nk + 1) / (len(d) + K); mu = (R.T @ db + 1) / (Nk[:, None] + 2)
            return np.exp(np.log(pk) + bits @ np.log(mu).T + (1 - bits) @ np.log(1 - mu).T).sum(1)
        Kb = min((2, 4, 8), key=lambda K: nll(bern_mix(inner, K), val)); out.append(('伯努利混合模型', 'classical', bern_mix(tr, Kb)))
        c = np.bincount(tr, minlength=N) / len(tr); probs = lambda th: np.abs(H @ (np.exp(1j * (th @ F)) / math.sqrt(N))) ** 2
        best_th = None
        for sig in (0.05, 0.4, 0.8, 1.2):  # random restarts; keep lowest training loss
            th = sig * rng.standard_normal(len(terms)); m = np.zeros_like(th); v2 = np.zeros_like(th)
            for st in range(800):
                vv = np.exp(1j * (th @ F)) / math.sqrt(N); a = H @ vv; pr = np.abs(a) ** 2
                bvec = c * a.conj() / np.maximum(pr, 1e-15); h = H @ bvec; g = 2 * (F @ np.imag(vv * h))
                m = .9 * m + .1 * g; v2 = .999 * v2 + .001 * g * g; th -= 0.05 * (m / (1 - .9 ** (st + 1))) / (np.sqrt(v2 / (1 - .999 ** (st + 1))) + 1e-8)
            if best_th is None or nll(probs(th), tr) < nll(probs(best_th), tr): best_th = th
        out.append((f'IQP 量子生成模型（{nb} 量子位元）', 'quantum', probs(best_th)))
        if nb <= 10:  # hardware-efficient quantum circuit Born machine, exact parameter-shift gradient of the NLL
            pq, _ = train_he(nb, 3, lambda p: np.where(c > 0, -c / np.maximum(p, 1e-12), 0), 150 if nb <= 8 else 80)
            out.append((f'量子電路生成模型 QCBM（{nb} 量子位元）', 'quantum', pq))
        return out
    nte = len(x) * 3 // 10; per = []   # per split: {name: (kind, per-sample log2 p on test)}
    for rep_ in range(REPEATS):
        idx = rng.permutation(len(x)); te, tr = x[idx[:nte]], x[idx[nte:]]
        per.append({nm: (k, np.log2(np.maximum(p[te], 1e-12))) for nm, k, p in models(tr)})
    names = list(per[0]); col = '測試 bits/筆（越低越好）'
    res = pd.DataFrame([{'模型': nm, '類型': per[0][nm][0], col: float(np.mean([-per[r][nm][1].mean() for r in range(REPEATS)])), '標準差（切分間）': float(np.std([-per[r][nm][1].mean() for r in range(REPEATS)]))} for nm in names])
    c = res[res.類型 == 'classical'].sort_values(col).iloc[0].模型; qm = res[res.類型 == 'quantum'].sort_values(col).iloc[0].模型
    d = [per[r][qm][1] - per[r][c][1] for r in range(REPEATS)]   # bits saved by quantum, paired per test row
    dm = float(np.mean([v.mean() for v in d])); se = float(np.mean([v.std(ddof=1) for v in d])) / math.sqrt(nte)   # SE of one split: conservative, splits overlap
    from scipy.stats import norm; z = dm / (se or 1e-12); pQ, pC = norm.sf(z), norm.sf(-z); nq = (res.類型 == 'quantum').sum()
    print(f'最好的量子：{qm}；最好的古典：{c}；量子每筆少 {dm:.3f} bits，z = {z:.2f}，p(量子較好) = {pQ:.3f}，p(古典較好) = {pC:.3f}（{REPEATS} 次切分）')
    print('綠燈：量子顯著較好，請用更多資料再驗證' if pQ < 0.05 / nq and dm >= 0.02 else ('紅燈：古典顯著較好' if pC < 0.05 else '黃燈：差距不顯著，選古典'))
    tools = {'各欄': '各欄獨立抽樣即可', 'Chow-Liu': '貝氏網路（pgmpy TreeSearch＋BayesianNetwork）', '平滑': '經驗分布（pandas value_counts 加平滑）', '伯努利': '潛在類別模型（StepMix／poLCA）'}
    print('建議：' + next(v for k, v in tools.items() if c.startswith(k)) + '；要大量合成資料可用 SDV（Gaussian Copula、CTGAN）。'); return res

# %% [markdown]
# ## 執行

# %%
qiskit_check()
result = {'opt': analyze_opt, 'cls': analyze_cls, 'gen': analyze_gen}[TASK](df)
result
