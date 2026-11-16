"""Compare the JS engine (web) with the Python version (Colab) on deterministic quantities.
Usage: python tests/crosscheck.py   (needs node, numpy, scipy, pandas)"""
import json, subprocess, pathlib, sys, io, numpy as np, pandas as pd
root = pathlib.Path(__file__).resolve().parent.parent
js = json.loads(subprocess.run(['node', str(root / 'tests' / 'crosscheck_js.js')], capture_output=True, check=True).stdout.decode('utf8'))
src = (root / 'colab' / 'qfc_colab.py').read_text(encoding='utf8').split('# %% [markdown]\n# ## 執行')[0]
src = src.replace('df = pd.read_csv(CSV_PATH)', 'df = None').replace("print(df.shape); df.head()", '')
ns = {}; sys.argv = ['x', 'none.csv', 'opt']; exec(compile(src, 'qfc_colab.py', 'exec'), ns)
fails = 0
def check(name, ok, info=''):
    global fails; print(('PASS ' if ok else 'FAIL ') + name + ('  ' + info if info else '')); fails += 0 if ok else 1
def warn(name, ok, info=''): print(('PASS ' if ok else 'WARN ') + name + ('  ' + info if info else ''))
# 1. correlations
E = [(0, 1, 1.0), (1, 2, -0.5), (2, 3, 0.8), (3, 4, 1.0), (0, 4, 0.3), (1, 3, 0.6)]; adj = [dict() for _ in range(5)]
for i, j, w in E: adj[i][j] = w; adj[j][i] = w
py = [ns['corr'](adj, u, v, g, b) for g, b in [(0.7, 0.4), (1.3, 0.9)] for u in range(5) for v in range(u + 1, 5)]
check('p=1 相關係數（RQAOA 核心公式）', np.allclose(py, js['corr'], atol=1e-12), f"max diff {np.max(np.abs(np.array(py) - js['corr'])):.1e}")
# 2. quantum kernel
X = np.array([[0.1, -0.5, 0.3], [1.2, 0.4, -0.7], [-0.3, 0.9, 0.2], [0.5, 0.5, 0.5]]); P = ns['zz_states'](X, 0.4); K = (np.abs(P.conj() @ P.T) ** 2).ravel()
check('ZZ 量子核函數', np.allclose(K, js['kernel'], atol=1e-10), f"max diff {np.max(np.abs(K - js['kernel'])):.1e}")
# 3. QUBO conversion and exact optimum
n, edges, meta = ns['build_qubo'](pd.read_csv(root / 'samples' / 'schedule.csv'))
z = ns['exact'](n, edges); cost = meta['const'] + meta['sumW'] - 2 * ns['cut'](edges, z)
check('QUBO 轉換常數', abs(meta['const'] - js['qubo']['const']) < 1e-9 and abs(meta['sumW'] - js['qubo']['sumW']) < 1e-9, f"const {meta['const']} vs {js['qubo']['const']}")
check('QUBO 精確最低成本', abs(cost - js['qubo']['exactCost']) < 1e-9, f'{cost} vs {js["qubo"]["exactCost"]}')
nj, ej = ns['build_graph'](pd.read_csv(root / 'samples' / 'jobs.csv'))
check('MaxCut 精確最佳值（jobs）', abs(ns['cut'](ej, ns['exact'](nj, ej)) - js['jobs']['exact']) < 1e-9)
# 4. RQAOA end result (both deterministic, but parameter optimisers differ: warn only)
warn('RQAOA 結果（jobs）', abs(ns['cut'](ej, ns['rqaoa'](nj, ej)) - js['jobs']['rqaoa']) < 1e-9, f"py {ns['cut'](ej, ns['rqaoa'](nj, ej))} / js {js['jobs']['rqaoa']}")
npn, ep = ns['build_graph'](pd.read_csv(root / 'samples' / 'paint.csv'))
warn('RQAOA 結果（paint）', abs(ns['cut'](ep, ns['rqaoa'](npn, ep)) - js['paint']['rqaoa']) < 1e-9, f"py {ns['cut'](ep, ns['rqaoa'](npn, ep))} / js {js['paint']['rqaoa']}")
print('JS 與 Python 版本一致' if not fails else f'{fails} 項不一致'); sys.exit(1 if fails else 0)
