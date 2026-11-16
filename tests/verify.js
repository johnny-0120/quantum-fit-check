// Correctness checks for engine.js. Usage: node tests/verify.js
const E = require('../engine.js'); let fail = 0;
const check = (name, ok, info) => { console.log((ok ? 'PASS ' : 'FAIL ') + name + (info ? '  ' + info : '')); if (!ok) fail++; };
// 1. p=1 analytic <ZZ> (used by RQAOA) vs full statevector
{ const r = E.rng(42), n = 7, edges = []; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (r() < 0.5) edges.push([i, j, Math.round((r() * 4 - 1.5) * 10) / 10 || 1]);
  const S = E._sv.svSetup(n, edges); const adj = Array.from({ length: n }, () => new Map()); for (const [a, b, w] of edges) { adj[a].set(b, w / S.scale); adj[b].set(a, w / S.scale); }
  let err = 0; for (const [g, b] of [[0.3, 0.2], [0.9, 0.7], [1.7, 1.1]]) { const p = E._sv.svRun(S, [g], [b], null); for (let u = 0; u < n; u++) for (let v = u + 1; v < n; v++) { let zz = 0; for (let x = 0; x < S.N; x++) zz += p[x] * ((((x >> u) & 1) === ((x >> v) & 1)) ? 1 : -1); err = Math.max(err, Math.abs(E._corrPair(adj, u, v, g, b) - zz)); } }
  check('RQAOA p=1 公式 vs 量子態模擬', err < 1e-12, 'max err ' + err.toExponential(1)); }
// 2. IQP exact gradient vs finite differences
{ const r = E.rng(1), data = []; for (let i = 0; i < 200; i++) data.push(Math.floor(r() * 16)); const iq = E._trainIQP(4, data, 0, E.rng(1)); const N = 16, c = new Float64Array(N); data.forEach(x => c[x] += 1 / data.length);
  const loss = th => { const { ar, ai } = iq._forward(th); let L = 0; for (let x = 0; x < N; x++) if (c[x]) L -= c[x] * Math.log(ar[x] ** 2 + ai[x] ** 2); return L; };
  const th0 = iq.theta.map((_, k) => 0.3 * Math.sin(k + 1)); const { vr, vi, ar, ai } = iq._forward(th0); const br = new Float64Array(N), bi = new Float64Array(N);
  for (let x = 0; x < N; x++) { if (!c[x]) continue; const pp = ar[x] ** 2 + ai[x] ** 2; br[x] = c[x] * ar[x] / pp; bi[x] = -c[x] * ai[x] / pp; } E._fwht(br); E._fwht(bi); for (let s = 0; s < N; s++) { br[s] /= 4; bi[s] /= 4; }
  let ge = 0; for (let k = 0; k < th0.length; k++) { let gk = 0; for (let s = 0; s < N; s++) gk += iq._F[k][s] * (vr[s] * bi[s] + vi[s] * br[s]); gk *= 2; const h = 1e-5, tp = th0.slice(), tm = th0.slice(); tp[k] += h; tm[k] -= h; ge = Math.max(ge, Math.abs((loss(tp) - loss(tm)) / (2 * h) - gk)); }
  check('IQP 精確梯度 vs 數值微分', ge < 1e-6, 'max err ' + ge.toExponential(1)); }
// 3. QUBO → Ising conversion: cost from cut equals direct evaluation, on every assignment
{ const g = E.buildQubo(E.parseCSV(E.samples.schedule()), {}); const Q = g.qubo; let maxErr = 0, best = Infinity;
  const lin = new Map(), pairs = Q.origPairs; // recompute penalised cost directly
  for (let mask = 0; mask < (1 << Q.nv); mask++) {
    const z = new Int8Array(Q.nv + 1); for (let i = 0; i < Q.nv; i++) z[i] = (mask >> i) & 1 ? -1 : 1; z[Q.nv] = 1;
    let cut = 0; for (const [a, b, w] of g.edges) if (z[a] !== z[b]) cut += w; const viaCut = Q.const + Q.sumW - 2 * cut;
    let direct = 0; for (let i = 0; i < Q.nv; i++) if ((mask >> i) & 1) direct += Q.origLin[i] || 0; for (const [k, v] of pairs) { const [i, j] = k.split(',').map(Number); if (((mask >> i) & 1) && ((mask >> j) & 1)) direct += v; }
    for (const gr of Q.groups) { const s = gr.members.reduce((a, i) => a + ((mask >> i) & 1), 0); direct += gr.penalty * (s - gr.k) ** 2; }
    maxErr = Math.max(maxErr, Math.abs(viaCut - direct)); best = Math.min(best, direct); }
  const res = E.analyzeOpt(g, () => {}); const ex = res.methods.find(m => m.name.startsWith('精確'));
  check('QUBO 轉換：每種分配的成本都一致', maxErr < 1e-9, 'max err ' + maxErr.toExponential(1));
  check('QUBO 窮舉找到真正最低成本且滿足限制', Math.abs(ex.cost - best) < 1e-9 && ex.feasible, 'cost ' + ex.cost + ' / true ' + best); }
// 4. t distribution tail
check('t 分布尾機率', Math.abs(E._tUpper(2.0, 14) - 0.0326) < 1e-3 && Math.abs(E._tUpper(1, 1) - 0.25) < 1e-6);
console.log(fail ? fail + ' 項失敗' : '全部通過'); process.exit(fail ? 1 : 0);
