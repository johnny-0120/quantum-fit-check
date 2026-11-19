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
  check('QUBO 窮舉找到真正最低成本且滿足限制', Math.abs(ex.cost - best) < 1e-9 && ex.feasible, 'cost ' + ex.cost + ' / true ' + best);
  const q1 = res.methods.find(m => m.name === 'QAOA p=1');
  check('QUBO 期望成本換算一致', q1 && q1.expectedCost != null && Math.abs(q1.expectedCost - (Q.const + Q.sumW - 2 * q1.expected)) < 1e-9); }
// 4. t distribution tail
check('t 分布尾機率', Math.abs(E._tUpper(2.0, 14) - 0.0326) < 1e-3 && Math.abs(E._tUpper(1, 1) - 0.25) < 1e-6);
// 5. trainHE simultaneous parameter-shift gradient & nonlinear NLL learning
{ const target = [0.5, 0, 0, 0.5]; const dLdp = p => p.map((v, i) => target[i] ? -target[i] / Math.max(v, 1e-12) : 0);
  const ref = E._trainHE(2, 1, dLdp, 0, E.rng(7), 0.05).theta.slice(); const got = E._trainHE(2, 1, dLdp, 2, E.rng(7), 0.05).theta;
  const P = ref.length, m = new Float64Array(P), v = new Float64Array(P);
  for (let st = 0; st < 2; st++) {
    const gp = dLdp(E._heProbs(2, ref, 1)), g = new Float64Array(P);
    for (let k = 0; k < P; k++) { const o = ref[k]; ref[k] = o + Math.PI / 2; const pp = E._heProbs(2, ref, 1); ref[k] = o - Math.PI / 2; const pm = E._heProbs(2, ref, 1); ref[k] = o; let gk = 0; for (let x = 0; x < pp.length; x++) gk += gp[x] * (pp[x] - pm[x]); g[k] = gk / 2; }
    for (let k = 0; k < P; k++) { m[k] = 0.9 * m[k] + 0.1 * g[k]; v[k] = 0.999 * v[k] + 0.001 * g[k] * g[k]; ref[k] -= 0.05 * (m[k] / (1 - Math.pow(0.9, st + 1))) / (Math.sqrt(v[k] / (1 - Math.pow(0.999, st + 1))) + 1e-8); }
  }
  let diff = 0; for (let k = 0; k < P; k++) diff = Math.max(diff, Math.abs(got[k] - ref[k]));
  const fit = E._trainHE(2, 2, dLdp, 80, E.rng(1), 0.05); const nll = -target.reduce((s, t, i) => t ? s + t * Math.log(Math.max(fit.p[i], 1e-12)) : s, 0);
  check('trainHE 同步梯度與非線性 NLL 收斂', diff < 1e-12 && nll < 0.72, 'max diff ' + diff.toExponential(1) + ', NLL ' + nll.toFixed(4)); }
// 6. rqaoa handles >20 disconnected/isolated nodes safely without 1<<m overflow
{ const edges = []; for (let i = 0; i < 15; i++) for (let j = i + 1; j < 15; j++) if ((i + j) % 3 === 0) edges.push([i, j, 1]);
  const r40 = E.rqaoa(40, edges, 8);
  check('RQAOA 含孤立節點不溢位／不卡死', r40.z.length === 40 && r40.value > 0, 'cut ' + r40.value); }
// 7. parseCSV truncation flag, headerless string edge list, leading blank lines
{ const rows = ['a,b,c']; for (let i = 0; i < 8; i++) rows.push(i + ',' + (i + 1) + ',1');
  const pOver = E.parseCSV(rows.join('\n'), 5), pExact = E.parseCSV(rows.slice(0, 6).join('\n'), 5), pEdge = E.parseCSV('\n\nA,B,3\nB,C,1\n');
  check('parseCSV 截斷旗標與無表頭邊清單', pOver.truncated && pOver.rows.length === 5 && !pExact.truncated && pExact.rows.length === 5 && pEdge.rows.length === 2); }
// 8. missing values not treated as 0, small-sample check, graph/QUBO guards
{ const pCls = E.prepareCls(E.parseCSV('x,y\n10,0\n12,1\n,0\n14,1\n16,0\n18,1\n20,0\n22,1\n24,0\n26,1\n28,0\n30,1'), 1);
  let threwSmall = false; try { E.prepareCls(E.parseCSV('x,y\n1,0\n2,1\n3,0\n4,1'), 1); } catch (e) { threwSmall = true; }
  const genRows = ['a,b,c']; for (let i = 0; i < 30; i++) genRows.push((i % 2 === 0 ? '' : String(10 + i)) + ',' + (i % 3 ? '1' : '0') + ',' + (i % 5 ? '1' : '0'));
  const pGen = E.prepareGen(E.parseCSV(genRows.join('\n')), [0, 1, 2]);
  const gOrph = E.buildGraph(E.parseCSV('s,t,w\nA,B,1\nB,C,2\nC,D,1\nX,Y,0'));
  const qDedup = E.buildQubo(E.parseCSV('kind,a,b,value\nlinear,x,,1\nlinear,y,,2\nchoose,g1,1,\nmember,g1,x\nmember,g1,x\nmember,g1,y'), {});
  let threwNaN = false; try { E.buildQubo(E.parseCSV('kind,a,b,value\nlinear,x,,abc\nlinear,y,,1'), {}); } catch (e) { threwNaN = true; }
  check('缺值處理與邊界防呆（分類／分布／圖／QUBO）', Math.abs(pCls.X[2][0]) < 1e-9 && threwSmall && Number(pGen.bits[0].desc.replace(/.*中位數\s*/, '')) >= 20 && gOrph.n === 4 && qDedup.qubo.groups[0].members.length === 2 && threwNaN); }
console.log(fail ? fail + ' 項失敗' : '全部通過'); process.exit(fail ? 1 : 0);
