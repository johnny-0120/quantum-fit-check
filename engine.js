/* 量子適用性健檢 engine — pure JS, runs in browser (Web Worker) and Node.
 * Three checks, each comparing simulated quantum methods against strong classical baselines
 * on the user's own small sample:
 *   opt : graph / Ising optimization (MaxCut form). QAOA p=1/p=2, CVaR-QAOA, Warm-start QAOA (statevector, n<=14),
 *         RQAOA (exact p=1 analytic correlations, n<=200) vs exact / greedy local search / simulated annealing / tabu.
 *   cls : classification. Quantum kernel (ZZ feature map, statevector, <=8 qubits) vs linear / RBF kernel ridge / kNN,
 *         nested hold-out tuning, 5-fold CV, kernel concentration + geometric difference (Huang et al. 2021).
 *   gen : distribution learning on binary data. IQP Born machine (exact gradients) vs independent / Chow-Liu / smoothed histogram,
 *         held-out negative log-likelihood.
 */
(function (root) {
  'use strict';
  const E = {};
  const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

  // ---------- utilities ----------
  function rng(seed) {
    let a = (seed >>> 0) || 1;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  E.rng = rng;
  function gauss(r) { let u = 0, v = 0; while (u === 0) u = r(); v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function shuffle(a, r) { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }
  const mean = a => a.reduce((s, x) => s + x, 0) / (a.length || 1);
  const std = a => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) * (x - m), 0) / Math.max(1, a.length - 1)); };
  function median(a) { const b = a.slice().sort((x, y) => x - y); const n = b.length; return n ? (n % 2 ? b[(n - 1) / 2] : (b[n / 2 - 1] + b[n / 2]) / 2) : 0; }
  E.stats = { mean, std, median };

  function nelderMead(f, x0, opt) {
    const step = (opt && opt.step) || 0.3, iters = (opt && opt.iters) || 80;
    const n = x0.length;
    let pts = [x0.slice()];
    for (let i = 0; i < n; i++) { const p = x0.slice(); p[i] += step; pts.push(p); }
    let vals = pts.map(f);
    for (let it = 0; it < iters; it++) {
      const idx = vals.map((v, i) => i).sort((a, b) => vals[a] - vals[b]);
      pts = idx.map(i => pts[i]); vals = idx.map(i => vals[i]);
      const c = new Array(n).fill(0);
      for (let i = 0; i < n; i++) for (let k = 0; k < n; k++) c[k] += pts[i][k] / n;
      const w = pts[n];
      const xr = c.map((ci, k) => ci + (ci - w[k])); const fr = f(xr);
      if (fr < vals[0]) {
        const xe = c.map((ci, k) => ci + 2 * (ci - w[k])); const fe = f(xe);
        if (fe < fr) { pts[n] = xe; vals[n] = fe; } else { pts[n] = xr; vals[n] = fr; }
      } else if (fr < vals[n - 1]) { pts[n] = xr; vals[n] = fr; }
      else {
        const xc = c.map((ci, k) => ci + 0.5 * (w[k] - ci)); const fc = f(xc);
        if (fc < vals[n]) { pts[n] = xc; vals[n] = fc; }
        else { for (let i = 1; i <= n; i++) { pts[i] = pts[i].map((x, k) => pts[0][k] + 0.5 * (x - pts[0][k])); vals[i] = f(pts[i]); } }
      }
    }
    let bi = 0; for (let i = 1; i <= n; i++) if (vals[i] < vals[bi]) bi = i;
    return { x: pts[bi], f: vals[bi] };
  }
  E.nelderMead = nelderMead;

  // symmetric eigen-decomposition (cyclic Jacobi). A: array of Float64Array rows. returns {vals, vecs (columns)}
  function jacobiEigen(Ain) {
    const n = Ain.length; const A = Ain.map(r => Float64Array.from(r));
    const V = []; for (let i = 0; i < n; i++) { V.push(new Float64Array(n)); V[i][i] = 1; }
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
      if (off < 1e-20) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        const apq = A[p][q]; if (Math.abs(apq) < 1e-300) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
        for (let k = 0; k < n; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
        for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
      }
    }
    return { vals: A.map((r, i) => r[i]), vecs: V };
  }
  E.jacobiEigen = jacobiEigen;

  function cholSolve(K, Y, alpha) { // (K + alpha I) X = Y ; K n×n array rows, Y n×c
    const n = K.length, L = []; for (let i = 0; i < n; i++) L.push(new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = K[i][j] + (i === j ? alpha : 0);
        const Li = L[i], Lj = L[j];
        for (let k = 0; k < j; k++) s -= Li[k] * Lj[k];
        if (i === j) L[i][i] = Math.sqrt(Math.max(s, 1e-12)); else L[i][j] = s / L[j][j];
      }
    }
    const c = Y[0].length, X = Y.map(r => Float64Array.from(r));
    for (let col = 0; col < c; col++) {
      for (let i = 0; i < n; i++) { let s = X[i][col]; for (let k = 0; k < i; k++) s -= L[i][k] * X[k][col]; X[i][col] = s / L[i][i]; }
      for (let i = n - 1; i >= 0; i--) { let s = X[i][col]; for (let k = i + 1; k < n; k++) s -= L[k][i] * X[k][col]; X[i][col] = s / L[i][i]; }
    }
    return X;
  }

  // ---------- CSV ----------
  E.parseCSV = function (text, maxRows) {
    maxRows = maxRows || 20000;
    text = String(text).replace(/^﻿/, '').replace(/\r\n?/g, '\n');
    const first = text.split('\n', 1)[0] || '';
    const cand = [',', '\t', ';', '|'];
    let delim = ','; let best = -1;
    for (const d of cand) { const c = first.split(d).length; if (c > best) { best = c; delim = d; } }
    const rows = []; let row = [], cell = '', q = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (q) {
        if (ch === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else q = false; }
        else cell += ch;
      } else if (ch === '"') q = true;
      else if (ch === delim) { row.push(cell.trim()); cell = ''; }
      else if (ch === '\n') { row.push(cell.trim()); cell = ''; if (row.some(x => x !== '')) rows.push(row); row = []; if (rows.length > maxRows) break; }
      else cell += ch;
    }
    if (cell !== '' || row.length) { row.push(cell.trim()); if (row.some(x => x !== '')) rows.push(row); }
    if (!rows.length) return { header: [], rows: [] };
    const isNum = s => s !== '' && isFinite(Number(s));
    let header;
    const r0 = rows[0], r1 = rows[1] || [];
    const hasHeader = r0.some((s, i) => !isNum(s) && (r1[i] === undefined || isNum(r1[i]) || r1[i] !== s)) && !(r0.every(isNum));
    if (hasHeader) header = rows.shift().map((h, i) => h || ('欄位' + (i + 1)));
    else header = r0.map((_, i) => '欄位' + (i + 1));
    const w = header.length;
    const body = rows.filter(r => r.length >= Math.min(2, w)).map(r => { const o = r.slice(0, w); while (o.length < w) o.push(''); return o; });
    return { header, rows: body, truncated: rows.length > maxRows };
  };

  E.profileColumns = function (t) {
    return t.header.map((h, j) => {
      const vals = t.rows.map(r => r[j]);
      const nonEmpty = vals.filter(v => v !== '' && v != null && v.toLowerCase() !== 'na' && v.toLowerCase() !== 'null');
      const nums = nonEmpty.filter(v => isFinite(Number(v)));
      const uniq = new Set(nonEmpty);
      const numeric = nonEmpty.length > 0 && nums.length / nonEmpty.length > 0.95;
      return { name: h, index: j, numeric, unique: uniq.size, missing: vals.length - nonEmpty.length,
        binary: uniq.size === 2, isInt: numeric && nums.every(v => Number.isInteger(Number(v))) };
    });
  };

  E.suggestTask = function (t) {
    const prof = E.profileColumns(t); const n = t.rows.length, w = t.header.length;
    const names = t.header.map(h => h.toLowerCase());
    const srcI = names.findIndex(h => /^(source|src|from|u|node1|node_a|i|起點|來源)$/.test(h));
    const dstI = names.findIndex(h => /^(target|dst|to|v|node2|node_b|j|終點|目標節點)$/.test(h));
    if (names.includes('kind') && names.includes('a')) return { task: 'opt', mode: 'spec', why: '偵測到 kind／a／b／value 欄位，是 0/1 問題規格表（可含「剛好選 k 個」限制）。' };
    if (srcI >= 0 && dstI >= 0) return { task: 'opt', why: '偵測到「起點／終點」欄位，看起來是關係網路（邊清單）。' };
    if (w === n && prof.every(p => p.numeric) && n >= 3) return { task: 'opt', why: '資料是方陣，看起來是兩兩關係矩陣。' };
    if (w <= 3 && w >= 2 && n >= 3) {
      const ids = new Set(t.rows.map(r => r[0]).concat(t.rows.map(r => r[1])));
      if (ids.size < n * 1.5 && ids.size >= 3 && (w === 2 || prof[2].numeric)) return { task: 'opt', why: '前兩欄像節點代號、第三欄像關係強度，看起來是邊清單。' };
    }
    const lab = E.guessLabel(t);
    const lowCard = prof.filter(p => p.unique <= 6).length;
    if (lowCard >= Math.max(3, w * 0.8) && lab === -1) return { task: 'gen', why: '大部分欄位只有少數幾種值，適合看「資料分布能不能被量子生成模型學起來」。' };
    if (lab >= 0) return { task: 'cls', why: '偵測到可能的預測目標欄位「' + t.header[lab] + '」，看起來是分類問題。' };
    if (lowCard >= Math.max(3, w * 0.8)) return { task: 'gen', why: '大部分欄位只有少數幾種值，適合做分布學習。' };
    return { task: 'cls', why: '預設以最後一欄當預測目標，你可以在下方改。' };
  };
  E.guessLabel = function (t) {
    const prof = E.profileColumns(t);
    const names = t.header.map(h => h.toLowerCase());
    let i = names.findIndex(h => /(^|_)(label|class|target|y|is_bot|bot|fraud|churn|outcome|attack|類別|標籤|結果)($|_)/.test(h));
    if (i >= 0 && prof[i].unique <= 20) return i;
    const last = prof[prof.length - 1];
    if (last && last.unique >= 2 && last.unique <= 10 && t.rows.length > 20) return last.index;
    return -1;
  };

  // =====================================================================
  // OPTIMIZATION (Ising / MaxCut form): maximize cut = Σ w_ij (1 - z_i z_j)/2
  // =====================================================================
  E.buildGraph = function (t, opt) {
    opt = opt || {};
    const prof = E.profileColumns(t);
    let n, labels, emap = new Map(), mode;
    const w = t.header.length;
    if (opt.mode === 'matrix' || (opt.mode !== 'edges' && w === t.rows.length && prof.every(p => p.numeric))) {
      mode = 'matrix'; n = w; labels = t.header.slice();
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        const a = Number(t.rows[i][j]) || 0, b = Number(t.rows[j][i]) || 0; const v = (a + b) / 2;
        if (v !== 0) emap.set(i + ',' + j, v);
      }
    } else {
      mode = 'edges';
      const si = opt.src != null ? opt.src : 0, di = opt.dst != null ? opt.dst : 1;
      const wi = opt.weight != null ? opt.weight : (w >= 3 && prof[2].numeric ? 2 : -1);
      const id = new Map(); labels = [];
      const get = s => { if (!id.has(s)) { id.set(s, labels.length); labels.push(s); } return id.get(s); };
      for (const r of t.rows) {
        const a = r[si], b = r[di]; if (a === '' || b === '' || a == null || b == null || a === b) continue;
        let x = get(a), y = get(b); const val = wi >= 0 ? Number(r[wi]) : 1;
        if (!isFinite(val) || val === 0) continue;
        if (x > y) [x, y] = [y, x];
        const k = x + ',' + y; emap.set(k, (emap.get(k) || 0) + val);
      }
      n = labels.length;
    }
    const edges = [];
    for (const [k, v] of emap) { if (v === 0) continue; const [a, b] = k.split(',').map(Number); edges.push([a, b, v]); }
    return { n, labels, edges, mode };
  };

  // keep the connected sample manageable: BFS from highest-degree node
  E.sampleGraph = function (g, maxN) {
    if (g.n <= maxN) return { g, sampled: false };
    const deg = new Array(g.n).fill(0), adj = Array.from({ length: g.n }, () => []);
    for (const [a, b] of g.edges) { deg[a]++; deg[b]++; adj[a].push(b); adj[b].push(a); }
    let s = 0; for (let i = 1; i < g.n; i++) if (deg[i] > deg[s]) s = i;
    const keep = [s], seen = new Set([s]);
    for (let h = 0; h < keep.length && keep.length < maxN; h++) for (const v of adj[keep[h]]) { if (!seen.has(v) && keep.length < maxN) { seen.add(v); keep.push(v); } }
    for (let i = 0; i < g.n && keep.length < maxN; i++) if (!seen.has(i)) { seen.add(i); keep.push(i); }
    const re = new Map(keep.map((v, i) => [v, i]));
    const edges = g.edges.filter(([a, b]) => re.has(a) && re.has(b)).map(([a, b, w]) => [re.get(a), re.get(b), w]);
    return { g: { n: keep.length, labels: keep.map(i => g.labels[i]), edges, mode: g.mode }, sampled: true, originalN: g.n };
  };

  function adjList(n, edges) { const A = Array.from({ length: n }, () => []); for (const [a, b, w] of edges) { A[a].push([b, w]); A[b].push([a, w]); } return A; }
  function cutValue(edges, z) { let c = 0; for (const [a, b, w] of edges) if (z[a] !== z[b]) c += w; return c; }
  E.cutValue = cutValue;

  function localSearch(n, A, z) { // 1-flip ascent on cut
    let improved = true;
    while (improved) {
      improved = false;
      for (let i = 0; i < n; i++) {
        let d = 0; for (const [j, w] of A[i]) d += w * z[i] * z[j];
        if (d > 1e-12) { z[i] = -z[i]; improved = true; }
      }
    }
    return z;
  }
  function randomZ(n, r) { const z = new Int8Array(n); for (let i = 0; i < n; i++) z[i] = r() < 0.5 ? 1 : -1; return z; }

  function solveExact(n, A, edges) { // Gray code over 2^(n-1)
    const z = new Int8Array(n).fill(1); let cur = 0, best = 0, bestZ = z.slice();
    const total = 1 << (n - 1);
    for (let k = 1; k < total; k++) {
      let b = 0; while (((k >> b) & 1) === 0) b++;
      let d = 0; for (const [j, w] of A[b]) d += w * z[b] * z[j];
      z[b] = -z[b]; cur += d;
      if (cur > best + 1e-12) { best = cur; bestZ = z.slice(); }
    }
    return { z: bestZ, value: cutValue(edges, bestZ) };
  }
  function solveGreedyLS(n, A, edges, r, starts) {
    let best = -Infinity, bestZ = null;
    for (let s = 0; s < starts; s++) { const z = localSearch(n, A, randomZ(n, r)); const v = cutValue(edges, z); if (v > best) { best = v; bestZ = z; } }
    return { z: bestZ, value: best };
  }
  function solveSA(n, A, edges, r, sweeps) {
    let z = randomZ(n, r); let cur = cutValue(edges, z), best = cur, bestZ = z.slice();
    const mw = mean(edges.map(e => Math.abs(e[2]))) || 1;
    let T = 2 * mw; const Tend = 0.005 * mw, factor = Math.pow(Tend / T, 1 / sweeps);
    for (let s = 0; s < sweeps; s++) {
      for (let i = 0; i < n; i++) {
        let d = 0; for (const [j, w] of A[i]) d += w * z[i] * z[j];
        if (d >= 0 || r() < Math.exp(d / T)) { z[i] = -z[i]; cur += d; if (cur > best + 1e-12) { best = cur; bestZ = z.slice(); } }
      }
      T *= factor;
    }
    return { z: bestZ, value: cutValue(edges, bestZ) };
  }
  function solveTabu(n, A, edges, r, timeMs) {
    let z = localSearch(n, A, randomZ(n, r)); let cur = cutValue(edges, z), best = cur, bestZ = z.slice();
    const tabu = new Int32Array(n); const gain = new Float64Array(n);
    const recompute = () => { for (let i = 0; i < n; i++) { let d = 0; for (const [j, w] of A[i]) d += w * z[i] * z[j]; gain[i] = d; } };
    recompute();
    const t0 = now(); let it = 0, sinceBest = 0;
    while (now() - t0 < timeMs) {
      it++;
      let bi = -1, bg = -Infinity;
      for (let i = 0; i < n; i++) {
        const ok = tabu[i] <= it || cur + gain[i] > best + 1e-12;
        if (ok && gain[i] > bg + 1e-12) { bg = gain[i]; bi = i; }
      }
      if (bi < 0) { bi = Math.floor(r() * n); bg = gain[bi]; }
      z[bi] = -z[bi]; cur += bg; gain[bi] = -gain[bi];
      for (const [j, w] of A[bi]) gain[j] += 2 * w * z[bi] * z[j] * (1);
      // gain_j = Σ w z_j z_k ; z_bi flipped → term changes from -w z_j z_bi(old) to +... recompute exactly for safety periodically
      tabu[bi] = it + Math.max(3, Math.floor(n / 10)) + Math.floor(r() * 4);
      if (cur > best + 1e-12) { best = cur; bestZ = z.slice(); sinceBest = 0; } else sinceBest++;
      if (it % 200 === 0) { recompute(); cur = cutValue(edges, z); }
      if (sinceBest > 40 * n) { z = localSearch(n, A, randomZ(n, r)); cur = cutValue(edges, z); recompute(); sinceBest = 0; }
    }
    return { z: bestZ, value: cutValue(edges, bestZ), iters: it };
  }

  // ---- p=1 analytic correlations for C = Σ J_uv Z_u Z_v (no local fields), |ψ> = e^{-iβΣX} e^{-iγC} |+>^n ----
  // Ozaeta, van Dam, McMahon (2022) form; checked numerically against the statevector simulator in tests.
  function corrPair(adj, u, v, g, b) {
    const Au = adj[u], Av = adj[v]; const c2 = 2 * g;
    const Juv = Au.get(v) || 0;
    let pu = 1, pv = 1, P1 = 1, P2 = 1;
    for (const [w, a] of Au) { if (w === v) continue; pu *= Math.cos(c2 * a); const bb = Av.get(w) || 0; P1 *= Math.cos(c2 * (a + bb)); P2 *= Math.cos(c2 * (a - bb)); }
    for (const [w, bb] of Av) { if (w === u) continue; pv *= Math.cos(c2 * bb); if (!Au.has(w)) { const cc = Math.cos(c2 * bb); P1 *= cc; P2 *= cc; } }
    const s2b = Math.sin(2 * b);
    return 0.5 * Math.sin(4 * b) * Math.sin(c2 * Juv) * (pu + pv) - 0.5 * s2b * s2b * (P1 - P2);
  }
  E._corrPair = corrPair;
  function energyAnalytic(adj, elist, g, b) { let e = 0; for (const [u, v, J] of elist) e += J * corrPair(adj, u, v, g, b); return e; }
  function optimizeP1Analytic(adj, elist, start) {
    let best = start ? { x: start, f: energyAnalytic(adj, elist, start[0], start[1]) } : null;
    if (!start) {
      for (let i = 1; i <= 12; i++) for (let k = 1; k <= 8; k++) {
        const x = [i * Math.PI / 13, k * Math.PI / 18]; const f = energyAnalytic(adj, elist, x[0], x[1]);
        if (!best || f < best.f) best = { x, f };
      }
    }
    const r = nelderMead(p => energyAnalytic(adj, elist, p[0], p[1]), best.x, { step: start ? 0.08 : 0.15, iters: start ? 25 : 60 });
    return r.f < best.f ? r : best;
  }

  function rqaoa(n, edges, ncut, progress) {
    const scale = Math.max(...edges.map(e => Math.abs(e[2]))) || 1;
    const adj = Array.from({ length: n }, () => new Map());
    for (const [a, b, w] of edges) { adj[a].set(b, (adj[a].get(b) || 0) + w / scale); adj[b].set(a, (adj[b].get(a) || 0) + w / scale); }
    const active = new Set(Array.from({ length: n }, (_, i) => i));
    const stack = []; let params = null; const trace = [];
    while (active.size > ncut) {
      const elist = []; for (const u of active) for (const [v, J] of adj[u]) if (u < v && J !== 0) elist.push([u, v, J]);
      if (!elist.length) break;
      const opt = optimizeP1Analytic(adj, elist, params); params = opt.x;
      // candidate pairs: adjacent or sharing a neighbour (others have zero correlation)
      let bu = -1, bv = -1, bc = 0;
      const seen = new Set();
      const consider = (u, v) => { if (u === v) return; const a = Math.min(u, v), b = Math.max(u, v); const k = a * n + b; if (seen.has(k)) return; seen.add(k); const c = corrPair(adj, a, b, params[0], params[1]); if (Math.abs(c) > Math.abs(bc) + 1e-12) { bc = c; bu = a; bv = b; } };
      for (const u of active) { for (const [v] of adj[u]) { consider(u, v); for (const [w] of adj[v]) consider(u, w); } }
      if (bu < 0) break;
      const s = bc >= 0 ? 1 : -1;
      // eliminate the endpoint with fewer neighbours: z_elim = s * z_keep
      let elim = bu, keep = bv; if (adj[bu].size > adj[bv].size) { elim = bv; keep = bu; }
      for (const [w, J] of adj[elim]) {
        adj[w].delete(elim);
        if (w === keep) continue;
        const nv = (adj[keep].get(w) || 0) + s * J;
        if (Math.abs(nv) < 1e-12) { adj[keep].delete(w); adj[w].delete(keep); } else { adj[keep].set(w, nv); adj[w].set(keep, nv); }
      }
      adj[elim].clear(); active.delete(elim); stack.push([elim, keep, s]);
      trace.push({ remaining: active.size, corr: bc, relation: s > 0 ? '同組' : '異組' });
      if (progress && trace.length % 5 === 0) progress('RQAOA 縮減中：剩 ' + active.size + ' 個變數');
    }
    // brute-force remaining
    const rest = Array.from(active); const m = rest.length; const pos = new Map(rest.map((v, i) => [v, i]));
    let bestE = Infinity, bestMask = 0;
    for (let mask = 0; mask < (1 << m); mask++) {
      let e = 0; for (let i = 0; i < m; i++) { const zi = (mask >> i) & 1 ? -1 : 1; for (const [w, J] of adj[rest[i]]) { const j = pos.get(w); if (j > i) e += J * zi * (((mask >> j) & 1) ? -1 : 1); } }
      if (e < bestE) { bestE = e; bestMask = mask; }
    }
    const z = new Int8Array(n).fill(1);
    rest.forEach((v, i) => { z[v] = ((bestMask >> i) & 1) ? -1 : 1; });
    for (let k = stack.length - 1; k >= 0; k--) { const [e, kp, s] = stack[k]; z[e] = s * z[kp]; }
    return { z, value: cutValue(edges, z), steps: stack.length, trace };
  }
  E.rqaoa = rqaoa;

  // ---- statevector simulator (n <= 14) ----
  function svSetup(n, edges) {
    const N = 1 << n; const scale = Math.max(...edges.map(e => Math.abs(e[2]))) || 1;
    const C = new Float64Array(N), cut = new Float64Array(N);
    for (let x = 0; x < N; x++) { let c = 0, k = 0; for (const [a, b, w] of edges) { const za = (x >> a) & 1, zb = (x >> b) & 1; const zz = za === zb ? 1 : -1; c += (w / scale) * zz; if (za !== zb) k += w; } C[x] = c; cut[x] = k; }
    const order = Array.from({ length: N }, (_, i) => i).sort((a, b) => C[a] - C[b]);
    return { n, N, C, cut, order, scale };
  }
  function svRun(S, gammas, betas, warm) {
    const { n, N, C } = S; const re = new Float64Array(N), im = new Float64Array(N);
    if (warm) { for (let x = 0; x < N; x++) { let a = 1; for (let q = 0; q < n; q++) a *= ((x >> q) & 1) ? Math.sin(warm[q] / 2) : Math.cos(warm[q] / 2); re[x] = a; } }
    else re.fill(1 / Math.sqrt(N));
    for (let l = 0; l < gammas.length; l++) {
      const g = gammas[l], b = betas[l];
      for (let x = 0; x < N; x++) { const c = Math.cos(g * C[x]), s = -Math.sin(g * C[x]); const r0 = re[x], i0 = im[x]; re[x] = r0 * c - i0 * s; im[x] = r0 * s + i0 * c; }
      for (let q = 0; q < n; q++) {
        let U;
        if (warm) { // Ry(θ) Rz(-2β) Ry(-θ)  (Egger et al. 2021)
          const th = warm[q], c = Math.cos(th / 2), s = Math.sin(th / 2);
          const e0r = Math.cos(b), e0i = Math.sin(b), e1r = Math.cos(b), e1i = -Math.sin(b); // Rz(-2β)=diag(e^{iβ}, e^{-iβ})
          // M = Ry(θ) D Ry(-θ); Ry(θ)=[[c,-s],[s,c]], Ry(-θ)=[[c,s],[-s,c]]
          U = [
            [c * c * e0r + s * s * e1r, c * c * e0i + s * s * e1i, c * s * e0r - s * c * e1r, c * s * e0i - s * c * e1i],
            [s * c * e0r - c * s * e1r, s * c * e0i - c * s * e1i, s * s * e0r + c * c * e1r, s * s * e0i + c * c * e1i]
          ]; // rows: [m00r,m00i,m01r,m01i],[m10r,m10i,m11r,m11i]
        } else { const c = Math.cos(b), s = Math.sin(b); U = [[c, 0, 0, -s], [0, -s, c, 0]]; }
        const [m00r, m00i, m01r, m01i] = U[0], [m10r, m10i, m11r, m11i] = U[1];
        const bit = 1 << q;
        for (let x = 0; x < N; x++) {
          if (x & bit) continue; const y = x | bit;
          const ar = re[x], ai = im[x], br = re[y], bi = im[y];
          re[x] = m00r * ar - m00i * ai + m01r * br - m01i * bi; im[x] = m00r * ai + m00i * ar + m01r * bi + m01i * br;
          re[y] = m10r * ar - m10i * ai + m11r * br - m11i * bi; im[y] = m10r * ai + m10i * ar + m11r * bi + m11i * br;
        }
      }
    }
    const p = new Float64Array(N); for (let x = 0; x < N; x++) p[x] = re[x] * re[x] + im[x] * im[x];
    return p;
  }
  E._sv = { svSetup, svRun };
  function svExpect(S, p) { let e = 0; for (let x = 0; x < S.N; x++) e += p[x] * S.C[x]; return e; }
  function svCVaR(S, p, alpha) { let acc = 0, e = 0; for (const x of S.order) { const take = Math.min(p[x], alpha - acc); if (take <= 0) break; e += take * S.C[x]; acc += take; } return e / alpha; }
  function svSample(S, p, shots, r) {
    const cdf = new Float64Array(S.N); let s = 0; for (let x = 0; x < S.N; x++) { s += p[x]; cdf[x] = s; }
    let best = -Infinity, bestX = 0;
    for (let k = 0; k < shots; k++) { const u = r() * s; let lo = 0, hi = S.N - 1; while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; } if (S.cut[lo] > best) { best = S.cut[lo]; bestX = lo; } }
    return { best, bestX };
  }
  function svOptimize(S, p, objective, warm) {
    const f = x => { const g = x.slice(0, p), b = x.slice(p); const pr = svRun(S, g, b, warm); return objective(pr); };
    let best = null;
    for (let i = 1; i <= 10; i++) for (let k = 1; k <= 8; k++) {
      const x = []; for (let l = 0; l < p; l++) x.push(i * Math.PI / 11 * (l + 1) / p); for (let l = 0; l < p; l++) x.push(k * Math.PI / 9 * (p - l) / p);
      const v = f(x); if (!best || v < best.f) best = { x, f: v };
    }
    const r = nelderMead(f, best.x, { step: 0.12, iters: 60 + 30 * p });
    return r.f < best.f ? r : best;
  }

  // ---- Goemans–Williamson SDP relaxation via low-rank mixing method (Burer–Monteiro) + random hyperplane rounding ----
  function solveGW(n, A, edges, r) {
    const k = Math.min(12, Math.max(2, Math.ceil(Math.sqrt(2 * n))));
    const V = []; for (let i = 0; i < n; i++) { const v = new Float64Array(k); let s = 0; for (let d = 0; d < k; d++) { v[d] = gauss(r); s += v[d] * v[d]; } s = Math.sqrt(s); for (let d = 0; d < k; d++) v[d] /= s; V.push(v); }
    const g = new Float64Array(k);
    for (let it = 0; it < 400; it++) {
      let change = 0;
      for (let i = 0; i < n; i++) {
        g.fill(0); for (const [j, w] of A[i]) { const vj = V[j]; for (let d = 0; d < k; d++) g[d] += w * vj[d]; }
        let s = 0; for (let d = 0; d < k; d++) s += g[d] * g[d]; s = Math.sqrt(s); if (s < 1e-12) continue;
        const vi = V[i]; for (let d = 0; d < k; d++) { const nv = -g[d] / s; change += Math.abs(nv - vi[d]); vi[d] = nv; }
      }
      if (change < 1e-7 * n) break;
    }
    let sdp = 0; for (const [a, b, w] of edges) { let dot = 0; for (let d = 0; d < k; d++) dot += V[a][d] * V[b][d]; sdp += w * (1 - dot) / 2; }
    let best = -Infinity, bz = null; const h = new Float64Array(k);
    for (let t = 0; t < 64; t++) {
      for (let d = 0; d < k; d++) h[d] = gauss(r);
      const z = new Int8Array(n); for (let i = 0; i < n; i++) { let s = 0; for (let d = 0; d < k; d++) s += V[i][d] * h[d]; z[i] = s >= 0 ? 1 : -1; }
      localSearch(n, A, z); const v = cutValue(edges, z); if (v > best) { best = v; bz = z; }
    }
    return { z: bz, value: best, sdp };
  }

  // ---- simulated quantum annealing: Trotterised H(s) = -(1-s) ΣX + s C, linear schedule, total time T ----
  function svAnneal(S, T) {
    const K = Math.min(120, Math.max(10, Math.round(4 * T))); const dt = T / K; const g = [], b = [];
    for (let t = 0; t < K; t++) { const s = (t + 0.5) / K; g.push(dt * s); b.push(-dt * (1 - s)); }
    return svRun(S, g, b, null);
  }

  // ---- real-amplitude hardware-efficient circuit: RY layer, then L × (CZ chain + RY layer) ----
  function heProbs(n, theta, L) {
    const N = 1 << n; const a = new Float64Array(N); a[0] = 1; let k = 0;
    const ry = (q, t) => { const c = Math.cos(t / 2), s = Math.sin(t / 2), bit = 1 << q; for (let x = 0; x < N; x++) { if (x & bit) continue; const y = x | bit; const a0 = a[x], a1 = a[y]; a[x] = c * a0 - s * a1; a[y] = s * a0 + c * a1; } };
    const cz = (p, q) => { const m = (1 << p) | (1 << q); for (let x = 0; x < N; x++) if ((x & m) === m) a[x] = -a[x]; };
    for (let q = 0; q < n; q++) ry(q, theta[k++]);
    for (let l = 0; l < L; l++) { for (let q = 0; q + 1 < n; q++) cz(q, q + 1); for (let q = 0; q < n; q++) ry(q, theta[k++]); }
    const p = new Float64Array(N); for (let x = 0; x < N; x++) p[x] = a[x] * a[x]; return p;
  }
  // Adam + exact parameter-shift gradient of any loss L(p): dL/dθ = Σ_x ∂L/∂p_x · (p_x(θ+π/2) − p_x(θ−π/2))/2
  function trainHE(n, L, dLdp, steps, r, lr) {
    const P = n * (L + 1); const th = []; for (let k = 0; k < P; k++) th.push((k < n ? Math.PI / 2 : 0) + 0.1 * gauss(r));
    const m = new Float64Array(P), v = new Float64Array(P);
    for (let st = 0; st < steps; st++) {
      const p0 = heProbs(n, th, L); const gp = dLdp(p0);
      for (let k = 0; k < P; k++) {
        const o = th[k]; th[k] = o + Math.PI / 2; const pp = heProbs(n, th, L); th[k] = o - Math.PI / 2; const pm = heProbs(n, th, L); th[k] = o;
        let gk = 0; for (let x = 0; x < pp.length; x++) gk += gp[x] * (pp[x] - pm[x]); gk /= 2;
        m[k] = 0.9 * m[k] + 0.1 * gk; v[k] = 0.999 * v[k] + 0.001 * gk * gk;
        th[k] -= (lr || 0.05) * (m[k] / (1 - Math.pow(0.9, st + 1))) / (Math.sqrt(v[k] / (1 - Math.pow(0.999, st + 1))) + 1e-8);
      }
    }
    return { theta: th, p: heProbs(n, th, L), params: P };
  }
  E._heProbs = heProbs;

  // ---- Grover adaptive search (Dürr–Høyer / Gilliam et al. 2021), exact amplitude-amplification statistics ----
  function groverAdaptive(S, r) {
    const N = S.N; const ord = Array.from({ length: N }, (_, i) => i).sort((a, b) => S.cut[a] - S.cut[b]); const sorted = Float64Array.from(ord, i => S.cut[i]);
    const above = y => { let lo = 0, hi = N; while (lo < hi) { const mid = (lo + hi) >> 1; if (sorted[mid] > y + 1e-9) hi = mid; else lo = mid + 1; } return N - lo; };
    let yi = Math.floor(r() * N); let y = S.cut[yi]; let calls = 0, rounds = 1, m = 1, since = 0; const cap = Math.sqrt(N);
    while (since < 4 * cap + 8) {
      const M = above(y); const k = Math.floor(r() * m); calls += k; rounds++;
      const ps = M ? Math.pow(Math.sin((2 * k + 1) * Math.asin(Math.sqrt(M / N))), 2) : 0;
      if (r() < ps) { const j = N - M + Math.floor(r() * M); y = sorted[j]; yi = ord[j]; m = 1; since = 0; }
      else { since += k + 1; m = Math.min(m * 1.2, cap); }
    }
    return { value: y, calls, rounds, index: yi };
  }

  // ---- recommendation when choosing a classical method ----
  function recommendOpt(c) {
    const fmt = ms => ms < 1 ? '不到 1 毫秒' : ms.toFixed(0) + ' 毫秒';
    let primary, why; const alts = [], tools = [];
    if (c.bip && c.negFrac === 0) { primary = '二分圖兩色著色（廣度優先搜尋）'; why = '你的關係網路是二分圖、強度都是正的：相鄰節點塗不同顏色就是最佳解，計算時間和關係數成正比，不需要任何搜尋。'; tools.push('Python networkx：nx.bipartite.color(G)'); }
    else if (c.isForest) { primary = '樹狀動態規劃'; why = '關係網路沒有迴圈，可以從葉子往根一層一層算出保證最佳解。'; tools.push('自行實作約 30 行；或用整數規劃求解器'); }
    else if (c.n <= 25) { primary = '精確求解（窮舉或整數規劃求解器）'; why = '只有 ' + c.n + ' 個變數，「' + c.top.name + '」' + fmt(c.top.ms) + '就找到最好答案；用精確方法還能保證它是最佳。'; tools.push('Google OR-Tools CP-SAT（免費、開源）', 'Gurobi／CPLEX（商用，有免費學術授權）'); alts.push('資料變大後改用禁忌搜尋或模擬退火'); }
    else { primary = c.top.name; why = '在你的資料上，它最快拿到所有方法中最好的答案（' + fmt(c.top.ms) + '）。'; alts.push('禁忌搜尋、模擬退火：多個起點平行跑，通常幾秒內就很接近最佳', '需要品質保證：整數規劃求解器設時間上限，它會回報離最佳解還差多少（MIP gap）'); if (c.negFrac === 0) alts.push('GW 半定規劃：強度都是正的時候，保證至少拿到最佳的 87.8%'); tools.push('Google OR-Tools、Gurobi（整數規劃）', 'cvxpy（GW 半定規劃）', 'dwave-samplers（模擬退火、禁忌搜尋）', 'MQLib（MaxCut 啟發式演算法合集）'); }
    return { primary, why, alts, tools, steps: ['用整份資料（不是小樣本）跑推薦的方法，記錄目標值和時間。', '和你們目前的做法比較，換算成實際指標，例如換色次數、衝突數或成本。'] };
  }

  // ---- QUBO / constraint-table input → Ising with one reference spin (fields become couplings to it) ----
  // minimise f(x) = Σ_i a_i x_i + Σ_{i<j} b_ij x_i x_j + c0,  x ∈ {0,1}
  function quboToGraph(nv, names, lin, pairs, c0, meta) {
    // x_i = (1 - s_i)/2, s_i = z_i z_ref  →  f = const + Σ h_i s_i + Σ J_ij s_i s_j
    const h = new Float64Array(nv); const J = new Map(); let cst = c0;
    for (let i = 0; i < nv; i++) { const a = lin[i] || 0; cst += a / 2; h[i] -= a / 2; }
    for (const [k, b] of pairs) { const [i, j] = k.split(',').map(Number); cst += b / 4; h[i] -= b / 4; h[j] -= b / 4; J.set(k, (J.get(k) || 0) + b / 4); }
    const edges = []; for (const [k, v] of J) { if (Math.abs(v) < 1e-12) continue; const [i, j] = k.split(',').map(Number); edges.push([i, j, v]); }
    for (let i = 0; i < nv; i++) if (Math.abs(h[i]) > 1e-12) edges.push([i, nv, h[i]]);
    const sumW = edges.reduce((s, e) => s + e[2], 0);
    return { n: nv + 1, labels: names.concat(['（參考位元）']), edges, mode: meta.mode, qubo: Object.assign({ nv, names, const: cst, sumW }, meta) };
  }
  E.buildQubo = function (t, opt) {
    opt = opt || {}; const hdr = t.header.map(h => h.toLowerCase().trim());
    if (opt.mode === 'spec' || hdr.includes('kind')) {
      const ki = hdr.indexOf('kind'), ai = hdr.indexOf('a'), bi = hdr.indexOf('b'), vi = hdr.indexOf('value'), pi = hdr.indexOf('penalty');
      if (ki < 0 || ai < 0) throw new Error('規格表需要 kind、a、b、value 欄位。');
      const id = new Map(), names = []; const vid = s => { s = String(s).trim(); if (!id.has(s)) { id.set(s, names.length); names.push(s); } return id.get(s); };
      const lin = [], pairs = new Map(), origLin = [], origPairs = new Map(), groups = new Map();
      const addPair = (M, i, j, v) => { if (i === j) return false; const k = Math.min(i, j) + ',' + Math.max(i, j); M.set(k, (M.get(k) || 0) + v); return true; };
      for (const r of t.rows) {
        const kind = (r[ki] || '').toLowerCase().trim(); const a = r[ai], b = bi >= 0 ? r[bi] : '', v = vi >= 0 ? Number(r[vi]) : 0;
        if (kind === 'linear' || kind === 'cost') { const i = vid(a); lin[i] = (lin[i] || 0) + v; origLin[i] = (origLin[i] || 0) + v; }
        else if (kind === 'pair' || kind === 'conflict') { const i = vid(a), j = vid(b); if (i === j) { lin[i] = (lin[i] || 0) + v; origLin[i] = (origLin[i] || 0) + v; } else { addPair(pairs, i, j, v); addPair(origPairs, i, j, v); } }
        else if (kind === 'member') { if (!groups.has(a)) groups.set(a, { name: a, k: 1, members: [], penalty: null }); groups.get(a).members.push(vid(b)); }
        else if (kind === 'choose' || kind === 'exactly') { if (!groups.has(a)) groups.set(a, { name: a, k: 1, members: [], penalty: null }); const g = groups.get(a); g.k = isFinite(Number(b)) && b !== '' ? Number(b) : 1; if (pi >= 0 && r[pi] !== '' && isFinite(Number(r[pi]))) g.penalty = Number(r[pi]); }
      }
      const nv = names.length; if (nv < 2) throw new Error('規格表至少要有 2 個變數。');
      // default penalty: twice the largest objective change a single variable can cause
      let sens = 0; for (let i = 0; i < nv; i++) { let s = Math.abs(origLin[i] || 0); for (const [k, v] of origPairs) { const [x, y] = k.split(',').map(Number); if (x === i || y === i) s += Math.abs(v); } sens = Math.max(sens, s); }
      const Pdef = Math.max(1, 2 * sens); let c0 = 0; const glist = [];
      for (const g of groups.values()) {
        const P = g.penalty != null ? g.penalty : Pdef; const k = g.k; glist.push({ name: g.name, k, members: g.members, penalty: P });
        for (const i of g.members) lin[i] = (lin[i] || 0) + P * (1 - 2 * k);
        for (let x = 0; x < g.members.length; x++) for (let y = x + 1; y < g.members.length; y++) addPair(pairs, g.members[x], g.members[y], 2 * P);
        c0 += P * k * k;
      }
      return quboToGraph(nv, names, lin, pairs, c0, { mode: 'spec', groups: glist, origLin, origPairs: Array.from(origPairs.entries()) });
    }
    // square QUBO matrix
    const nv = t.header.length; if (t.rows.length !== nv) throw new Error('QUBO 矩陣必須是方陣（列數 = 欄數）。');
    const lin = [], pairs = new Map();
    for (let i = 0; i < nv; i++) { lin[i] = Number(t.rows[i][i]) || 0; for (let j = i + 1; j < nv; j++) { const v = (Number(t.rows[i][j]) || 0) + (Number(t.rows[j][i]) || 0); if (v) pairs.set(i + ',' + j, v); } }
    return quboToGraph(nv, t.header.slice(), lin, pairs, 0, { mode: 'qubo', groups: [], origLin: lin.slice(), origPairs: Array.from(pairs.entries()) });
  };
  function decodeQubo(q, z) {
    const zr = z[q.nv]; const x = new Uint8Array(q.nv); for (let i = 0; i < q.nv; i++) x[i] = z[i] * zr < 0 ? 1 : 0;
    let obj = 0; for (let i = 0; i < q.nv; i++) if (x[i]) obj += q.origLin[i] || 0;
    for (const [k, v] of q.origPairs) { const [i, j] = k.split(',').map(Number); if (x[i] && x[j]) obj += v; }
    const viol = q.groups.filter(g => g.members.reduce((s, i) => s + x[i], 0) !== g.k).map(g => g.name);
    return { x, obj, viol, chosen: Array.from(x).map((v, i) => v ? q.names[i] : null).filter(Boolean) };
  }

  E.analyzeOpt = function (gIn, progress, opts) {
    opts = opts || {}; progress = progress || (() => {});
    const seed = opts.seed || 7; const SEEDS = opts.seeds || 3;
    const smp = gIn.qubo ? { g: gIn, sampled: false } : E.sampleGraph(gIn, opts.maxN || 200); const g = smp.g; const { n, edges } = g;
    if (gIn.qubo && n > 201) throw new Error('0/1 問題最多 200 個變數。');
    if (n < 3 || !edges.length) throw new Error('關係網路至少需要 3 個節點和 1 條關係。');
    const A = adjList(n, edges);
    const m = edges.length; const avgDeg = 2 * m / n;
    const absW = edges.map(e => Math.abs(e[2])); const wRatio = Math.max(...absW) / (median(absW) || 1);
    const negFrac = edges.filter(e => e[2] < 0).length / m;
    let bip = true; const col = new Int8Array(n);
    for (let s = 0; s < n && bip; s++) { if (col[s]) continue; col[s] = 1; const qu = [s]; while (qu.length && bip) { const u = qu.shift(); for (const [v] of A[u]) { if (!col[v]) { col[v] = -col[u]; qu.push(v); } else if (col[v] === col[u]) { bip = false; break; } } } }
    let comps = 0; { const seen = new Uint8Array(n); for (let s = 0; s < n; s++) { if (seen[s]) continue; comps++; const st = [s]; seen[s] = 1; while (st.length) { const u = st.pop(); for (const [v] of A[u]) if (!seen[v]) { seen[v] = 1; st.push(v); } } } }
    const isForest = m === n - comps;
    const methods = [];
    // run a method SEEDS times with independent random streams; keep every value, the best answer and the mean time
    const multi = (name, kind, fn, extra) => {
      const vals = [], times = []; let bestZ = null, bv = -Infinity, last = null;
      for (let k = 0; k < SEEDS; k++) { const r = rng(seed * 1000 + 17 * k + 1); const t0 = now(); const o = fn(r, k); times.push(now() - t0); vals.push(o.value); last = o; if (o.value > bv) { bv = o.value; bestZ = o.z; } }
      const mm = Object.assign({ name, kind, value: bv, values: vals, mean: mean(vals), ms: mean(times), z: bestZ }, extra || {}, typeof extra === 'function' ? {} : {});
      if (last && last.note) mm.note = last.note; if (last && last.expected != null) { mm.expected = last.expected; mm.pOpt = last.pOpt; }
      methods.push(mm); return mm;
    };
    const once = (name, kind, fn, extra) => { const t0 = now(); const o = fn(rng(seed * 1000 + 999)); const mm = Object.assign({ name, kind, value: o.value, values: [o.value], mean: o.value, ms: now() - t0, z: o.z, deterministic: true }, extra || {}); if (o.note) mm.note = o.note; methods.push(mm); return mm; };

    progress('古典方法（每種獨立跑 ' + SEEDS + ' 次）', 8);
    let exact = null;
    if (n <= 20) exact = once('精確窮舉（保證最佳）', 'classical', () => solveExact(n, A, edges), { note: '列出所有 ' + (2 ** (n - 1)).toLocaleString() + ' 種分法' });
    const greedyZ = [];
    multi('貪婪區域搜尋（1 次）', 'classical', (r) => { const o = solveGreedyLS(n, A, edges, r, 1); greedyZ.push(o.z); return o; }, { note: '最便宜的古典做法' });
    multi('區域搜尋（32 個起點）', 'classical', r => solveGreedyLS(n, A, edges, r, 32));
    multi('模擬退火', 'classical', r => solveSA(n, A, edges, r, Math.max(300, Math.min(3000, 60000 / n))));
    multi('禁忌搜尋', 'classical', r => solveTabu(n, A, edges, r, Math.min(600, 50 + 4 * n)), { note: '每次 ' + Math.min(600, 50 + 4 * n) + ' 毫秒' });
    progress('古典方法：GW 半定規劃', 22);
    let gwRelax = null; multi('GW 半定規劃＋隨機超平面', 'classical', r => { const o = solveGW(n, A, edges, r); gwRelax = o.sdp; return o; }, { note: 'Goemans–Williamson 近似演算法' });

    progress('量子方法：RQAOA（p=1 精確模擬）', 32);
    const rq = once('RQAOA（遞迴 QAOA）', 'quantum', () => { const o = rqaoa(n, edges, Math.min(8, n - 1), progress); o.note = '縮減 ' + o.steps + ' 步後窮舉剩下的變數（確定性演算法）'; return o; }, { qubits: n, twoQ: 2 * m });

    let sv = null;
    if (n <= 14) {
      const S = svSetup(n, edges); const shots = 1024;
      const bestKnown = Math.max(...methods.map(x => x.value));
      const pOpt = pr => { let s = 0; for (let x = 0; x < S.N; x++) if (S.cut[x] >= bestKnown - 1e-9) s += pr[x]; return s; };
      const zOf = x => { const z = new Int8Array(n); for (let i = 0; i < n; i++) z[i] = ((x >> i) & 1) ? -1 : 1; return z; };
      const sampled = (name, pr, note, twoQ, t0) => { // optimisation is deterministic; the 1024-shot measurement is repeated SEEDS times
        let ec = 0; for (let x = 0; x < S.N; x++) ec += pr[x] * S.cut[x]; const po = pOpt(pr); const prep = now() - t0;
        const mm = multi(name, 'quantum', r => { const s = svSample(S, pr, shots, r); return { value: s.best, z: zOf(s.bestX) }; }, { note: note + '；每次取 ' + shots + ' 次量測中最好的', qubits: n, twoQ, expected: ec, pOpt: po });
        mm.ms += prep; return mm;
      };
      const runQ = (name, p, obj, warm, note) => { const t0 = now(); const o = svOptimize(S, p, obj, warm); sampled(name, svRun(S, o.x.slice(0, p), o.x.slice(p), warm), note, 2 * m * p, t0); };
      progress('量子方法：QAOA（量子態完整模擬，' + n + ' 量子位元）', 45);
      runQ('QAOA p=1', 1, pr => svExpect(S, pr), null, '標準 QAOA');
      progress('量子方法：QAOA p=2', 55);
      runQ('QAOA p=2', 2, pr => svExpect(S, pr), null, '兩層線路');
      progress('量子方法：CVaR-QAOA', 65);
      runQ('CVaR-QAOA（α=0.1）', 1, pr => svCVaR(S, pr, 0.1), null, '只看最好的 10% 量測結果來訓練');
      progress('量子方法：Warm-start QAOA', 72);
      const eps = 0.25; const warm = Array.from(greedyZ[0], zi => 2 * Math.asin(Math.sqrt(zi > 0 ? eps : 1 - eps)));
      runQ('Warm-start QAOA', 1, pr => svExpect(S, pr), warm, '從 1 次貪婪搜尋的答案出發（ε=0.25）');
      progress('量子方法：量子退火（模擬）', 80);
      { const t0 = now(); let bestP = null, bestE = Infinity, bestT = 0;
        for (const T of [2, 5, 10, 25]) { const pr = svAnneal(S, T); const e = svExpect(S, pr); if (e < bestE) { bestE = e; bestP = pr; bestT = T; } }
        sampled('量子退火（模擬）', bestP, '線性排程，總演化時間 ' + bestT + '（從 2／5／10／25 中選平均最好的）', 2 * m * Math.min(120, Math.max(10, 4 * bestT)), t0); }
      if (n <= 12) {
        progress('量子方法：VQE（硬體高效線路，' + SEEDS + ' 組隨機初值）', 86);
        multi('VQE（硬體高效線路）', 'quantum', r => { const he = trainHE(n, 2, () => S.C, 40, r, 0.08); const s = svSample(S, he.p, shots, r); let ec = 0; for (let x = 0; x < S.N; x++) ec += he.p[x] * S.cut[x]; return { value: s.best, z: zOf(s.bestX), expected: ec, pOpt: pOpt(he.p) }; }, { note: '2 層 RY＋CZ，每組初值訓練 40 步', qubits: n, twoQ: 2 * (n - 1) });
      }
      progress('量子方法：Grover 適應性搜尋', 93);
      multi('Grover 適應性搜尋', 'quantum', r => { const gs = groverAdaptive(S, r); return { value: gs.value, z: zOf(gs.index), note: '平均約 ' + gs.calls.toLocaleString() + ' 次 Grover 疊代；每次疊代都要在線路裡算一次完整目標函數，實際成本遠高於一次古典評估' }; }, { qubits: n });
      sv = { n, shots };
    }
    progress('整理報告', 96);

    // QUBO decoding: cost (lower is better), constraint check
    const Q = g.qubo; const costOf = cut => Q.const + Q.sumW - 2 * cut;
    if (Q) for (const mm of methods) { const d = decodeQubo(Q, mm.z); mm.cost = costOf(mm.value); mm.costMean = costOf(mm.mean); mm.feasible = !d.viol.length; mm.viol = d.viol; mm.obj = d.obj; mm.chosen = d.chosen.slice(0, 40); }
    for (const mm of methods) delete mm.z;
    const cls = methods.filter(x => x.kind === 'classical'), qs = methods.filter(x => x.kind === 'quantum');
    const bestC = Math.max(...cls.map(x => x.value)), bestQ = Math.max(...qs.map(x => x.value));
    const bestQm = qs.filter(x => x.value === bestQ).sort((a, b) => b.mean - a.mean)[0];
    const fastestC = cls.filter(x => x.value >= bestC - 1e-9).sort((a, b) => a.ms - b.ms)[0];
    // how reliably does each side hit its best? (fraction of runs reaching the overall best)
    const best = Math.max(bestC, bestQ); const hit = mm => mm.values.filter(v => v >= best - 1e-9).length / mm.values.length;
    const diags = []; const nDec = Q ? Q.nv : n;
    diags.push({ name: '問題規模', value: nDec + ' 個決策變數' + (Q ? '（另加 1 個參考位元）' : '') + '，' + m + ' 條關係', status: nDec <= 20 ? 'bad' : nDec <= 60 ? 'warn' : 'ok', explain: nDec <= 20 ? '20 個變數以內，古典電腦直接全部列舉就能拿到保證最佳解，量子沒有發揮空間。' : nDec <= 60 ? '中等規模，古典啟發式通常已能找到最佳或接近最佳。' : '規模較大，古典方法開始無法保證最佳。' });
    if (!Q) diags.push({ name: '結構是否本來就簡單', value: (bip && negFrac === 0) ? '二分圖（全部關係都可同時滿足）' : isForest ? '樹狀（沒有迴圈）' : '有迴圈、有挫折', status: (bip && negFrac === 0) || isForest ? 'bad' : 'ok', explain: (bip && negFrac === 0) ? '這種結構用古典方法一步就能算出最佳解，不需要任何量子方法。' : isForest ? '沒有迴圈的結構可用動態規劃精確求解。' : '有互相矛盾的關係，問題本身不簡單。' });
    diags.push({ name: '關係強弱差距', value: '最強／一般 ≈ ' + wRatio.toFixed(1) + ' 倍', status: wRatio > 100 ? 'bad' : wRatio > 10 ? 'warn' : 'ok', explain: (Q && Q.groups.length ? '限制條件轉成懲罰項後，懲罰係數通常遠大於原本的成本，這會拉大強弱差距。' : '') + (wRatio > 10 ? '少數超強關係會主導結果；RQAOA 先綁定的會是這些關係，後面容易綁錯。（門檻 10 倍來自隊上 Wi-Fi 與噴漆實驗，屬經驗規則）' : '關係強度相近，是 RQAOA 擅長的情況。') });
    diags.push({ name: '平均連線數', value: avgDeg.toFixed(1), status: avgDeg > 10 ? 'warn' : 'ok', explain: avgDeg > 10 ? '連線很密，淺層 QAOA 看到的局部資訊有限，量子線路也會很深。' : '連線稀疏，淺層線路較可行。' });
    diags.push({ name: '目前硬體可行性', value: n + ' 量子位元、約 ' + (2 * m) + ' 個雙位元閘（每層，未計晶片接線所需的交換閘）', status: 2 * m > 2000 ? 'bad' : 2 * m > 300 ? 'warn' : 'ok', explain: 2 * m > 300 ? '雙位元閘數量多，在目前有雜訊的硬體上結果會明顯變差。' : '線路規模在目前硬體可實驗範圍內；實際晶片若量子位元不相鄰，還要再加交換閘。' });
    diags.push({ name: '重複執行的穩定度', value: '古典最穩定的方法 ' + Math.round(Math.max(...cls.map(hit)) * 100) + '% 次數拿到最佳；量子最穩定的方法 ' + Math.round(Math.max(...qs.map(hit)) * 100) + '%', status: Math.max(...qs.map(hit)) < Math.max(...cls.map(hit)) ? 'warn' : 'ok', explain: '每種隨機方法都用 ' + SEEDS + ' 組不同的亂數各跑一次。量子方法的參數最佳化是確定性的，重複的是 1024 次量測。' });
    if (Q && Q.groups.length) { const feasC = cls.filter(x => x.feasible).length, feasQ = qs.filter(x => x.feasible).length; diags.push({ name: '限制條件是否滿足', value: '古典 ' + feasC + '/' + cls.length + ' 種方法、量子 ' + feasQ + '/' + qs.length + ' 種方法的最好答案滿足全部 ' + Q.groups.length + ' 條限制', status: feasQ < qs.length ? 'warn' : 'ok', explain: '限制是用懲罰項加進目標函數的，方法不保證一定滿足；整數規劃求解器可以直接處理限制，不需要懲罰項。' }); }

    let level, title; const reasons = [];
    const gap = bestC === 0 ? 0 : (bestC - bestQ) / Math.abs(bestC);
    const fmtV = v => Q ? '成本 ' + (+costOf(v).toFixed(3)) : (+v.toFixed(3));
    if (!Q && ((bip && negFrac === 0) || isForest)) { level = 'red'; title = '不建議用量子：這個結構古典方法可以直接算出最佳解'; }
    else if (bestQ < bestC - 1e-9) {
      level = 'red'; title = '不建議用量子：古典方法找到更好的答案';
      reasons.push('最好的量子方法（' + bestQm.name + '）' + fmtV(bestQ) + '，最好的古典方法 ' + fmtV(bestC) + '，差 ' + Math.abs(gap * 100).toFixed(1) + '%（雙方各跑 ' + SEEDS + ' 次取最好）。');
    } else if (nDec <= 20) {
      level = 'red'; title = '不建議用量子：規模太小，古典窮舉就有保證最佳解';
      reasons.push('量子方法有找到同樣好的答案，但古典' + (exact ? '窮舉只要 ' + exact.ms.toFixed(0) + ' 毫秒就能保證最佳' : '方法很快就找到') + '。');
    } else if (bestQ > bestC + 1e-9) {
      level = 'green'; title = '值得深入測試：量子方法找到比古典更好的答案';
      reasons.push(bestQm.name + ' 找到 ' + fmtV(bestQ) + '，古典最好 ' + fmtV(bestC) + '。請先把古典方法的時間預算加大 10 倍再比一次，確認不是古典搜尋時間不夠。');
    } else {
      level = 'yellow'; title = '可以做研究探索：量子方法追平最強古典方法';
      reasons.push('量子方法和最強古典方法找到一樣好的答案（' + fmtV(bestQ) + '）。追平不代表有優勢，因為這裡的量子模擬本身就跑在古典電腦上。');
    }
    reasons.push('最快拿到最好答案的古典方法是「' + fastestC.name + '」，每次 ' + fastestC.ms.toFixed(0) + ' 毫秒。');
    if (wRatio > 10) reasons.push('關係強弱差距 ' + wRatio.toFixed(0) + ' 倍，量子方法容易被少數強關係帶偏。');
    if (sv) reasons.push('說明：量子態完整模擬只能做到 14 個變數，而這個規模古典窮舉一定能解，所以小規模時這個檢查只能判斷「不需要量子」，不能證明量子有優勢。');
    const rec = recommendOpt({ n: nDec, bip: !Q && bip, negFrac, isForest: !Q && isForest, top: fastestC });
    if (Q && Q.groups.length) rec.alts.unshift('有限制條件的 0/1 問題，最直接的做法是整數規劃：把限制直接寫成條件，不用懲罰項，也保證答案合法（OR-Tools CP-SAT）');
    return {
      task: 'opt', n, m, sampled: smp.sampled, originalN: smp.originalN, seeds: SEEDS, qubo: Q ? { nv: Q.nv, groups: Q.groups.map(g2 => ({ name: g2.name, k: g2.k, size: g2.members.length, penalty: g2.penalty })), mode: Q.mode } : null,
      best, bestClassical: bestC, bestQuantum: bestQ, methods, diagnostics: diags, sv, gwRelax,
      recommend: rec, verdict: { level, title, reasons }, rqTrace: rq ? [] : []
    };
  };


  // =====================================================================
  // CLASSIFICATION
  // =====================================================================
  E.prepareCls = function (t, labelIdx, opts) {
    opts = opts || {}; const maxN = opts.maxN || 240; const r = rng(opts.seed || 11);
    const prof = E.profileColumns(t);
    const feats = []; // {name, kind:'num'|'cat', idx, levels}
    prof.forEach(p => {
      if (p.index === labelIdx) return;
      if (p.numeric && p.unique > 1) feats.push({ name: p.name, kind: 'num', idx: p.index });
      else if (!p.numeric && p.unique >= 2 && p.unique <= 10) {
        const cnt = new Map(); t.rows.forEach(rw => cnt.set(rw[p.index], (cnt.get(rw[p.index]) || 0) + 1));
        const levels = Array.from(cnt.keys()).sort((a, b) => cnt.get(b) - cnt.get(a)).slice(0, 8);
        feats.push({ name: p.name, kind: 'cat', idx: p.index, levels });
      }
    });
    if (!feats.length) throw new Error('找不到可用的特徵欄位（需要數值欄，或種類 10 種以內的文字欄）。');
    let rows = t.rows.filter(rw => rw[labelIdx] !== '' && rw[labelIdx] != null);
    const classes = Array.from(new Set(rows.map(rw => rw[labelIdx]))).sort();
    if (classes.length < 2) throw new Error('預測目標欄只有一種值，無法做分類。');
    if (classes.length > 6) throw new Error('預測目標欄有 ' + classes.length + ' 種值；請選種類 6 種以內的欄位。');
    // stratified subsample
    const by = new Map(classes.map(c => [c, []])); rows.forEach(rw => by.get(rw[labelIdx]).push(rw));
    let sampled = false;
    if (rows.length > maxN) {
      sampled = true; const out = [];
      for (const c of classes) { const arr = shuffle(by.get(c).slice(), r); const k = Math.max(5, Math.round(maxN * arr.length / rows.length)); out.push(...arr.slice(0, k)); }
      rows = shuffle(out, r).slice(0, maxN + classes.length * 5);
    }
    const means = feats.map(f => f.kind === 'num' ? mean(rows.map(rw => Number(rw[f.idx])).filter(v => isFinite(v) && rw_ok(v))) : 0);
    function rw_ok() { return true; }
    const names = [];
    feats.forEach(f => { if (f.kind === 'num') names.push(f.name); else f.levels.forEach(l => names.push(f.name + '=' + l)); });
    const X = rows.map(rw => { const v = []; feats.forEach((f, fi) => { if (f.kind === 'num') { const x = Number(rw[f.idx]); v.push(rw[f.idx] !== '' && isFinite(x) ? x : means[fi]); } else f.levels.forEach(l => v.push(rw[f.idx] === l ? 1 : 0)); }); return v; });
    const y = rows.map(rw => classes.indexOf(rw[labelIdx]));
    // standardize
    const d = names.length;
    for (let j = 0; j < d; j++) { const col = X.map(x => x[j]); const m = mean(col), s = std(col) || 1; X.forEach(x => { x[j] = (x[j] - m) / s; }); }
    return { X, y, classes, names, sampled, totalRows: t.rows.length, labelName: t.header[labelIdx] };
  };

  function pca(X, q) {
    const n = X.length, d = X[0].length;
    if (d <= q) return { Z: X.map(x => x.slice()), explained: 1, q: d };
    const Cm = []; for (let i = 0; i < d; i++) Cm.push(new Float64Array(d));
    for (const x of X) for (let i = 0; i < d; i++) for (let j = i; j < d; j++) Cm[i][j] += x[i] * x[j] / n;
    for (let i = 0; i < d; i++) for (let j = 0; j < i; j++) Cm[i][j] = Cm[j][i];
    const { vals, vecs } = jacobiEigen(Cm);
    const idx = vals.map((v, i) => i).sort((a, b) => vals[b] - vals[a]).slice(0, q);
    const tot = vals.reduce((s, v) => s + Math.max(v, 0), 0);
    const Z = X.map(x => idx.map(k => { let s = 0; for (let i = 0; i < d; i++) s += x[i] * vecs[i][k]; return s; }));
    for (let j = 0; j < q; j++) { const col = Z.map(z => z[j]); const m = mean(col), s = std(col) || 1; Z.forEach(z => { z[j] = (z[j] - m) / s; }); }
    return { Z, explained: idx.reduce((s, k) => s + Math.max(vals[k], 0), 0) / (tot || 1), q };
  }

  // ZZ feature map (Havlíček et al. 2019), 2 repetitions: |φ(x)> = U(x) H U(x) H |0>
  function zzState(x, lam) {
    const q = x.length, N = 1 << q; const a = x.map(v => lam * v);
    const ph = new Float64Array(N);
    for (let s = 0; s < N; s++) { let p = 0; for (let i = 0; i < q; i++) { const zi = ((s >> i) & 1) ? -1 : 1; p += a[i] * zi; for (let j = i + 1; j < q; j++) { const zj = ((s >> j) & 1) ? -1 : 1; p += (Math.PI - a[i]) * (Math.PI - a[j]) * zi * zj * 0.5 * lam; } } ph[s] = p; }
    let re = new Float64Array(N).fill(1 / Math.sqrt(N)), im = new Float64Array(N);
    for (let rep = 0; rep < 2; rep++) {
      for (let s = 0; s < N; s++) { const c = Math.cos(ph[s]), sn = Math.sin(ph[s]); const r0 = re[s], i0 = im[s]; re[s] = r0 * c - i0 * sn; im[s] = r0 * sn + i0 * c; }
      if (rep === 0) { fwht(re); fwht(im); const k = 1 / Math.sqrt(N); for (let s = 0; s < N; s++) { re[s] *= k; im[s] *= k; } }
    }
    return { re, im };
  }
  function fwht(a) { const N = a.length; for (let h = 1; h < N; h <<= 1) for (let i = 0; i < N; i += h << 1) for (let j = i; j < i + h; j++) { const x = a[j], y = a[j + h]; a[j] = x + y; a[j + h] = x - y; } }
  E._fwht = fwht; E._zzState = (x, l) => zzState(x, l);
  function quantumKernel(Z, lam) {
    const st = Z.map(z => zzState(z, lam)); const n = Z.length; const K = [];
    for (let i = 0; i < n; i++) K.push(new Float64Array(n));
    for (let i = 0; i < n; i++) { K[i][i] = 1; for (let j = i + 1; j < n; j++) { const A = st[i], B = st[j]; let r = 0, m = 0; for (let s = 0; s < A.re.length; s++) { r += A.re[s] * B.re[s] + A.im[s] * B.im[s]; m += A.re[s] * B.im[s] - A.im[s] * B.re[s]; } K[i][j] = K[j][i] = r * r + m * m; } }
    return K;
  }
  function rbfKernel(X, g) { const n = X.length, K = []; for (let i = 0; i < n; i++) K.push(new Float64Array(n)); for (let i = 0; i < n; i++) { K[i][i] = 1; for (let j = i + 1; j < n; j++) { let d = 0; for (let k = 0; k < X[i].length; k++) { const t = X[i][k] - X[j][k]; d += t * t; } K[i][j] = K[j][i] = Math.exp(-g * d); } } return K; }
  function linKernel(X) { const n = X.length, K = []; for (let i = 0; i < n; i++) K.push(new Float64Array(n)); for (let i = 0; i < n; i++) for (let j = i; j < n; j++) { let d = 1; for (let k = 0; k < X[i].length; k++) d += X[i][k] * X[j][k]; K[i][j] = K[j][i] = d; } return K; }
  function sqDist(X) { const n = X.length, D = []; for (let i = 0; i < n; i++) D.push(new Float64Array(n)); for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { let d = 0; for (let k = 0; k < X[i].length; k++) { const t = X[i][k] - X[j][k]; d += t * t; } D[i][j] = D[j][i] = d; } return D; }

  function krrFitPredict(K, y, nc, tr, te, alpha) {
    const Ktr = tr.map(i => tr.map(j => K[i][j]));
    const Y = tr.map(i => { const v = new Array(nc).fill(-1); v[y[i]] = 1; return v; });
    const A = cholSolve(Ktr, Y, alpha);
    return te.map(i => { const s = new Array(nc).fill(0); tr.forEach((j, a) => { const k = K[i][j]; for (let c = 0; c < nc; c++) s[c] += k * A[a][c]; }); let b = 0; for (let c = 1; c < nc; c++) if (s[c] > s[b]) b = c; return b; });
  }
  function knnPredict(D, y, nc, tr, te, k) {
    return te.map(i => { const nb = tr.map(j => [D[i][j], y[j]]).sort((a, b) => a[0] - b[0]).slice(0, k); const v = new Array(nc).fill(0); nb.forEach(([, c]) => v[c]++); let b = 0; for (let c = 1; c < nc; c++) if (v[c] > v[b]) b = c; return b; });
  }
  const acc = (pred, y, te) => pred.reduce((s, p, i) => s + (p === y[te[i]] ? 1 : 0), 0) / te.length;
  function balancedAcc(pred, y, te, nc) { const hit = new Array(nc).fill(0), tot = new Array(nc).fill(0); te.forEach((i, k) => { tot[y[i]]++; if (pred[k] === y[i]) hit[y[i]]++; }); const vals = hit.map((h, c) => tot[c] ? h / tot[c] : null).filter(v => v != null); return mean(vals); }

  function stratFolds(y, k, r) { const folds = Array.from({ length: k }, () => []); const by = new Map(); y.forEach((c, i) => { if (!by.has(c)) by.set(c, []); by.get(c).push(i); }); let f = 0; for (const arr of by.values()) { shuffle(arr, r); for (const i of arr) { folds[f % k].push(i); f++; } } return folds; }

  // ---- classical ML: logistic regression, random forest, small neural network ----
  function softmaxRow(z) { const mx = Math.max(...z); const e = z.map(v => Math.exp(v - mx)); const s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); }
  function lrFP(X, y, nc, tr, te, lam) {
    const d = X[0].length; const W = Array.from({ length: nc }, () => new Float64Array(d + 1));
    for (let it = 0; it < 300; it++) {
      const G = Array.from({ length: nc }, () => new Float64Array(d + 1));
      for (const i of tr) { const x = X[i]; const z = W.map(w => { let s = w[d]; for (let j = 0; j < d; j++) s += w[j] * x[j]; return s; }); const p = softmaxRow(z); for (let c = 0; c < nc; c++) { const g = p[c] - (y[i] === c ? 1 : 0); for (let j = 0; j < d; j++) G[c][j] += g * x[j]; G[c][d] += g; } }
      for (let c = 0; c < nc; c++) for (let j = 0; j <= d; j++) W[c][j] -= 0.5 * (G[c][j] / tr.length + (j < d ? lam * W[c][j] : 0));
    }
    return te.map(i => { const x = X[i]; let b = 0, bv = -Infinity; W.forEach((w, c) => { let s = w[d]; for (let j = 0; j < d; j++) s += w[j] * x[j]; if (s > bv) { bv = s; b = c; } }); return b; });
  }
  function buildTree(X, y, nc, idx, depth, r, mtry) {
    const cnt = new Array(nc).fill(0); idx.forEach(i => cnt[y[i]]++);
    const leaf = () => ({ p: cnt.map(c => c / idx.length) });
    if (depth >= 8 || idx.length < 4 || cnt.some(c => c === idx.length)) return leaf();
    const d = X[0].length; const feats = shuffle(Array.from({ length: d }, (_, j) => j), r).slice(0, mtry);
    const gini = c => { const t = c.reduce((a, b) => a + b, 0); return t ? 1 - c.reduce((s, v) => s + (v / t) * (v / t), 0) : 0; };
    let best = null;
    for (const j of feats) {
      const ord = idx.slice().sort((a, b) => X[a][j] - X[b][j]); const L = new Array(nc).fill(0), R = cnt.slice();
      for (let k = 0; k < ord.length - 1; k++) {
        L[y[ord[k]]]++; R[y[ord[k]]]--;
        if (X[ord[k]][j] === X[ord[k + 1]][j] || k < 1) continue;
        const sc = (k + 1) * gini(L) + (ord.length - k - 1) * gini(R);
        if (!best || sc < best.sc) best = { sc, j, t: (X[ord[k]][j] + X[ord[k + 1]][j]) / 2 };
      }
    }
    if (!best) return leaf();
    const li = idx.filter(i => X[i][best.j] <= best.t), ri = idx.filter(i => X[i][best.j] > best.t);
    if (!li.length || !ri.length) return leaf();
    return { j: best.j, t: best.t, l: buildTree(X, y, nc, li, depth + 1, r, mtry), r: buildTree(X, y, nc, ri, depth + 1, r, mtry) };
  }
  function rfFP(X, y, nc, tr, te, r) {
    const mtry = Math.max(1, Math.round(Math.sqrt(X[0].length))); const trees = [];
    for (let t = 0; t < 40; t++) { const boot = tr.map(() => tr[Math.floor(r() * tr.length)]); trees.push(buildTree(X, y, nc, boot, 0, r, mtry)); }
    return te.map(i => { const p = new Array(nc).fill(0); for (let node of trees) { while (!node.p) node = X[i][node.j] <= node.t ? node.l : node.r; node.p.forEach((v, c) => p[c] += v); } let b = 0; for (let c = 1; c < nc; c++) if (p[c] > p[b]) b = c; return b; });
  }
  function mlpFP(X, y, nc, tr, te, lam, r) {
    const d = X[0].length, H = 24; const W1 = Array.from({ length: H }, () => Float64Array.from({ length: d + 1 }, () => gauss(r) / Math.sqrt(d))), W2 = Array.from({ length: nc }, () => Float64Array.from({ length: H + 1 }, () => gauss(r) / Math.sqrt(H)));
    const params = [...W1, ...W2]; const M = params.map(p => new Float64Array(p.length)), V = params.map(p => new Float64Array(p.length));
    const fwd = x => { const h = W1.map(w => { let s = w[d]; for (let j = 0; j < d; j++) s += w[j] * x[j]; return Math.tanh(s); }); const o = W2.map(w => { let s = w[H]; for (let j = 0; j < H; j++) s += w[j] * h[j]; return s; }); return { h, p: softmaxRow(o) }; };
    for (let ep = 0; ep < 250; ep++) {
      const G = params.map(p => new Float64Array(p.length));
      for (const i of tr) { const x = X[i]; const { h, p } = fwd(x); const dh = new Float64Array(H);
        for (let c = 0; c < nc; c++) { const g = p[c] - (y[i] === c ? 1 : 0); const Gc = G[H + c]; for (let j = 0; j < H; j++) { Gc[j] += g * h[j]; dh[j] += g * W2[c][j]; } Gc[H] += g; }
        for (let k = 0; k < H; k++) { const g = dh[k] * (1 - h[k] * h[k]); const Gk = G[k]; for (let j = 0; j < d; j++) Gk[j] += g * x[j]; Gk[d] += g; } }
      params.forEach((p, a) => { for (let j = 0; j < p.length; j++) { const g = G[a][j] / tr.length + lam * p[j]; M[a][j] = 0.9 * M[a][j] + 0.1 * g; V[a][j] = 0.999 * V[a][j] + 0.001 * g * g; p[j] -= 0.02 * (M[a][j] / (1 - Math.pow(0.9, ep + 1))) / (Math.sqrt(V[a][j] / (1 - Math.pow(0.999, ep + 1))) + 1e-8); } });
    }
    return te.map(i => { const { p } = fwd(X[i]); let b = 0; for (let c = 1; c < nc; c++) if (p[c] > p[b]) b = c; return b; });
  }

  // ---- quantum ML: projected quantum kernel features (Huang et al. 2021) and a variational quantum classifier ----
  function projectedFeatures(Z, lam) {
    return Z.map(z => { const { re, im } = zzState(z, lam); const q = z.length, N = re.length; const f = [];
      for (let k = 0; k < q; k++) { const bit = 1 << k; let ex = 0, ey = 0, ez = 0;
        for (let s = 0; s < N; s++) { const pz = re[s] * re[s] + im[s] * im[s]; ez += (s & bit) ? -pz : pz; if (!(s & bit)) { const t = s | bit; ex += 2 * (re[s] * re[t] + im[s] * im[t]); ey += 2 * (re[s] * im[t] - im[s] * re[t]); } }
        f.push(ex, ey, ez); }
      return f; });
  }
  function vqcApply(st, th, q, L) { // RY layer + L × (CZ chain + RY layer) on a complex state copy
    const N = st.re.length; const re = Float64Array.from(st.re), im = Float64Array.from(st.im); let k = 0;
    const ry = (qq, t) => { const c = Math.cos(t / 2), s = Math.sin(t / 2), bit = 1 << qq; for (let x = 0; x < N; x++) { if (x & bit) continue; const y2 = x | bit; const r0 = re[x], i0 = im[x], r1 = re[y2], i1 = im[y2]; re[x] = c * r0 - s * r1; im[x] = c * i0 - s * i1; re[y2] = s * r0 + c * r1; im[y2] = s * i0 + c * i1; } };
    for (let a = 0; a < q; a++) ry(a, th[k++]);
    for (let l = 0; l < L; l++) { for (let a = 0; a + 1 < q; a++) { const m = (1 << a) | (1 << (a + 1)); for (let x = 0; x < N; x++) if ((x & m) === m) { re[x] = -re[x]; im[x] = -im[x]; } } for (let a = 0; a < q; a++) ry(a, th[k++]); }
    return { re, im };
  }
  function vqcExp(st, th, q, L, nc) { const { re, im } = vqcApply(st, th, q, L); const out = new Array(nc).fill(0); for (let x = 0; x < re.length; x++) { const p = re[x] * re[x] + im[x] * im[x]; for (let c = 0; c < nc; c++) out[c] += ((x >> c) & 1) ? -p : p; } return out; }
  // logits_c = w_c <Z_c> + b_c ; circuit gradient by exact parameter shift, mini-batch Adam
  function vqcFP(states, y, nc, q, tr, te, r) {
    const L = 2, P = q * (L + 1); const th = Array.from({ length: P }, () => 0.3 * gauss(r)); const w = new Array(nc).fill(2), b = new Array(nc).fill(0);
    const all = th.length + 2 * nc; const M = new Float64Array(all), V = new Float64Array(all); let t = 0;
    for (let step = 0; step < 150; step++) {
      const batch = shuffle(tr.slice(), r).slice(0, 24); const g = new Float64Array(all);
      for (const i of batch) {
        const E0 = vqcExp(states[i], th, q, L, nc); const p = softmaxRow(E0.map((e, c) => w[c] * e + b[c]));
        const dz = p.map((v, c) => v - (y[i] === c ? 1 : 0));
        for (let c = 0; c < nc; c++) { g[P + c] += dz[c] * E0[c]; g[P + nc + c] += dz[c]; }
        for (let k = 0; k < P; k++) { const o = th[k]; th[k] = o + Math.PI / 2; const Ep = vqcExp(states[i], th, q, L, nc); th[k] = o - Math.PI / 2; const Em = vqcExp(states[i], th, q, L, nc); th[k] = o; let s2 = 0; for (let c = 0; c < nc; c++) s2 += dz[c] * w[c] * (Ep[c] - Em[c]) / 2; g[k] += s2; }
      }
      t++; for (let k = 0; k < all; k++) { const gk = g[k] / batch.length; M[k] = 0.9 * M[k] + 0.1 * gk; V[k] = 0.999 * V[k] + 0.001 * gk * gk; const d = 0.08 * (M[k] / (1 - Math.pow(0.9, t))) / (Math.sqrt(V[k] / (1 - Math.pow(0.999, t))) + 1e-8); if (k < P) th[k] -= d; else if (k < P + nc) w[k - P] -= d; else b[k - P - nc] -= d; }
    }
    return te.map(i => { const E0 = vqcExp(states[i], th, q, L, nc); let bi = 0, bv = -Infinity; for (let c = 0; c < nc; c++) { const z = w[c] * E0[c] + b[c]; if (z > bv) { bv = z; bi = c; } } return bi; });
  }

  function recommendCls(res, classCounts) {
    const simple = ['邏輯回歸', '線性', 'k 近鄰', 'RBF', '隨機森林', '神經網路'];
    const info = { '邏輯回歸': ['邏輯回歸', 'scikit-learn LogisticRegression', '最容易解釋：每個特徵的係數就是它的影響方向和大小。'], '線性': ['邏輯回歸或線性 SVM', 'scikit-learn LogisticRegression／LinearSVC', '資料本身大致可以用一條線（超平面）分開。'], 'k 近鄰': ['k 近鄰', 'scikit-learn KNeighborsClassifier', '相似的紀錄通常結果一樣；資料要先標準化。'], 'RBF': ['RBF 核支持向量機', 'scikit-learn SVC(kernel="rbf")', '有非線性的邊界，但資料量不大時很穩定。'], '隨機森林': ['隨機森林或梯度提升樹', 'scikit-learn RandomForestClassifier；XGBoost、LightGBM', '能處理數值和類別混合的特徵，也能看出哪些特徵最重要。'], '神經網路': ['神經網路', 'scikit-learn MLPClassifier；PyTorch', '資料量大、關係複雜時有優勢，但需要較多資料和調參。'] };
    const cls = res.filter(m => m.kind === 'classical').sort((a, b) => b.mean - a.mean); const top = cls[0];
    const tol = Math.max(top.sd / Math.sqrt(5), 0.005);
    const rank = m => { const i = simple.findIndex(s => m.name.startsWith(s)); return i < 0 ? 99 : i; };
    const near = cls.filter(m => top.mean - m.mean <= tol).sort((a, b) => rank(a) - rank(b));
    const pick = near[0]; const key = simple.find(s => pick.name.startsWith(s)) || '線性'; const [nm, tool, note] = info[key];
    const why = (pick === top ? '在你的資料上分數最高（' + (top.mean * 100).toFixed(1) + '%）。' : '分數和第一名（' + top.name + '）差距在誤差內，而且更簡單、更好維護。') + note;
    const alts = cls.filter(m => m !== pick).slice(0, 2).map(m => m.name + '：' + (m.mean * 100).toFixed(1) + '%');
    const steps = ['用整份資料重新訓練，保留一段最新的資料當測試，看準確率會不會掉。'];
    const mn = Math.min(...classCounts), mx = Math.max(...classCounts);
    if (mn / mx < 0.3) steps.push('類別很不平衡：訓練時設 class_weight="balanced"，評估看各類別的召回率，不要只看準確率。');
    steps.push('如果要上線，定期用新資料重新驗證，資料分布變了模型就要重訓。');
    return { primary: nm, why, alts, tools: [tool], steps };
  }

  // model = {name, kind, grid:[{K or D, param} | {fp, label}], type:'krr'|'knn'|'fp'}
  function cvModel(model, y, nc, folds, r) {
    const scores = [], chosen = [];
    const fitPred = (cand, tr, te) => model.type === 'krr' ? krrFitPredict(cand.M, y, nc, tr, te, cand.alpha) : model.type === 'knn' ? knnPredict(cand.M, y, nc, tr, te, cand.k) : cand.fp(tr, te);
    for (let f = 0; f < folds.length; f++) {
      const te = folds[f], tr = folds.filter((_, g) => g !== f).flat();
      // inner hold-out for tuning
      const yTr = tr.map(i => y[i]); const inner = stratFolds(yTr, 4, r); const iva = inner[0].map(i => tr[i]); const itr = inner.slice(1).flat().map(i => tr[i]);
      let best = null;
      if (model.grid.length === 1) best = { cand: model.grid[0] };
      else for (const cand of model.grid) {
        const s = balancedAcc(fitPred(cand, itr, iva), y, iva, nc);
        if (!best || s > best.s + 1e-9) best = { s, cand };
      }
      const c = best.cand;
      const pred = fitPred(c, tr, te);
      scores.push(balancedAcc(pred, y, te, nc)); chosen.push(c.label);
    }
    return { scores, chosen };
  }

  function geometricDifference(KC, KQ, lamReg) {
    const n = KC.length; const tr = M => M.reduce((s, r, i) => s + r[i], 0);
    const sc = n / tr(KC), sq = n / tr(KQ);
    const C = KC.map(r => Float64Array.from(r, v => v * sc)), Q = KQ.map(r => Float64Array.from(r, v => v * sq));
    const eq = jacobiEigen(Q); const ec = jacobiEigen(C);
    const sqrtQ = []; for (let i = 0; i < n; i++) sqrtQ.push(new Float64Array(n));
    for (let k = 0; k < n; k++) { const s = Math.sqrt(Math.max(eq.vals[k], 0)); if (s === 0) continue; for (let i = 0; i < n; i++) { const vi = eq.vecs[i][k] * s; for (let j = 0; j < n; j++) sqrtQ[i][j] += vi * eq.vecs[j][k]; } }
    const Cinv = []; for (let i = 0; i < n; i++) Cinv.push(new Float64Array(n));
    for (let k = 0; k < n; k++) { const s = 1 / (Math.max(ec.vals[k], 0) + lamReg); for (let i = 0; i < n; i++) { const vi = ec.vecs[i][k] * s; for (let j = 0; j < n; j++) Cinv[i][j] += vi * ec.vecs[j][k]; } }
    const mul = (A, B) => { const R = []; for (let i = 0; i < n; i++) { const row = new Float64Array(n); for (let k = 0; k < n; k++) { const a = A[i][k]; if (a === 0) continue; const Bk = B[k]; for (let j = 0; j < n; j++) row[j] += a * Bk[j]; } R.push(row); } return R; };
    const M = mul(mul(sqrtQ, Cinv), sqrtQ);
    // spectral norm via power iteration
    let v = new Float64Array(n).fill(1 / Math.sqrt(n)), lam = 0;
    for (let it = 0; it < 200; it++) { const w = new Float64Array(n); for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += M[i][j] * v[j]; w[i] = s; } const nr = Math.hypot(...w); if (!nr) break; lam = nr; for (let i = 0; i < n; i++) v[i] = w[i] / nr; }
    return Math.sqrt(lam);
  }

  function centeredAlignment(K, y, idx) {
    const n = idx.length; const rk = new Float64Array(n), ry = new Float64Array(n); let gk = 0, gy = 0;
    for (let a = 0; a < n; a++) { const Ka = K[idx[a]]; for (let b = 0; b < n; b++) { rk[a] += Ka[idx[b]]; ry[a] += y[idx[a]] === y[idx[b]] ? 1 : 0; } rk[a] /= n; ry[a] /= n; gk += rk[a] / n; gy += ry[a] / n; }
    let kk = 0, yy = 0, ky = 0;
    for (let a = 0; a < n; a++) { const Ka = K[idx[a]]; for (let b = 0; b < n; b++) { const kc = Ka[idx[b]] - rk[a] - rk[b] + gk, yc = (y[idx[a]] === y[idx[b]] ? 1 : 0) - ry[a] - ry[b] + gy; kk += kc * kc; yy += yc * yc; ky += kc * yc; } }
    return ky / Math.sqrt(kk * yy || 1);
  }

  E.analyzeCls = function (P, progress, opts) {
    opts = opts || {}; progress = progress || (() => {}); const r = rng(opts.seed || 23);
    const { X, y, classes } = P; const n = X.length, nc = classes.length;
    const qMax = opts.qubits || 6; const pc = pca(X, qMax); const Z = pc.Z; const q = pc.q;
    const REPS = opts.repeats || 3; const foldSets = Array.from({ length: REPS }, () => stratFolds(y, 5, r));
    const alphas = [0.01, 0.1, 1];
    const models = [];
    progress('準備古典模型', 10);
    const Dfull = sqDist(X), Dz = sqDist(Z);
    const mkKRR = (name, kind, Ks) => ({ name, kind, type: 'krr', grid: Ks.flatMap(k => alphas.map(a => ({ M: k.K, alpha: a, label: k.label + ', α=' + a }))) });
    const rbfFrom = (D, d) => [0.03, 0.1, 0.3, 1, 3].map(s => { const g = s / d; const K = D.map(row => Float64Array.from(row, v => Math.exp(-g * v))); return { K, label: 'γ=' + g.toPrecision(2), g }; });
    const linK = X2 => { const K = linKernel(X2); return [{ K, label: '線性' }]; };
    models.push(mkKRR('線性模型（全部特徵）', 'classical', linK(X)));
    const rbfFull = rbfFrom(Dfull, X[0].length);
    models.push(mkKRR('RBF 核函數（全部特徵）', 'classical', rbfFull));
    models.push({ name: 'k 近鄰（全部特徵）', kind: 'classical', type: 'knn', grid: [3, 5, 9, 15].map(k => ({ M: Dfull, k, label: 'k=' + k })) });
    const rbfZ = rbfFrom(Dz, q);
    models.push(mkKRR('RBF 核函數（同樣 ' + q + ' 維輸入）', 'classical', rbfZ));
    progress('計算量子核函數（' + q + ' 量子位元 ZZ 特徵映射）', 30);
    const lams = [0.05, 0.1, 0.2, 0.4, 0.8];
    const QK = lams.map((l, i) => { progress('量子核函數：頻寬 ' + l, 30 + i * 8); return { K: quantumKernel(Z, l), label: '頻寬=' + l, lam: l }; });
    const zzName = '量子核函數（ZZ 特徵映射，' + q + ' 量子位元）';
    models.push(mkKRR(zzName, 'quantum', QK));
    progress('量子方法：投影量子核函數', 62);
    const PK = [0.1, 0.4].flatMap(l => { const F = projectedFeatures(Z, l); const D = sqDist(F); return [0.3, 1, 3].map(s => { const g = s / (3 * q); return { K: D.map(row => Float64Array.from(row, v => Math.exp(-g * v))), label: '頻寬=' + l + ', γ=' + g.toPrecision(2) }; }); });
    models.push(mkKRR('投影量子核函數（' + q + ' 量子位元）', 'quantum', PK));
    // VQC encoding bandwidth is chosen per fold from training data only (centred kernel–target alignment, Cortes et al. 2012)
    const vqcLams = [0.05, 0.1, 0.2, 0.4]; const vqcStates = new Map(vqcLams.map(l => [l, Z.map(z => zzState(z, l))]));
    if (nc <= q) models.push({ name: '變分量子分類器 VQC（' + q + ' 量子位元）', kind: 'quantum', type: 'fp', grid: [{ label: '2 層', fp: (tr, te) => { let bl = vqcLams[0], ba = -Infinity; for (const l of vqcLams) { const a = centeredAlignment(QK.find(k => k.lam === l).K, y, tr); if (a > ba) { ba = a; bl = l; } } return vqcFP(vqcStates.get(bl), y, nc, q, tr, te, r); } }] });
    models.push({ name: '邏輯回歸（全部特徵）', kind: 'classical', type: 'fp', grid: [0.001, 0.01, 0.1].map(l => ({ label: 'λ=' + l, fp: (tr, te) => lrFP(X, y, nc, tr, te, l) })) });
    models.push({ name: '隨機森林（全部特徵）', kind: 'classical', type: 'fp', grid: [{ label: '40 棵樹', fp: (tr, te) => rfFP(X, y, nc, tr, te, r) }] });
    models.push({ name: '神經網路（全部特徵）', kind: 'classical', type: 'fp', grid: [1e-4, 1e-2].map(l => ({ label: 'λ=' + l, fp: (tr, te) => mlpFP(X, y, nc, tr, te, l, r) })) });
    const res = models.map((md, i) => { progress('交叉驗證（5 折 × ' + REPS + ' 次）：' + md.name, 66 + 22 * i / models.length); const scores = [], chosen = []; for (const folds of foldSets) { const cv = cvModel(md, y, nc, folds, r); scores.push(...cv.scores); chosen.push(...cv.chosen); } return { name: md.name, kind: md.kind, scores, mean: mean(scores), sd: std(scores), chosen }; });
    // majority baseline
    const cnt = new Array(nc).fill(0); y.forEach(c => cnt[c]++);
    const cls = res.filter(m => m.kind === 'classical'); const qzz = res.find(m => m.name === zzName);
    const qm = res.filter(m => m.kind === 'quantum').sort((a, b) => b.mean - a.mean)[0];
    const bestC = cls.slice().sort((a, b) => b.mean - a.mean)[0];
    const diff = qm.scores.map((s, i) => s - bestC.scores[i]); const dMean = mean(diff), dSd = std(diff);
    const tt = correctedT(diff, 0.2); const nQ = res.filter(m => m.kind === 'quantum').length; const alphaG = 0.05 / nQ;
    // diagnostics: concentration per bandwidth, geometric difference at the most chosen bandwidth
    progress('診斷：核函數集中度與幾何差距', 88);
    const conc = QK.map(k => { const vals = []; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) vals.push(k.K[i][j]); return { lam: k.lam, mean: mean(vals), sd: std(vals) }; });
    const freq = new Map(); qzz.chosen.forEach(c => { const l = c.split(',')[0]; freq.set(l, (freq.get(l) || 0) + 1); });
    const lamLabel = Array.from(freq.entries()).sort((a, b) => b[1] - a[1])[0][0];
    const kq = QK.find(k => k.label === lamLabel);
    const ng = Math.min(n, 100); const sub = Array.from({ length: ng }, (_, i) => i);
    const bestRbfZ = rbfZ[2];
    const gcq = geometricDifference(sub.map(i => sub.map(j => bestRbfZ.K[i][j])), sub.map(i => sub.map(j => kq.K[i][j])), 0.01 * ng);
    const concChosen = conc.find(c => 'λ' && c.lam === kq.lam);
    const diags = [];
    diags.push({ name: '樣本數', value: n + ' 筆' + (P.sampled ? '（從 ' + P.totalRows + ' 筆中分層抽樣）' : ''), status: n < 60 ? 'warn' : 'ok', explain: n < 60 ? '樣本太少，交叉驗證分數的誤差會很大。' : '足夠做初步比較。' });
    diags.push({ name: '輸入壓縮', value: X[0].length + ' 個特徵 → ' + q + ' 維（保留 ' + (pc.explained * 100).toFixed(0) + '% 變異）', status: pc.explained < 0.6 ? 'warn' : 'ok', explain: '量子線路的量子位元數有限，所以先用主成分分析壓縮。古典模型同時用完整特徵和壓縮後特徵各測一次。' });
    diags.push({ name: '核函數集中', value: '選用頻寬下，兩兩相似度平均 ' + concChosen.mean.toFixed(3) + '、標準差 ' + concChosen.sd.toFixed(3), status: concChosen.sd < 0.01 ? 'bad' : concChosen.sd < 0.05 ? 'warn' : 'ok', explain: '若所有資料點在量子特徵空間裡看起來都差不多（標準差接近 0），模型就學不到東西；這是量子核函數放大規模時的已知問題。' });
    const ratio = gcq / Math.sqrt(ng);
    diags.push({ name: '幾何差距 g（Huang 等 2021）', value: 'g = ' + gcq.toFixed(2) + '（上限約 √N = ' + Math.sqrt(ng).toFixed(1) + '）', status: ratio < 0.3 ? 'bad' : ratio < 0.6 ? 'warn' : 'ok', explain: 'g 小代表古典 RBF 核函數能學到的東西涵蓋了量子核函數能學的，量子在這份資料上理論上沒有預測優勢；g 大只代表「有可能」，仍要看實際分數。' });
    diags.push({ name: '統計檢定（Nadeau–Bengio 修正 t 檢定）', value: '量子 − 古典 = ' + (dMean * 100).toFixed(2) + ' 個百分點，t = ' + tt.t.toFixed(2) + '（自由度 ' + tt.df + '），單尾 p：量子較好 ' + tt.pGreater.toFixed(3) + '、古典較好 ' + tt.pLess.toFixed(3), status: tt.pLess < 0.05 ? 'bad' : tt.pGreater < alphaG ? 'ok' : 'warn', explain: '5 折交叉驗證重複 ' + REPS + ' 次，兩個模型用完全相同的切分逐折比較。重複交叉驗證的分數彼此相關，這個檢定有修正，避免把運氣當成差異。量子模型有 ' + nQ + ' 個，挑最好的會占便宜，所以「量子較好」要求 p < ' + alphaG.toFixed(3) + '（Bonferroni 校正）。' });
    let level, title; const reasons = [];
    if (tt.pGreater < alphaG && dMean >= 0.01) { level = 'green'; title = '值得深入測試：量子模型在這份樣本上顯著較好'; reasons.push('最好的量子模型（' + qm.name + '）平均高出最好的古典模型 ' + (dMean * 100).toFixed(1) + ' 個百分點，p = ' + tt.pGreater.toFixed(4) + '。下一步請用更多資料、在 Colab 用 Qiskit 驗證，並考慮真機雜訊。'); }
    else if (tt.pLess >= 0.05) { level = 'yellow'; title = '和古典差不多：用量子沒有明顯好處'; reasons.push('最好的量子模型（' + qm.name + '）' + (qm.mean * 100).toFixed(1) + '%，最好的古典模型（' + bestC.name + '）' + (bestC.mean * 100).toFixed(1) + '%，差距在統計上不顯著。'); reasons.push('分數一樣時，古典模型更便宜、更快、不需要量子硬體，應優先選古典。'); }
    else { level = 'red'; title = '不建議用量子：古典模型預測得更準'; reasons.push('最好的古典模型（' + bestC.name + '）平衡準確率 ' + (bestC.mean * 100).toFixed(1) + '%，最好的量子模型（' + qm.name + '）' + (qm.mean * 100).toFixed(1) + '%，低了 ' + (-dMean * 100).toFixed(1) + ' 個百分點（p = ' + tt.pLess.toFixed(4) + '）。'); }
    if (ratio < 0.3) reasons.push('幾何差距很小：從理論上看，這份資料用古典核函數就能學到量子核函數能學的東西。');
    if (concChosen.sd < 0.01) reasons.push('量子核函數已經出現集中現象，再加量子位元只會更糟。');
    progress('整理報告', 98);
    return { task: 'cls', n, classes, classCounts: cnt, q, models: res, bestClassical: bestC.name, quantum: qm, diffMean: dMean, diffSd: dSd, repeats: REPS, test: tt, alphaG, concentration: conc, gcq, diagnostics: diags, verdict: { level, title, reasons }, labelName: P.labelName, recommend: recommendCls(res, cnt) };
  };

  // =====================================================================
  // GENERATIVE (binary distribution learning)
  // =====================================================================
  E.prepareGen = function (t, colIdx, opts) {
    opts = opts || {}; const maxBits = opts.maxBits || 10;
    const prof = E.profileColumns(t);
    let cols = (colIdx && colIdx.length ? colIdx : prof.map(p => p.index)).map(i => prof[i]).filter(p => p.unique >= 2);
    const bitsSpec = cols.map(p => {
      const vals = t.rows.map(rw => rw[p.index]);
      let fn, desc;
      if (p.binary) { const lv = Array.from(new Set(vals.filter(v => v !== ''))).sort(); fn = v => v === lv[1] ? 1 : 0; desc = '= ' + lv[1]; }
      else if (p.numeric) { const md = median(vals.map(Number).filter(isFinite)); fn = v => Number(v) > md ? 1 : 0; desc = '> 中位數 ' + (+md.toPrecision(4)); }
      else { const cnt = new Map(); vals.forEach(v => cnt.set(v, (cnt.get(v) || 0) + 1)); const mode = Array.from(cnt.entries()).sort((a, b) => b[1] - a[1])[0][0]; fn = v => v === mode ? 1 : 0; desc = '= ' + mode; }
      const bits = vals.map(fn); const pr = mean(bits); const H = pr <= 0 || pr >= 1 ? 0 : -(pr * Math.log2(pr) + (1 - pr) * Math.log2(1 - pr));
      return { name: p.name, idx: p.index, fn, desc, H };
    }).filter(b => b.H > 0.05);
    bitsSpec.sort((a, b) => b.H - a.H); const chosen = bitsSpec.slice(0, maxBits).sort((a, b) => a.idx - b.idx);
    if (chosen.length < 3) throw new Error('至少需要 3 個有變化的欄位才能做分布學習。');
    const data = t.rows.map(rw => { let x = 0; chosen.forEach((b, k) => { if (b.fn(rw[b.idx])) x |= 1 << k; }); return x; });
    return { bits: chosen.map(b => ({ name: b.name, desc: b.desc })), nb: chosen.length, data, dropped: bitsSpec.length - chosen.length };
  };

  function nllBits(prob, data) { let s = 0; for (const x of data) s -= Math.log2(Math.max(prob(x), 1e-12)); return s / data.length; }

  function trainIQP(nb, train, steps, r, progress, sigma) {
    const N = 1 << nb; const terms = [];
    for (let i = 0; i < nb; i++) terms.push([i]);
    for (let i = 0; i < nb; i++) for (let j = i + 1; j < nb; j++) terms.push([i, j]);
    const P = terms.length; const F = terms.map(tm => { const f = new Int8Array(N); for (let s = 0; s < N; s++) { let v = 1; for (const q of tm) if ((s >> q) & 1) v = -v; f[s] = v; } return f; });
    const c = new Float64Array(N); for (const x of train) c[x] += 1 / train.length;
    let th = Array.from({ length: P }, () => (sigma || 0.05) * gauss(r));
    const mA = new Float64Array(P), vA = new Float64Array(P); const lr = 0.05, b1 = 0.9, b2 = 0.999;
    const forward = (theta) => {
      const vr = new Float64Array(N), vi = new Float64Array(N); const u = 1 / Math.sqrt(N);
      for (let s = 0; s < N; s++) { let ph = 0; for (let k = 0; k < P; k++) ph += theta[k] * F[k][s]; vr[s] = u * Math.cos(ph); vi[s] = u * Math.sin(ph); }
      const ar = Float64Array.from(vr), ai = Float64Array.from(vi); fwht(ar); fwht(ai); for (let s = 0; s < N; s++) { ar[s] *= u; ai[s] *= u; }
      return { vr, vi, ar, ai };
    };
    let last = null;
    for (let st = 0; st < steps; st++) {
      const { vr, vi, ar, ai } = forward(th);
      // b_x = c_x conj(a_x)/|a_x|^2 ; h = H b ; grad_k = 2 Im Σ_z f_k(z) v_z h_z  (loss = -Σ c log|a|^2, natural log)
      const br = new Float64Array(N), bi = new Float64Array(N);
      for (let x = 0; x < N; x++) { if (!c[x]) continue; const p = Math.max(ar[x] * ar[x] + ai[x] * ai[x], 1e-15); br[x] = c[x] * ar[x] / p; bi[x] = -c[x] * ai[x] / p; }
      fwht(br); fwht(bi); const u = 1 / Math.sqrt(N); for (let s = 0; s < N; s++) { br[s] *= u; bi[s] *= u; }
      const imVH = new Float64Array(N); for (let s = 0; s < N; s++) imVH[s] = vr[s] * bi[s] + vi[s] * br[s];
      for (let k = 0; k < P; k++) {
        let gk = 0; const f = F[k]; for (let s = 0; s < N; s++) gk += f[s] * imVH[s]; gk *= 2;
        mA[k] = b1 * mA[k] + (1 - b1) * gk; vA[k] = b2 * vA[k] + (1 - b2) * gk * gk;
        const mh = mA[k] / (1 - Math.pow(b1, st + 1)), vh = vA[k] / (1 - Math.pow(b2, st + 1));
        th[k] -= lr * mh / (Math.sqrt(vh) + 1e-8);
      }
      if (progress && st % 50 === 0) progress('訓練量子生成模型（IQP 線路）：第 ' + st + ' / ' + steps + ' 步', 40 + 50 * st / steps);
      last = ar;
    }
    const { ar, ai } = forward(th);
    const p = new Float64Array(N); for (let s = 0; s < N; s++) p[s] = ar[s] * ar[s] + ai[s] * ai[s];
    return { p, params: P, theta: th, _forward: forward, _F: F };
  }
  E._trainIQP = trainIQP;

  function chowLiu(nb, train) {
    const n = train.length; const bit = (x, i) => (x >> i) & 1;
    const p1 = []; for (let i = 0; i < nb; i++) p1.push((train.reduce((s, x) => s + bit(x, i), 0) + 1) / (n + 2));
    const joint = (i, j) => { const c = [1, 1, 1, 1]; for (const x of train) c[bit(x, i) * 2 + bit(x, j)]++; const t = n + 4; return c.map(v => v / t); };
    const MI = []; for (let i = 0; i < nb; i++) MI.push(new Float64Array(nb));
    for (let i = 0; i < nb; i++) for (let j = i + 1; j < nb; j++) { const J = joint(i, j); let m = 0; for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) { const pa = a ? p1[i] : 1 - p1[i], pb = b ? p1[j] : 1 - p1[j]; m += J[a * 2 + b] * Math.log(J[a * 2 + b] / (pa * pb)); } MI[i][j] = MI[j][i] = m; }
    const inT = [0], parent = new Array(nb).fill(-1); const used = new Set([0]);
    while (used.size < nb) { let bi = -1, bj = -1, bv = -1; for (const i of used) for (let j = 0; j < nb; j++) if (!used.has(j) && MI[i][j] > bv) { bv = MI[i][j]; bi = i; bj = j; } parent[bj] = bi; used.add(bj); inT.push(bj); }
    const cond = parent.map((pa, j) => { if (pa < 0) return null; const c = [[1, 1], [1, 1]]; for (const x of train) c[bit(x, pa)][bit(x, j)]++; return c.map(rw => { const t = rw[0] + rw[1]; return [rw[0] / t, rw[1] / t]; }); });
    return x => { let p = 1; for (let j = 0; j < nb; j++) { if (parent[j] < 0) p *= bit(x, j) ? p1[j] : 1 - p1[j]; else p *= cond[j][bit(x, parent[j])][bit(x, j)]; } return p; };
  }

  function bernMix(nb, train, K, r) {
    let pk = new Array(K).fill(1 / K); let mu = Array.from({ length: K }, () => Array.from({ length: nb }, () => 0.25 + 0.5 * r()));
    const bit = (x, j) => (x >> j) & 1;
    for (let it = 0; it < 80; it++) {
      const R = train.map(x => { const l = pk.map((p, k) => { let v = Math.log(p); for (let j = 0; j < nb; j++) v += Math.log(bit(x, j) ? mu[k][j] : 1 - mu[k][j]); return v; }); const mx = Math.max(...l); const e = l.map(v => Math.exp(v - mx)); const t = e.reduce((a, b) => a + b, 0); return e.map(v => v / t); });
      const Nk = new Array(K).fill(0); const S = Array.from({ length: K }, () => new Array(nb).fill(0));
      train.forEach((x, i) => { for (let k = 0; k < K; k++) { Nk[k] += R[i][k]; for (let j = 0; j < nb; j++) if (bit(x, j)) S[k][j] += R[i][k]; } });
      pk = Nk.map(v => (v + 1) / (train.length + K)); mu = S.map((row, k) => row.map(v => (v + 1) / (Nk[k] + 2)));
    }
    return x => pk.reduce((sum, p, k) => { let v = p; for (let j = 0; j < nb; j++) v *= bit(x, j) ? mu[k][j] : 1 - mu[k][j]; return sum + v; }, 0);
  }
  function recommendGen(models, coverage) {
    const cls = models.filter(m => m.kind === 'classical').sort((a, b) => a.test - b.test); const top = cls[0];
    const info = { '各欄': ['各欄獨立抽樣', '不需要套件：每欄照比例抽', '欄位之間幾乎沒有關聯，最簡單的模型就夠了。'], 'Chow-Liu': ['貝氏網路（樹狀結構）', 'pgmpy（TreeSearch 結構學習＋BayesianNetwork）', '欄位之間主要是兩兩相依，用一棵樹就描述得很好，也看得出誰影響誰。'], '平滑': ['經驗分布（直接數次數）', 'pandas value_counts，加一點平滑', '資料量已經足夠涵蓋大部分組合，直接用出現比例就是最好的模型。'], '伯努利': ['潛在類別模型（伯努利混合）', 'StepMix（Python）或 poLCA（R）', '資料像是由幾種「客群」混合而成，每群有自己的特徵比例。'] };
    const key = Object.keys(info).find(k => top.name.startsWith(k)) || '平滑'; const [nm, tool, note] = info[key];
    const steps = ['用整份資料重新估計模型，再拿模型產生的資料和真實資料比對各欄比例與兩兩相關。'];
    if (coverage < 0.3) steps.push('組合數比資料多很多：欄位再多時可試 SDV 套件的 Gaussian Copula 或 CTGAN 產生合成資料。');
    return { primary: nm, why: '在沒看過的測試資料上表現最好（' + top.test.toFixed(3) + ' bits/筆）。' + note, alts: cls.slice(1, 3).map(m => m.name + '：' + m.test.toFixed(3)), tools: [tool, '要產生大量合成資料、欄位有數值也有類別：SDV（Gaussian Copula、CTGAN）'], steps };
  }

  // ---- statistics: Student t tail via regularised incomplete beta (Numerical Recipes betacf) ----
  function lgamma(x) { const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]; let y = x, t = x + 5.5; t -= (x + 0.5) * Math.log(t); let s = 1.000000000190015; for (const ci of c) s += ci / ++y; return -t + Math.log(2.5066282746310005 * s / x); }
  function betacf(a, b, x) { let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap; if (Math.abs(d) < 1e-30) d = 1e-30; d = 1 / d; let h = d;
    for (let m = 1; m <= 200; m++) { const m2 = 2 * m; let aa = m * (b - m) * x / ((qam + m2) * (a + m2)); d = 1 + aa * d; if (Math.abs(d) < 1e-30) d = 1e-30; c = 1 + aa / c; if (Math.abs(c) < 1e-30) c = 1e-30; d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)); d = 1 + aa * d; if (Math.abs(d) < 1e-30) d = 1e-30; c = 1 + aa / c; if (Math.abs(c) < 1e-30) c = 1e-30; d = 1 / d; const del = d * c; h *= del; if (Math.abs(del - 1) < 3e-12) break; }
    return h; }
  function ibeta(a, b, x) { if (x <= 0) return 0; if (x >= 1) return 1; const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x)); return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b; }
  function tUpper(t, df) { const p = 0.5 * ibeta(df / 2, 0.5, df / (df + t * t)); return t >= 0 ? p : 1 - p; } // P(T > t)
  E._tUpper = tUpper;
  // Nadeau & Bengio (2003) corrected resampled t-test for repeated k-fold CV; returns one-sided p-values
  function correctedT(d, testFrac) {
    const J = d.length, m = mean(d), s2 = d.reduce((s, v) => s + (v - m) * (v - m), 0) / Math.max(1, J - 1);
    const se = Math.sqrt((1 / J + testFrac / (1 - testFrac)) * s2) || 1e-12; const t = m / se;
    return { mean: m, t, df: J - 1, pGreater: tUpper(t, J - 1), pLess: tUpper(-t, J - 1) };
  }
  function normUpper(z) { return 0.5 * erfc(z / Math.SQRT2); }
  function erfc(x) { const z = Math.abs(x), t = 1 / (1 + 0.5 * z); const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))); return x >= 0 ? r : 2 - r; }
  E._stats = { correctedT, normUpper };

  E.analyzeGen = function (P, progress, opts) {
    opts = opts || {}; progress = progress || (() => {}); const SEEDS = opts.seeds || 3;
    const { nb, data } = P; const N = 1 << nb;
    const nTe = Math.max(10, Math.round(data.length * 0.3));
    if (data.length - nTe < 20) throw new Error('資料太少：至少需要約 30 筆。');
    const runs = []; // per split: [{name, kind, params, ll:Float64Array(test log2 p), test, train}]
    let lastTrain = null;
    for (let sd = 0; sd < SEEDS; sd++) {
      const r = rng((opts.seed || 31) * 100 + sd);
      const idx = shuffle(data.map((_, i) => i), r);
      const test = idx.slice(0, nTe).map(i => data[i]), train = idx.slice(nTe).map(i => data[i]); lastTrain = train;
      const out = []; const add = (name, kind, prob, params, note) => { const ll = Float64Array.from(test, x => Math.log2(Math.max(prob(x), 1e-12))); out.push({ name, kind, params, ll, test: -mean(Array.from(ll)), train: nllBits(prob, train), note }); };
      const base = 100 * sd / SEEDS;
      progress('第 ' + (sd + 1) + '/' + SEEDS + ' 次切分：古典模型', base + 5 / SEEDS);
      const pi = []; for (let i = 0; i < nb; i++) pi.push((train.reduce((s, x) => s + ((x >> i) & 1), 0) + 1) / (train.length + 2));
      add('各欄獨立（最簡單）', 'classical', x => { let p = 1; for (let i = 0; i < nb; i++) p *= ((x >> i) & 1) ? pi[i] : 1 - pi[i]; return p; }, nb);
      add('Chow-Liu 樹（兩兩相依）', 'classical', chowLiu(nb, train), nb + 2 * (nb - 1));
      const inner = train.slice(0, Math.floor(train.length * 0.8)), val = train.slice(Math.floor(train.length * 0.8));
      const hist = (tr, a) => { const c = new Float64Array(N); tr.forEach(x => c[x]++); const tot = tr.length + a * N; return x => (c[x] + a) / tot; };
      let bestA = 1, bestV = Infinity; for (const a of [0.01, 0.05, 0.2, 0.5, 1, 2]) { const v = nllBits(hist(inner, a), val); if (v < bestV) { bestV = v; bestA = a; } }
      add('平滑直方圖（直接數次數）', 'classical', hist(train, bestA), N - 1, '平滑係數 ' + bestA);
      { let bk = 2, bv = Infinity; for (const K of [2, 4, 8]) { const v = nllBits(bernMix(nb, inner, K, r), val); if (v < bv) { bv = v; bk = K; } }
        add('伯努利混合模型', 'classical', bernMix(nb, train, bk, r), bk * (nb + 1) - 1, bk + ' 群（在訓練資料內挑）'); }
      progress('第 ' + (sd + 1) + '/' + SEEDS + ' 次切分：IQP 量子生成模型', base + 40 / SEEDS);
      let iq = null; for (const sig of [0.05, 0.4, 0.8, 1.2]) { const cand = trainIQP(nb, train, opts.steps || 800, r, null, sig); if (!iq || nllBits(x => cand.p[x], train) < nllBits(x => iq.p[x], train)) iq = cand; }
      add('IQP 量子電路生成模型（' + nb + ' 量子位元）', 'quantum', x => iq.p[x], iq.params, '4 組隨機初值，取訓練損失最低的');
      progress('第 ' + (sd + 1) + '/' + SEEDS + ' 次切分：QCBM', base + 75 / SEEDS);
      const c = new Float64Array(N); train.forEach(x => c[x] += 1 / train.length);
      const he = trainHE(nb, 3, p => p.map((v, x) => c[x] ? -c[x] / Math.max(v, 1e-12) : 0), nb <= 8 ? 250 : 120, r, 0.05);
      add('量子電路生成模型 QCBM（' + nb + ' 量子位元）', 'quantum', x => he.p[x], he.params);
      runs.push(out);
    }
    // aggregate over splits
    const models = runs[0].map((m0, k) => { const tests = runs.map(o => o[k].test); return { name: m0.name, kind: m0.kind, params: m0.params, note: m0.note, test: mean(tests), testSd: std(tests), train: mean(runs.map(o => o[k].train)) }; });
    const cls = models.filter(m => m.kind === 'classical'), qms = models.filter(m => m.kind === 'quantum');
    const bestC = cls.slice().sort((a, b) => a.test - b.test)[0], qm = qms.slice().sort((a, b) => a.test - b.test)[0];
    const iC = models.indexOf(bestC), iQ = models.indexOf(qm);
    // paired per-sample difference (bits saved by quantum), averaged over splits; SE from one split (conservative: splits overlap)
    const dSplit = runs.map(o => { const a = o[iQ].ll, b = o[iC].ll; const d = Array.from(a, (v, i) => v - b[i]); return { m: mean(d), sd: std(d) }; });
    const dMean = mean(dSplit.map(x => x.m)), se = mean(dSplit.map(x => x.sd)) / Math.sqrt(nTe);
    const z = dMean / (se || 1e-12); const pGreater = normUpper(z), pLess = normUpper(-z); const alphaG = 0.05 / qms.length;
    const train = lastTrain; const coverage = train.length / N;
    const diags = [];
    diags.push({ name: '樣本數 vs 可能組合數', value: train.length + ' 筆訓練資料，' + N.toLocaleString() + ' 種可能組合（' + nb + ' 個欄位）', status: coverage > 2 ? 'bad' : coverage > 0.3 ? 'warn' : 'ok', explain: coverage > 2 ? '資料量遠大於組合數，直接數每種組合出現幾次就是近乎最好的模型，不需要任何生成模型。' : '組合數多於資料，需要模型去「猜」沒看過的組合，這才是生成模型能發揮的地方。' });
    diags.push({ name: '欄位間相依程度', value: '獨立模型比 Chow-Liu 樹差 ' + (models[0].test - models[1].test).toFixed(3) + ' bits', status: 'ok', explain: '差距小代表欄位之間幾乎沒有關聯，任何模型都學不到額外的結構。' });
    diags.push({ name: '過度擬合', value: qm.name + ' 訓練 ' + qm.train.toFixed(3) + ' / 測試 ' + qm.test.toFixed(3) + ' bits', status: qm.test - qm.train > 0.3 ? 'warn' : 'ok', explain: '測試明顯比訓練差，代表模型在背答案。' });
    diags.push({ name: '統計檢定', value: '量子 − 古典 = ' + (-dMean).toFixed(3) + ' bits/筆（負值代表量子較好），z = ' + z.toFixed(2) + '，單尾 p：量子較好 ' + pGreater.toFixed(3) + '、古典較好 ' + pLess.toFixed(3), status: pLess < 0.05 ? 'bad' : pGreater < alphaG ? 'ok' : 'warn', explain: '同一批測試資料上逐筆比較兩個模型，' + SEEDS + ' 次不同切分取平均。量子模型有 ' + qms.length + ' 個，挑最好的那個會占便宜，所以「量子較好」要求 p < ' + alphaG.toFixed(3) + '（Bonferroni 校正）。' });
    diags.push({ name: '量子位元數', value: nb + ' 個（每個欄位一個）' + (P.dropped ? '；另有 ' + P.dropped + ' 個欄位因上限未納入' : ''), status: 'ok', explain: '瀏覽器版上限 10 個欄位；更多欄位請用 Colab 版。' });
    let level, title; const reasons = [];
    if (pGreater < alphaG && dMean >= 0.02) { level = 'green'; title = '值得深入測試：量子生成模型在測試資料上顯著比古典好'; reasons.push(qm.name + ' 比最好的古典模型（' + bestC.name + '）每筆少 ' + dMean.toFixed(3) + ' bits，p = ' + pGreater.toFixed(4) + '。請用更多資料與更多欄位在 Colab 版驗證。'); }
    else if (pLess < 0.05) { level = 'red'; title = '不建議用量子：古典模型學得更好'; reasons.push('最好的古典模型（' + bestC.name + '）' + bestC.test.toFixed(3) + ' bits/筆，最好的量子模型（' + qm.name + '）' + qm.test.toFixed(3) + '，差距在統計上顯著（p = ' + pLess.toFixed(4) + '）。'); }
    else { level = 'yellow'; title = '和古典差不多：用量子沒有明顯好處'; reasons.push('最好的量子模型（' + qm.name + '）' + qm.test.toFixed(3) + '，最好的古典模型（' + bestC.name + '）' + bestC.test.toFixed(3) + ' bits/筆，差距在統計上不顯著。分數一樣時應選古典。'); }
    if (coverage > 2) reasons.push('資料筆數是組合數的 ' + coverage.toFixed(1) + ' 倍，直接數次數就夠了。');
    return { task: 'gen', nb, nTrain: train.length, nTest: nTe, seeds: SEEDS, models, bestClassical: bestC.name, quantum: qm, gap: -dMean, test: { z, pGreater, pLess, alphaG }, bits: P.bits, diagnostics: diags, verdict: { level, title, reasons }, recommend: recommendGen(models, coverage) };
  };


  // =====================================================================
  // SAMPLE DATA (synthetic, clearly labelled)
  // =====================================================================
  E.samples = {
    cloudflare: function (seed) {
      const r = rng(seed || 5); const rows = [['country', 'asn_type', 'method', 'tls', 'req_per_min', 'path_depth', 'resp_kb', 'ua_length', 'has_cookie', 'js_challenge_ms', 'is_bot']];
      const cs = ['TW', 'US', 'JP', 'DE', 'SG', 'BR', 'NL'];
      for (let i = 0; i < 360; i++) {
        const bot = r() < 0.4 ? 1 : 0;
        const hosting = bot ? r() < 0.75 : r() < 0.12;
        const rpm = Math.max(1, Math.round(bot ? Math.exp(2.6 + 0.9 * gauss(r)) : Math.exp(1.2 + 0.7 * gauss(r))));
        const depth = Math.max(1, Math.round((bot ? 1.6 : 3.1) + 1.2 * gauss(r)));
        const kb = Math.max(0.2, +(bot ? 3 + 4 * r() : 20 + 40 * r() + 10 * gauss(r)).toFixed(1));
        const ua = Math.max(0, Math.round(bot ? (r() < 0.3 ? 0 : 40 + 60 * r()) : 110 + 30 * gauss(r)));
        const cookie = bot ? (r() < 0.25 ? 1 : 0) : (r() < 0.85 ? 1 : 0);
        const js = bot ? (r() < 0.5 ? 0 : Math.round(30 + 40 * r())) : Math.round(250 + 120 * gauss(r));
        rows.push([cs[Math.floor(r() * cs.length)], hosting ? 'hosting' : 'residential', r() < (bot ? 0.35 : 0.15) ? 'POST' : 'GET', r() < (bot ? 0.5 : 0.85) ? '1.3' : '1.2', rpm, depth, kb, ua, cookie, Math.max(0, js), bot]);
      }
      return rows.map(r2 => r2.join(',')).join('\n');
    },
    jobs: function (seed) { // 14 batch jobs, conflict scores → split into two time slots (MaxCut)
      const r = rng(seed || 9); const n = 14; const rows = [['source', 'target', 'conflict']];
      const pos = Array.from({ length: n }, () => [r(), r()]);
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const d = Math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]); if (d < 0.42) rows.push(['job' + String(i + 1).padStart(2, '0'), 'job' + String(j + 1).padStart(2, '0'), (1 + Math.round(4 * r()))]); }
      return rows.map(x => x.join(',')).join('\n');
    },
    paint: function (seed, cars) { // binary paint shop, Ising edge list (negative weights allowed)
      const r = rng(seed || 3); cars = cars || 40; const seq = shuffle(Array.from({ length: 2 * cars }, (_, i) => i % cars), r);
      const seen = new Set(); const sigma = seq.map(c => { const s = seen.has(c) ? -1 : 1; seen.add(c); return s; });
      const w = new Map();
      for (let t = 0; t + 1 < seq.length; t++) { const a = seq[t], b = seq[t + 1]; if (a === b) continue; const k = Math.min(a, b) + ',' + Math.max(a, b); w.set(k, (w.get(k) || 0) - sigma[t] * sigma[t + 1]); }
      const rows = [['source', 'target', 'weight']];
      for (const [k, v] of w) { if (!v) continue; const [a, b] = k.split(','); rows.push(['car' + a, 'car' + b, v]); }
      return rows.map(x => x.join(',')).join('\n');
    },
    schedule: function (seed) { // 4 orders × 3 production lines, each order on exactly one line; same-line changeovers cost
      const r = rng(seed || 21); const orders = ['A', 'B', 'C', 'D']; const lines = ['L1', 'L2', 'L3']; const rows = [['kind', 'a', 'b', 'value', 'penalty']];
      const color = { A: '黑', B: '白', C: '黑', D: '紅' };
      for (const o of orders) for (const l of lines) rows.push(['linear', o + '@' + l, '', 1 + Math.round(4 * r()), '']);
      for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) for (const l of lines) { const c = color[orders[i]] === color[orders[j]] ? 1 : 4 + Math.round(3 * r()); rows.push(['pair', orders[i] + '@' + l, orders[j] + '@' + l, c, '']); }
      for (const o of orders) { rows.push(['choose', '訂單' + o, 1, '', '']); for (const l of lines) rows.push(['member', '訂單' + o, o + '@' + l, '', '']); }
      return rows.map(x => x.join(',')).join('\n');
    },
    behavior: function (seed) { // 8 binary behaviour flags from a latent tree
      const r = rng(seed || 13); const rows = [['mobile', 'night_visit', 'from_ad', 'viewed_3plus', 'added_cart', 'used_coupon', 'logged_in', 'purchased']];
      for (let i = 0; i < 500; i++) {
        const mobile = r() < 0.6, night = r() < (mobile ? 0.45 : 0.25), ad = r() < 0.35;
        const v3 = r() < (ad ? 0.35 : 0.55), cart = r() < (v3 ? 0.5 : 0.12), coupon = r() < (ad ? 0.4 : 0.1) && r() < 0.8;
        const login = r() < (cart ? 0.7 : 0.35); const buy = r() < (cart ? (coupon ? 0.75 : 0.45) : 0.03) || (login && cart && r() < 0.2);
        rows.push([mobile, night, ad, v3, cart, coupon, login, buy].map(b => b ? 1 : 0));
      }
      return rows.map(x => x.join(',')).join('\n');
    }
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = E; else root.QFC = E;
})(typeof self !== 'undefined' ? self : this);
