// Emits deterministic results from the JS engine for tests/crosscheck.py to compare with the Python version.
const E = require('../engine.js'); const fs = require('fs'); const out = {};
// 1. p=1 correlations on a fixed weighted graph
const edges = [[0, 1, 1.0], [1, 2, -0.5], [2, 3, 0.8], [3, 4, 1.0], [0, 4, 0.3], [1, 3, 0.6]]; const n = 5;
const adj = Array.from({ length: n }, () => new Map()); for (const [a, b, w] of edges) { adj[a].set(b, w); adj[b].set(a, w); }
out.corr = []; for (const [g, b] of [[0.7, 0.4], [1.3, 0.9]]) for (let u = 0; u < n; u++) for (let v = u + 1; v < n; v++) out.corr.push(E._corrPair(adj, u, v, g, b));
// 2. ZZ feature-map kernel on fixed points
const X = [[0.1, -0.5, 0.3], [1.2, 0.4, -0.7], [-0.3, 0.9, 0.2], [0.5, 0.5, 0.5]];
const st = X.map(x => E._zzState(x, 0.4)); out.kernel = [];
for (let i = 0; i < X.length; i++) for (let j = 0; j < X.length; j++) { let re = 0, im = 0; for (let k = 0; k < st[i].re.length; k++) { re += st[i].re[k] * st[j].re[k] + st[i].im[k] * st[j].im[k]; im += st[i].re[k] * st[j].im[k] - st[i].im[k] * st[j].re[k]; } out.kernel.push(re * re + im * im); }
// 3. QUBO conversion + exact optimum on the scheduling sample; exact MaxCut on the jobs sample
const g = E.buildQubo(E.parseCSV(fs.readFileSync(__dirname + '/../samples/schedule.csv', 'utf8')), {});
const res = E.analyzeOpt(g, () => {}, { seeds: 1 }); out.qubo = { nv: g.qubo.nv, const: g.qubo.const, sumW: g.qubo.sumW, exactCost: res.methods.find(m => m.name.startsWith('精確')).cost };
const gj = E.buildGraph(E.parseCSV(fs.readFileSync(__dirname + '/../samples/jobs.csv', 'utf8')), {}); const rj = E.analyzeOpt(gj, () => {}, { seeds: 1 });
out.jobs = { exact: rj.methods.find(m => m.name.startsWith('精確')).value, rqaoa: rj.methods.find(m => m.name.startsWith('RQAOA')).value };
const gp = E.buildGraph(E.parseCSV(fs.readFileSync(__dirname + '/../samples/paint.csv', 'utf8')), {}); out.paint = { rqaoa: E.rqaoa(gp.n, gp.edges, 8).value };
process.stdout.write(JSON.stringify(out));
