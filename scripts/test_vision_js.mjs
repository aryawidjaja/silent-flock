// Parity: app/sim/vision.js vs scripts/vision.py on a fixed scene. From the project root:
//   uv run python scripts/vision.py /tmp/x/vision_fixture.json && node scripts/test_vision_js.mjs /tmp/x/vision_fixture.json
import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import { perceive, Tracker } from '../app/sim/vision.js';

const F = JSON.parse(readFileSync(process.argv[2], 'utf8')), n = F.n;
const pos = Float64Array.from(F.pos), vel = Float64Array.from(F.vel), alive = Uint8Array.from(F.alive);
const BIG = 1e300;   // Python writes nan as 1e300
function cmp(a, b, what) {
  let m = 0;
  for (let q = 0; q < b.length; q++) {
    if (b[q] === BIG) { assert.ok(Number.isNaN(a[q]), `${what}[${q}] should be NaN`); continue; }
    m = Math.max(m, Math.abs(a[q] - b[q]));
  }
  assert.ok(m <= 1e-12, `${what} max abs diff ${m}`);
  return m;
}
let worst = 0, nvis = 0;
for (const c of F.cases) {
  const o = perceive(pos, vel, alive, c.sensing, c.seed, c.step);
  for (let q = 0; q < n * n; q++) assert.equal(o.visible[q], c.visible[q], `visible[${q}] ${JSON.stringify(c.sensing)}`);
  nvis += c.visible.reduce((s, x) => s + x, 0);
  worst = Math.max(worst, cmp(o.rel, c.rel, 'rel'));
}
const tr = new Tracker(n, 0.1), S = F.cases[0].sensing;
let xf, ve;
for (let k = 0; k < 30; k++) {
  const p = pos.map((x, q) => x + vel[q] * 0.1 * k);
  const o = perceive(p, vel, alive, S, 2, k);
  [xf, ve] = tr.update(o.visible, o.rel, vel);
}
for (let q = 0; q < n * n; q++) assert.equal(tr.age[q], F.tracker.age[q], `age[${q}]`);
const wt = Math.max(cmp(xf, F.tracker.xf, 'xf'), cmp(ve, F.tracker.ve, 've'));
const t0 = performance.now(); for (let k = 0; k < 200; k++) perceive(pos, vel, alive, F.cases[1].sensing, 0, k);
console.log(`vision parity OK: ${F.cases.length} cases, ${nvis} detections, masks identical; rel max |diff| ${worst.toExponential(2)}; ` +
  `tracker 30 ticks max |diff| ${wt.toExponential(2)}; JS perceive n=${n}: ${((performance.now() - t0) / 200).toFixed(3)} ms`);
