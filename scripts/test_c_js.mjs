// Node parity test for the shared RNG, vehicle and wind (rng.js, vehicle.js, wind.js vs the Python reference).
// Run from the project root:
//   D=$(mktemp -d); uv run python scripts/rng.py $D/rng.json && uv run python scripts/vehicle.py $D/veh.json \
//     && node scripts/test_c_js.mjs $D/rng.json $D/veh.json
import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import * as R from '../app/sim/rng.js';
import { Vehicle, velocityController, PARAMS } from '../app/sim/vehicle.js';
import { Wind, attritionMask } from '../app/sim/wind.js';

const [rngPath, vehPath] = process.argv.slice(2);
const load = (p) => JSON.parse(readFileSync(p, 'utf8'));

// 1. RNG: hash and uniform bit-identical, normal within 1e-14 relative
{
  let n = 0, maxZ = 0, diffZ = 0;
  for (const r of load(rngPath)) r.idx.forEach((i, j) => {
    assert.equal(R.hash32(r.seed, r.stream, r.step, i), r.h[j], `hash ${[r.seed, r.stream, r.step, i]}`);
    assert.equal(R.uniform(r.seed, r.stream, r.step, i), r.u[j]);
    const z = R.normal(r.seed, r.stream, r.step, i);
    maxZ = Math.max(maxZ, Math.abs(z - r.z[j]) / Math.max(1e-300, Math.abs(r.z[j]))); if (z !== r.z[j]) diffZ++;
    n++;
  });
  assert.ok(maxZ <= 1e-14, `normal rel err ${maxZ}`);
  console.log(`rng parity OK: ${n} draws; hash32 + uniform bit-identical; normal max rel err ${maxZ.toExponential(2)} (${diffZ}/${n} differ in last bits)`);
}


// 3. 200-step trajectory: controller + wind + attrition + vehicle
{
  const F = load(vehPath), n = F.n, dt = F.dt;
  assert.deepEqual(PARAMS, F.params, 'embedded PARAMS != vehicle.load_params()');  // 2. params identical
  const veh = new Vehicle(n); veh.reset(F.p0, F.v0);
  const wnd = new Wind(n, F.wind_cfg, F.seed, dt);
  let eP = 0, eV = 0, eW = 0, eA = 0;
  const maxErr = (a, b) => { let e = 0; for (let i = 0; i < a.length; i++) e = Math.max(e, Math.abs(a[i] - b[i])); return e; };
  for (let k = 0; k < F.steps; k++) {
    const t = k * dt;
    const alive = attritionMask(n, F.attrition, F.seed, t);
    assert.deepEqual(Array.from(alive), F.alive[k], `alive at step ${k}`);
    const vDes = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) {
      const g = 1 + (i % 3);
      vDes[3 * i] = 3 * Math.cos(0.3 * t + i) * g; vDes[3 * i + 1] = 3 * Math.sin(0.2 * t + i) * g; vDes[3 * i + 2] = 0.5 * Math.sin(t + i) * g;
    }
    const a = velocityController(vDes, veh.v);
    if (k % 50 === 0) for (let i = 0; i < n; i += 4) { a[3 * i] = 30; a[3 * i + 1] = -20; a[3 * i + 2] = 50; }
    eA = Math.max(eA, maxErr(a, F.acmd[k]));
    const w = wnd.step(k, veh.p);
    eW = Math.max(eW, maxErr(w, F.w[k]));
    veh.step(a, w, alive, dt);
    eP = Math.max(eP, maxErr(veh.p, F.p[k])); eV = Math.max(eV, maxErr(veh.v, F.v[k]));
  }
  const dead = F.alive[F.steps - 1].filter((x) => !x).length;
  assert.ok(Math.max(eP, eV, eW, eA) <= 1e-9, `parity ${[eP, eV, eW, eA]}`);
  console.log(`vehicle+wind parity OK: n=${n}, ${F.steps} steps, ${dead} dead at end; max |err| p ${eP.toExponential(2)} m, ` +
    `v ${eV.toExponential(2)} m/s, wind ${eW.toExponential(2)} m/s, a_cmd ${eA.toExponential(2)} m/s^2`);
}

// 4. JS speed (n = 50)
{
  const n = 50, veh = new Vehicle(n), wnd = new Wind(n, { mean_mps: [2, 0, 0], turbulence: 'moderate', shared_fraction: 0.5 }, 0, 0.02);
  const alive = new Uint8Array(n).fill(1), vd = new Float64Array(3 * n).fill(1), a = new Float64Array(3 * n);
  const t0 = performance.now(), N = 30000;
  for (let k = 0; k < N; k++) { velocityController(vd, veh.v, PARAMS, a); veh.step(a, wnd.step(k), alive, 0.02); }
  console.log(`JS timing n=50: controller+wind+vehicle ${((performance.now() - t0) / N * 1000).toFixed(2)} us/step`);
}
