"""Sensing: what each drone's cameras report about its neighbours, plus a per-pair
alpha-beta tracker that turns noisy relative positions into neighbour velocities.
JS twin: app/sim/vision.js (same maths, same RNG draws; parity: scripts/test_vision_js.mjs).

Model (every value is an ASSUMPTION; sweep ranges in brackets; sources listed below):
  1. Range limit `range_m` [4, 6, 10, 15]. Not a measured detection limit: Schilling et al. 2021 (RA-L)
     verified localisation only out to ~6 m (mocap accuracy test); UVDAR (Walter et al.) reports
     detection of blinking-UV markers out to ~15 m; a bare ~10 cm Crazyflie at 10 m spans ~3 px on a
     640 px / 120 deg camera, so passive detection beyond ~10 m is optimistic.
  2. Field of view: cone of full angle `fov_deg` [120, 360] about the heading = unit(own velocity),
     fallback +x if speed < 0.2 m/s (caller may pass `heading` to hold the last one).
  3. Occlusion (`occlusion` on/off): point-target sphere ray-cast. j is hidden from i if some other
     alive k with |p_k - p_i| < |p_j - p_i| lies within BODY_R of the segment i->j. Partial occlusion
     (target disc half covered) is ignored, so this slightly under-counts occlusion.
  4. Noise: range error grows as d^2 (stereo/size-based ranging: depth sigma ~ d^2 / (f * baseline),
     or d = f*W/w_px with w_px ~ 1/d), bearing error constant in angle (pixel centroid error), so lateral
     error grows as d. Scaled so that at d = 10 m the error is isotropic with 1-sigma = `pos_noise_m`
     per axis [0.1, 0.3, 1.0]: sigma_r(d) = p (d/10)^2, sigma_lat(d) = p (d/10). White across ticks.
     Draws: rng.normal(seed, S_VISION, step, 4*(i*n+j)+a), a=0 range, a=1..3 lateral (projected).
  5. No detection-probability falloff (detection is certain inside range/FOV/unoccluded). Add if the
     kill test shows results hinge on the range edge.
  6. Own velocity is known without GNSS (onboard IMU + optical-flow/VIO, as on Crazyflie Flow deck).
  7. Perfect data association: detections carry the true neighbour identity (no false positives, no
     ID swaps), so the Tracker indexes tracks by (i, j). Real passive vision would need association.
"""
import numpy as np
from rng import normal, S_VISION

BODY_R = 0.06       # m, occluder sphere radius. Crazyflie 2.1 is 92x92x29 mm motor-to-motor incl. mount feet
                    # (Bitcraze datasheet Rev 3); the MuJoCo Menagerie cf2 meshes incl. props reach 0.060 m
                    # horizontally / 0.061 m in 3D from the body origin (measured). No prop guards.
MIN_SPEED = 0.2     # m/s, below this the FOV heading falls back to +x
REF_D = 10.0        # m, distance at which pos_noise_m is the isotropic 1-sigma error
ALPHA, BETA = 0.23, 0.03   # alpha-beta gains: Kalata optimum for tracking index
                           # lam = sigma_a dt^2 / sigma_n = 1.0 * 0.01 / 0.3 = 0.033 (sigma_a ~ 1 m/s^2)


def headings(vel):
    """unit(vel), +x where speed < MIN_SPEED."""
    s = np.linalg.norm(vel, axis=1, keepdims=True)
    return np.where(s >= MIN_SPEED, vel / np.maximum(s, 1e-12), np.array([1.0, 0.0, 0.0]))


def perceive(pos, vel, alive, sensing, seed, step, heading=None):
    """Noisy camera detections. Returns visible (n,n) bool, rel (n,n,3) measured p_j - p_i (nan if not
    visible), dist (n,n) = |rel| (inf if not visible)."""
    n = len(pos)
    alive = np.asarray(alive, bool)
    h = headings(vel) if heading is None else heading
    d = pos[None, :, :] - pos[:, None, :]                    # d[i,j] = p_j - p_i
    r = np.sqrt(np.einsum("ijc,ijc->ij", d, d))
    ok = alive[:, None] & alive[None, :] & ~np.eye(n, dtype=bool) & (r <= sensing["range_m"])
    u = d / np.where(r > 0, r, 1.0)[..., None]               # unit bearings (0 on the diagonal)
    if sensing["fov_deg"] < 360:
        ok &= np.einsum("ijc,ic->ij", u, h) >= np.cos(np.radians(sensing["fov_deg"]) / 2)
    if sensing["occlusion"]:
        ok &= ~occluded(r, u, alive, ok)
    # noise: range along u ~ d^2, lateral (isotropic gaussian projected off u) ~ d
    p = sensing["pos_noise_m"]
    g = normal(seed, S_VISION, step, 4 * np.arange(n * n)[:, None] + np.arange(4)).reshape(n, n, 4)
    sr, sl = p * (r / REF_D) ** 2, p * (r / REF_D)
    gl = g[..., 1:]
    gl = gl - np.einsum("ijc,ijc->ij", gl, u)[..., None] * u
    rel = d + (sr * g[..., 0])[..., None] * u + sl[..., None] * gl
    rel[~ok] = np.nan
    dist = np.where(ok, np.sqrt(np.einsum("ijc,ijc->ij", rel, rel)), np.inf)
    return {"visible": ok, "rel": rel, "dist": dist}


def occluded(r, u, alive, cand):
    """(n,n) bool: occ[i,j] if a nearer alive k != i sits within BODY_R of segment i->j.
    For r_ik < r_ij the segment distance is r_ik*sin(angle jik) if cos > 0, else r_ik."""
    n = len(r)
    occ = np.zeros((n, n), bool)
    rows = np.flatnonzero(cand.any(1))
    if len(rows) == 0:
        return occ
    rk = np.where(alive[None, :], r[rows], np.inf)            # (m,n) occluder distances
    rk[np.arange(len(rows)), rows] = np.inf                   # k != i
    c = u[rows] @ u[rows].transpose(0, 2, 1)                  # c[m,j,k] = cos(angle j-i-k)
    rk2 = (rk * rk)[:, None, :]
    with np.errstate(invalid="ignore"):                       # inf*0 -> nan for dead/self k: never a hit
        perp2 = np.where(c > 0, rk2 * (1 - c * c), rk2)
    hit = (perp2 < BODY_R ** 2) & (rk[:, None, :] < r[rows][:, :, None])
    occ[rows] = hit.any(2)
    return occ


class Tracker:
    """Alpha-beta filter on rel[i,j] at the control rate. vel_est = own_vel[i] + filtered d(rel)/dt."""

    def __init__(self, n, dt_ctrl):
        self.n, self.dt = n, dt_ctrl
        self.x = np.full((n, n, 3), np.nan)   # filtered rel position
        self.v = np.full((n, n, 3), np.nan)   # filtered rel velocity
        self.age = np.zeros((n, n), int)      # updates since track start; 0 = no track; fresh while < 3

    def update(self, visible, rel, own_vel):
        new, cont = visible & (self.age == 0), visible & (self.age > 0)
        xp = self.x[cont] + self.v[cont] * self.dt
        res = rel[cont] - xp
        self.x[cont] = xp + ALPHA * res
        self.v[cont] += (BETA / self.dt) * res
        self.x[new], self.v[new] = rel[new], 0.0
        self.x[~visible] = self.v[~visible] = np.nan
        self.age = np.where(visible, self.age + 1, 0)
        return self.x.copy(), own_vel[:, None, :] + self.v


def _scene(n=50, spacing=2.5, seed=0):
    """Jittered cubic-lattice blob with nearest-neighbour spacing ~spacing, all flying +x at 3 m/s."""
    m = int(np.ceil(n ** (1 / 3)))
    g = np.stack(np.meshgrid(*[np.arange(m)] * 3, indexing="ij"), -1).reshape(-1, 3)[:n] * spacing
    from rng import uniform
    pos = g + (uniform(seed, 99, 0, np.arange(n * 3)).reshape(n, 3) - 0.5) * 0.5 * spacing
    return pos, np.tile([3.0, 0, 0], (n, 1))


if __name__ == "__main__":
    import json, sys, time
    S = {"range_m": 10.0, "fov_deg": 360.0, "occlusion": True, "pos_noise_m": 0.3}
    T = lambda **k: {**S, **k}
    v0 = np.tile([1.0, 0, 0], (4, 1))
    # FOV: drone directly behind invisible at 120 deg, visible at 360; range limit
    pos = np.array([[0, 0, 0], [-5, 0, 0], [5, 0, 0], [12, 0, 0.0]])
    a4 = np.ones(4, bool)
    o = perceive(pos, v0, a4, T(fov_deg=120, occlusion=False), 0, 0)
    assert not o["visible"][0, 1] and o["visible"][0, 2] and not o["visible"][0, 3]
    o = perceive(pos, v0, a4, T(occlusion=False), 0, 0)
    assert o["visible"][0, 1] and not o["visible"][0, 3] and not o["visible"].diagonal().any()
    assert np.isnan(o["rel"][0, 3]).all() and np.isinf(o["dist"][0, 3])
    # occlusion: j exactly behind k is hidden; offset by > BODY_R is visible; k itself visible
    pos = np.array([[0, 0, 0], [3, 0, 0], [6, 0, 0], [6, 1.5 * BODY_R * 2, 0.0]])
    o = perceive(pos, v0, a4, S, 0, 0)
    assert o["visible"][0, 1] and not o["visible"][0, 2] and o["visible"][0, 3], o["visible"]
    assert not o["visible"][2, 0]  # symmetric: 1 blocks 2->0 too
    assert perceive(pos, v0, a4, T(occlusion=False), 0, 0)["visible"][0, 2]
    pos2 = pos.copy(); pos2[1, 1] = BODY_R * 2      # move occluder off the line by 2 BODY_R
    assert perceive(pos2, v0, a4, S, 0, 0)["visible"][0, 2]
    # dead drones neither see nor are seen (and do not occlude)
    o = perceive(pos, v0, np.array([1, 0, 1, 1], bool), S, 0, 0)
    assert not o["visible"][1].any() and not o["visible"][:, 1].any() and o["visible"][0, 2]
    # noise: per-axis sigma at 10 m == pos_noise_m (radial and lateral), mean ~ 0
    K = 4000
    pos = np.zeros((2, 3)); pos[1] = [10.0, 0, 0]
    errs = np.array([perceive(pos, v0[:2], np.ones(2, bool), T(range_m=15), 7, k)["rel"][0, 1] - pos[1]
                     for k in range(K)])
    sd = errs.std(0)
    assert np.all(np.abs(sd / 0.3 - 1) < 0.1) and np.all(np.abs(errs.mean(0)) < 0.03), sd
    errs5 = np.array([perceive(pos * 0.5, v0[:2], np.ones(2, bool), S, 7, k)["rel"][0, 1] - pos[1] * 0.5
                      for k in range(K)])
    sd5 = errs5.std(0)
    assert abs(sd5[0] / 0.075 - 1) < 0.1 and np.all(np.abs(sd5[1:] / 0.15 - 1) < 0.1), sd5  # d^2, d
    # determinism and order-independence (subset of drones gives the same draws for the same (i,j))
    pos, vel = _scene(50, 2.5)
    al = np.ones(50, bool)
    a, b = perceive(pos, vel, al, S, 3, 11), perceive(pos, vel, al, S, 3, 11)
    assert np.array_equal(a["visible"], b["visible"]) and np.array_equal(a["rel"], b["rel"], equal_nan=True)
    assert not np.array_equal(a["rel"], perceive(pos, vel, al, S, 3, 12)["rel"], equal_nan=True)
    # timing
    tm = {}
    for n in (50, 100):
        p, v = _scene(n, 2.5)
        aln = np.ones(n, bool)
        perceive(p, v, aln, S, 0, 0)
        t0 = time.perf_counter(); R = 200
        for k in range(R): perceive(p, v, aln, S, 0, k)
        tm[n] = (time.perf_counter() - t0) / R * 1e3
    # occluded fraction in dense flocks: hidden / (in range & FOV)
    occ = {}
    for sp in (1.0, 2.5):
        for fov in (360.0, 120.0):
            p, v = _scene(50, sp)
            vis_off = perceive(p, v, al, T(fov_deg=fov, occlusion=False), 0, 0)["visible"]
            vis_on = perceive(p, v, al, T(fov_deg=fov), 0, 0)["visible"]
            occ[(sp, fov)] = (vis_off.sum() - vis_on.sum()) / vis_off.sum(), vis_on.sum(1).mean()
    # tracker: neighbour at 3 m/s alongside a 3 m/s observer at 5 m and 10 m; RMS velocity error after 2 s
    trk = {}
    for rng_m in (5.0, 10.0):
        tr, pos = Tracker(2, 0.1), np.array([[0, 0, 0], [0, rng_m, 0.0]])
        vel, err = np.tile([3.0, 0, 0], (2, 1)), []
        for k in range(1200):
            o = perceive(pos, vel, np.ones(2, bool), T(range_m=15), 5, k)
            xf, ve = tr.update(o["visible"], o["rel"], vel)
            if k >= 20: err.append(ve[0, 1] - vel[1])
            pos = pos + vel * 0.1
        e = np.array(err)
        trk[rng_m] = (np.sqrt((e ** 2).sum(1).mean()), np.sqrt((e ** 2).mean(0)))
        assert np.abs(e.mean(0)).max() < 0.05 and trk[rng_m][0] < 0.5, trk[rng_m]
        assert tr.age[0, 1] == 1200 and tr.age[1, 1] == 0
    # tracker reset when j leaves view; fresh track starts at own velocity
    tr = Tracker(2, 0.1); vis = np.array([[0, 1], [1, 0]], bool); rel = np.zeros((2, 2, 3)); rel[0, 1] = [1, 0, 0]
    tr.update(vis, rel, np.ones((2, 3))); tr.update(np.zeros((2, 2), bool), rel, np.ones((2, 3)))
    assert tr.age[0, 1] == 0 and np.isnan(tr.x[0, 1]).all()
    _, ve = tr.update(vis, rel, np.ones((2, 3)))
    assert tr.age[0, 1] == 1 and np.allclose(ve[0, 1], 1.0)
    if len(sys.argv) > 1:   # dump a fixed scene + Python outputs for the JS parity test
        p, v = _scene(50, 1.5, seed=4)
        alv = np.ones(50, bool); alv[[3, 17]] = False
        cases = []
        for kk, sens in enumerate([T(fov_deg=120.0), S, T(range_m=6.0, pos_noise_m=1.0)]):
            o = perceive(p, v, alv, sens, 9, kk)
            cases.append({"sensing": sens, "seed": 9, "step": kk, "visible": o["visible"].astype(int).ravel().tolist(),
                          "rel": np.nan_to_num(o["rel"], nan=1e300).ravel().tolist()})
        tr, outs = Tracker(50, 0.1), []
        for k in range(30):
            o = perceive(p + v * 0.1 * k, v, alv, T(fov_deg=120.0), 2, k)
            xf, ve = tr.update(o["visible"], o["rel"], v)
        outs = {"xf": np.nan_to_num(xf, nan=1e300).ravel().tolist(), "ve": np.nan_to_num(ve, nan=1e300).ravel().tolist(),
                "age": tr.age.ravel().tolist()}
        json.dump({"n": 50, "pos": p.ravel().tolist(), "vel": v.ravel().tolist(), "alive": alv.astype(int).tolist(),
                   "cases": cases, "tracker": outs}, open(sys.argv[1], "w"))
    print(f"vision self-check OK (BODY_R={BODY_R} m, alpha={ALPHA}, beta={BETA})")
    print(f"  noise sd at 10 m (radial, lat, lat) = {np.round(sd, 4)} target 0.3; at 5 m = {np.round(sd5, 4)} target (0.075, 0.15, 0.15)")
    print(f"  perceive time: n=50 {tm[50]:.3f} ms, n=100 {tm[100]:.3f} ms per call (occlusion on, 360 deg, 10 m)")
    for (sp, fov), (f, nv) in occ.items():
        print(f"  occluded fraction n=50 lattice spacing {sp} m, fov {fov:.0f}: {f:.4f} (mean visible per drone {nv:.1f})")
    for rr, (tot, ax) in trk.items():
        print(f"  tracker vel RMS error at {rr:.0f} m (3 m/s, |e| 3D): {tot:.3f} m/s; per axis (x, y=radial, z) = {np.round(ax, 3)}")
