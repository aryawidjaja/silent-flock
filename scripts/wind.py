"""Wind gusts and attrition. Python reference; app/sim/wind.js is the twin.

Wind at drone i = mean_mps + gust_i(t), gust per world axis a (x, y, z):
    gust_i = sqrt(f) * s(t) + sqrt(1 - f) * e_i(t),   f = shared_fraction   (variance preserving)
s (one per flock) and e_i (one per drone) are independent first-order Gauss-Markov processes with
standard deviation sigma_a and time constant tau_a = L_a / V, updated exactly at any dt:
    x_0 = sigma eta_0;   x_k = phi x_{k-1} + sigma sqrt(1 - phi^2) eta_k,   phi = exp(-dt / tau)
eta from rng.normal: s uses stream S_WIND_SHARED idx a; e uses S_WIND_EACH idx 3 i + a; counter = step index.

Intensities and scales: MIL-F-8785C low-altitude Dryden model (h < 1000 ft), as given in the MathWorks
"Dryden Wind Turbulence Model (Discrete)" documentation, evaluated at H_REF = 20 m (65.6 ft):
    sigma_w = 0.1 W20,   sigma_u = sigma_v = sigma_w / (0.177 + 0.000823 h)^0.4
    L_w = h,             L_u = L_v = h / (0.177 + 0.000823 h)^1.2           (h, L in ft)
    W20 (wind speed at 20 ft) = 15 / 30 / 45 kt for light / moderate / severe; "none" = no gusts.
Simplifications (quoted by the app's Model panel):
 - Every axis is first-order Gauss-Markov (Dryden u is first order; Dryden v, w are second order).
 - Gusts depend on time only, not on position (p is accepted and ignored): no spatial field, no
   frozen-turbulence correlation between nearby drones; shared_fraction stands in for that correlation.
 - Turbulence axes are world x, y (horizontal, sigma_u = sigma_v, L_u = L_v in MIL-F-8785C, so the frame
   does not matter) and z (vertical); fixed reference height 20 m, not each drone's altitude.
 - V in tau = L / V is V = max(|mean wind|, V_FLOOR = 2 m/s): the speed at which the turbulence passes the
   drone (a hovering drone in calm mean wind still needs a floor). Assumption; 2 m/s is the nominal airspeed
   of ../mutant-brain-drone's Dryden model (which uses the full second-order v, w filters and V fixed at 2 m/s).
 - W20 is not tied to mean_mps: intensity and mean wind are separate knobs.
Attrition: round(fraction n) drones (round half up), chosen once per seed as the smallest
rng.uniform(seed, S_ATTRITION, 0, i) among the candidates (ties by index), are dead for t >= t_s.
Run: uv run python scripts/wind.py"""
import math
import numpy as np
from rng import normal, uniform, S_WIND_SHARED, S_WIND_EACH, S_ATTRITION

KT = 0.514444
FT = 0.3048
H_REF_M = 20.0
V_FLOOR = 2.0  # m/s, = the sister project's nominal Dryden airspeed
BLOCK = 250
W20_KT = {"none": 0.0, "light": 15.0, "moderate": 30.0, "severe": 45.0}


def turbulence_params(level, h_m=H_REF_M):
    """(sigma (3,) m/s, L (3,) m) for world axes x, y, z (MIL-F-8785C low altitude)."""
    h = h_m / FT
    q = 0.177 + 0.000823 * h
    sw = 0.1 * W20_KT[level] * KT
    su = sw / q ** 0.4
    Lu, Lw = h / q ** 1.2 * FT, h * FT
    return np.array([su, su, sw]), np.array([Lu, Lu, Lw])


class Wind:
    def __init__(self, n, wind_cfg, seed, dt):
        self.n, self.cfg, self.seed, self.dt = n, dict(wind_cfg), seed, dt
        self.mean = np.asarray(wind_cfg["mean_mps"], dtype=np.float64)
        self.sigma, self.L = turbulence_params(wind_cfg["turbulence"])
        self.V = max(math.sqrt(float(self.mean @ self.mean)), V_FLOOR)
        self.tau = self.L / self.V
        self.phi = np.exp(-dt / self.tau)
        self.q = self.sigma * np.sqrt(1.0 - self.phi ** 2)
        f = wind_cfg["shared_fraction"]
        self.cs, self.ce = math.sqrt(f), math.sqrt(1.0 - f)
        self.on = bool(self.sigma.any())
        self._idx_e = np.arange(3 * n).reshape(n, 3)
        self._idx_s = np.arange(3)
        self.shared, self.each, self._next, self._b0 = np.zeros(3), np.zeros((n, 3)), 0, -1

    def step(self, step_idx, p=None):
        """(n,3) air velocity at each drone for dynamics step step_idx (call 0, 1, 2, ... in order)."""
        if not self.on:
            return np.broadcast_to(self.mean, (self.n, 3)).copy()
        assert step_idx in (0, self._next), f"Wind.step expects step {self._next} or 0, got {step_idx}"
        b0 = step_idx - step_idx % BLOCK                     # noise drawn in blocks of steps (same numbers, faster)
        if b0 != self._b0:
            ks = np.arange(b0, b0 + BLOCK)[:, None]
            self._es = normal(self.seed, S_WIND_SHARED, ks, self._idx_s[None, :])
            self._ee = normal(self.seed, S_WIND_EACH, ks, self._idx_e.ravel()[None, :]).reshape(BLOCK, self.n, 3)
            self._b0 = b0
        es, ee = self._es[step_idx - b0], self._ee[step_idx - b0]
        if step_idx == 0:
            self.shared, self.each = self.sigma * es, self.sigma * ee
        else:
            self.shared = self.phi * self.shared + self.q * es
            self.each = self.phi * self.each + self.q * ee
        self._next = step_idx + 1
        return self.mean + self.cs * self.shared + self.ce * self.each


def attrition_mask(n, attrition_cfg, seed, t, candidates=None):
    """bool (n,): True = alive at time t."""
    alive = np.ones(n, dtype=bool)
    k = math.floor(attrition_cfg["fraction"] * n + 0.5)
    if k == 0 or t < attrition_cfg["t_s"]:
        return alive
    cand = np.arange(n) if candidates is None else np.asarray(candidates, dtype=np.int64)
    u = uniform(seed, S_ATTRITION, 0, cand)
    alive[cand[np.lexsort((cand, u))[:k]]] = False
    return alive


if __name__ == "__main__":
    dt, n, T = 0.02, 200, 30000
    # sigma and tau of the Gauss-Markov update: empirical vs nominal (moderate, calm mean wind)
    w = Wind(n, {"mean_mps": [0, 0, 0], "turbulence": "moderate", "shared_fraction": 0.0}, 1, dt)
    xs = np.empty((T, n, 3))
    for k in range(T): xs[k] = w.step(k)
    sd = xs.std(axis=(0, 1))
    assert np.allclose(sd, w.sigma, rtol=0.05), (sd, w.sigma)
    lag = int(round(w.tau[2] / dt))   # vertical: tau 10 s; autocorrelation at lag tau should be e^-1
    x = xs[:, :, 2] - xs[:, :, 2].mean(0)
    rho = (x[:-lag] * x[lag:]).mean() / x.var()
    assert abs(rho - math.exp(-1)) < 0.05, rho
    lag1 = (x[:-1] * x[1:]).mean() / x.var()
    assert abs(lag1 - w.phi[2]) < 2e-3, (lag1, w.phi[2])
    # same sigma at another dt (exact discretisation)
    w2 = Wind(n, w.cfg, 2, 0.1); xs2 = np.array([w2.step(k) for k in range(6000)])
    assert np.allclose(xs2.std(axis=(0, 1)), w.sigma, rtol=0.06)
    # f = 1: all drones identical; f = 0: uncorrelated across drones
    w1 = Wind(10, {"mean_mps": [1, 2, 0], "turbulence": "light", "shared_fraction": 1.0}, 3, dt)
    g1 = np.array([w1.step(k) for k in range(500)])
    assert np.all(g1 == g1[:, :1, :]) and abs(g1[:, 0, 0].mean() - 1) < 1.5
    pair_corr = lambda g: np.corrcoef(g[:, :, 0].T)[np.triu_indices(g.shape[1], 1)].mean()  # mean over drone pairs
    c = pair_corr(xs)
    assert abs(c) < 0.03, c
    wh = Wind(50, {"mean_mps": [0, 0, 0], "turbulence": "severe", "shared_fraction": 0.5}, 4, dt)
    gh = np.array([wh.step(k) for k in range(20000)])
    ch = pair_corr(gh)
    assert abs(ch - 0.5) < 0.15, ch  # half the variance shared (one shared path, tau_x 58 s -> wide CI)
    # none: constant mean; determinism
    w0 = Wind(5, {"mean_mps": [3, 0, 0], "turbulence": "none", "shared_fraction": 0.3}, 0, dt)
    assert np.all(w0.step(0) == [3, 0, 0])
    wa, wb = (Wind(5, w.cfg, 9, dt) for _ in range(2))
    assert all(np.array_equal(wa.step(k), wb.step(k)) for k in range(10))
    e1 = normal(1, S_WIND_EACH, T - 1, np.arange(3 * n)).reshape(n, 3)  # block cache == direct draw (f=0: wind = each)
    assert np.allclose(w.each, w.phi * xs[-2] + w.q * e1, atol=1e-12)
    # attrition
    cfg = {"fraction": 0.2, "t_s": 60.0}
    assert attrition_mask(50, cfg, 0, 59.99).all()
    m0 = attrition_mask(50, cfg, 0, 60.0)
    assert (~m0).sum() == 10 and np.array_equal(m0, attrition_mask(50, cfg, 0, 300.0))
    assert not np.array_equal(m0, attrition_mask(50, cfg, 1, 60.0))
    mc = attrition_mask(50, cfg, 0, 61, candidates=np.arange(3, 50))
    assert (~mc).sum() == 10 and mc[:3].all()
    assert (~attrition_mask(10, {"fraction": 0.25, "t_s": 0}, 0, 0)).sum() == 3  # 2.5 rounds half up
    import time
    wt = Wind(50, {"mean_mps": [2, 0, 0], "turbulence": "moderate", "shared_fraction": 0.5}, 0, dt)
    t0 = time.perf_counter()
    for k in range(3000): wt.step(k)
    us = (time.perf_counter() - t0) / 3000 * 1e6
    s, L = turbulence_params("moderate")
    print(f"wind self-check OK: moderate sigma nominal {np.round(w.sigma, 3)} empirical {np.round(sd, 3)} m/s; "
          f"tau {np.round(w.tau, 2)} s; rho(tau_z) {rho:.3f} (e^-1 = 0.368); lag-1 {lag1:.5f} vs phi {w.phi[2]:.5f}; "
          f"corr(f=0) {c:.3f}, corr(f=0.5) {ch:.3f}; L {np.round(L, 1)} m")
    for lev in ("light", "moderate", "severe"):
        s, L = turbulence_params(lev); print(f"  {lev:8s} sigma_xyz {np.round(s, 3)} m/s  L_xyz {np.round(L, 1)} m")
    print(f"timing: Wind.step n=50 {us:.1f} us/step")
