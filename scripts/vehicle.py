"""Translational quadrotor model, Crazyflie 2.x class. Python reference; app/sim/vehicle.js is the twin.

Parameters: results/vehicle_params.json (= ../mutant-brain-drone's file, unchanged) + results/vehicle_params_flock.json
(flock extras), merged by load_params(); both made/copied by scripts/extract_vehicle_params.py, every value sourced.
Each drone i has state p, v (world, m), thrust magnitude T (N) and thrust direction b (unit vector, body z axis).

One step of length dt, given commanded acceleration a_cmd and air velocity w at the drone:
 1. Desired thrust vector, altitude first:
      f_z  = clip(m (a_z + g), T_min, T_max)
      f_h  = m a_h, scaled down to |f_h| <= min(f_z tan(tilt_max), sqrt(T_max^2 - f_z^2))
      T_d  = |f_d|,  b_d = f_d / T_d          (so tilt(b_d) <= tilt_max and T_min <= T_d <= T_max)
 2. Lags (exact first-order factors, so the time constants hold at any dt):
      b <- normalise(b + alpha (b_d - b)),  alpha = 1 - exp(-dt / tau_att)   (nlerp: a first-order lag on the
           thrust direction, equal to a first-order lag on the tilt vector to second order in the per-step angle)
      T <- T + beta (T_d - T),              beta  = 1 - exp(-dt / tau_motor)
    The cone |tilt| <= tilt_max is convex, so the lagged b never exceeds tilt_max; T stays in [T_min, T_max].
 3. Acceleration:  a = T b / m - g z_hat - (D (v - w) + k2 |v - w| (v - w)) / m
      D = diag(drag_lin) (Förster 2015 rotor drag at hover rotor speed), k2 = quadratic body drag (assumption).
 4. Semi-implicit Euler:  v <- v + a dt;  p <- p + v dt.

Simplifications (quoted by the app's Model panel):
 - No rotational dynamics: attitude is a first-order lag on the thrust direction (tau_att from the stock
   firmware attitude gain); yaw is ignored; no rotor-level dynamics beyond one thrust lag.
 - Rotor drag is linear in air-relative velocity with a constant coefficient (rotor speed fixed at hover)
   and applied per world axis (body ~ world for tilt <= 20 deg). Quadratic body drag is an assumption.
 - No ground effect, no downwash between drones, no collisions, no battery sag.
 - Dead drones (alive False) are removed: their p is frozen, v set to 0, and they get no dynamics.
 - velocity_controller: P on velocity with gain from the stock firmware, plus drag feed-forward at the
   desired ground velocity (air velocity unknown onboard), clipped to the tilt and thrust limits.
Run: uv run python scripts/vehicle.py"""
import json
from pathlib import Path
import numpy as np

G = 9.81
_RES = Path(__file__).resolve().parents[1] / "results"
PARAMS_FILE, EXTRA_FILE = _RES / "vehicle_params.json", _RES / "vehicle_params_flock.json"


def load_params(path=PARAMS_FILE, extra=EXTRA_FILE):
    """Flat {key: value}: the sister project's file (unchanged) mapped to generic keys, plus the flock extras.
    Derived here: thrust_max = 4 f_max; thrust_min = 4 f(cmd 0) = 4 thrust_a0; linear drag = |Kaero| * sum of
    rotor speeds at hover (Förster eq. 4.2 with rotor speed fixed at hover: f(cmd) = m g / 4, w = b1 cmd + b0)."""
    b = {k: v["value"] for k, v in json.loads(Path(path).read_text())["params"].items()}
    out = {k: v["value"] for k, v in json.loads(Path(extra).read_text()).items() if not k.startswith("_")}
    a2, a1, a0 = b["thrust_a2"], b["thrust_a1"], b["thrust_a0"]
    cmd_h = (-a1 + np.sqrt(a1 * a1 - 4 * a2 * (a0 - b["mass"] * b["g"] / 4))) / (2 * a2)
    w_sum = 4 * (b["rotor_b1"] * cmd_h + b["rotor_b0"])
    assert b["g"] == G
    out.update(mass_kg=b["mass"], inertia_kgm2=[b["Ixx"], b["Iyy"], b["Izz"]], arm_m=b["arm_length"],
               thrust_max_N=4 * b["f_max"], thrust_min_N=4 * a0, tau_motor_s=b["motor_tau"],
               drag_lin_Nspm=[float(-b["kaero_xy"] * w_sum)] * 2 + [float(-b["kaero_z"] * w_sum)])
    return out


PARAMS = load_params()


class Vehicle:
    def __init__(self, n, params=PARAMS):
        self.n, self.params = n, params
        self.m = params["mass_kg"]
        self.T_max, self.T_min = params["thrust_max_N"], params["thrust_min_N"]
        self.tan_tilt = np.tan(params["max_tilt_rad"])
        self.D = np.array(params["drag_lin_Nspm"], float)
        self.k2 = params["drag_quad_Nsm2m2"]
        self.tau_att, self.tau_motor = params["tau_att_s"], params["tau_motor_s"]
        self._dt = None
        self.reset(np.zeros((n, 3)))

    def reset(self, p0, v0=None):
        self.p = np.array(p0, dtype=np.float64).reshape(self.n, 3)
        self.v = np.zeros((self.n, 3)) if v0 is None else np.array(v0, dtype=np.float64).reshape(self.n, 3)
        self.b = np.zeros((self.n, 3)); self.b[:, 2] = 1.0       # level
        self.T = np.full(self.n, self.m * G)                      # hover thrust

    @property
    def tilt(self):
        return np.arccos(np.clip(self.b[:, 2], -1.0, 1.0))

    @property
    def thrust(self):
        return self.T

    def step(self, a_cmd, wind_vel, alive, dt):
        if dt != self._dt:
            self._dt, self._alpha, self._beta = dt, 1 - np.exp(-dt / self.tau_att), 1 - np.exp(-dt / self.tau_motor)
        m = self.m
        fd = m * a_cmd                                            # (n,3) copy
        fz = np.clip(fd[:, 2] + m * G, self.T_min, self.T_max)
        fd[:, 2] = fz
        h = np.sqrt(fd[:, 0] ** 2 + fd[:, 1] ** 2)
        hmax = np.minimum(fz * self.tan_tilt, np.sqrt(self.T_max ** 2 - fz ** 2))
        s = hmax / np.maximum(h, hmax)                            # 1 if within limit (h <= hmax), h=0 -> 1 (hmax > 0)
        fd[:, :2] *= s[:, None]
        Td = np.sqrt(np.einsum("ij,ij->i", fd, fd))
        b = self.b + self._alpha * (fd / Td[:, None] - self.b)
        b /= np.sqrt(np.einsum("ij,ij->i", b, b))[:, None]
        T = self.T + self._beta * (Td - self.T)
        vr = self.v - wind_vel
        acc = (T[:, None] * b - (self.D + self.k2 * np.sqrt(np.einsum("ij,ij->i", vr, vr))[:, None]) * vr) / m
        acc[:, 2] -= G
        v = self.v + acc * dt
        p = self.p + v * dt
        if alive.all():
            self.p, self.v, self.b, self.T = p, v, b, T
        else:
            a3 = alive[:, None]
            self.p = np.where(a3, p, self.p); self.v = np.where(a3, v, 0.0)
            self.b = np.where(a3, b, self.b); self.T = np.where(alive, T, self.T)


def velocity_controller(v_des, v, params=PARAMS):
    """a_cmd = kv (v_des - v) + drag(v_des)/m, horizontal |a| <= g tan(tilt_max), a_z within thrust limits."""
    m, tilt = params["mass_kg"], params["max_tilt_rad"]
    vd = np.asarray(v_des, dtype=np.float64)
    sp = np.sqrt(np.einsum("ij,ij->i", vd, vd))[:, None]
    a = params["kv_per_s"] * (vd - v) + (np.asarray(params["drag_lin_Nspm"]) + params["drag_quad_Nsm2m2"] * sp) * vd / m
    h = np.sqrt(a[:, 0] ** 2 + a[:, 1] ** 2)
    hmax = G * np.tan(tilt)
    a[:, :2] *= (hmax / np.maximum(h, hmax))[:, None]
    a[:, 2] = np.clip(a[:, 2], params["thrust_min_N"] / m - G, params["thrust_max_N"] * np.cos(tilt) / m - G)
    return a


def terminal_speed(params=PARAMS):
    """Level flight at max tilt with f_z = mg: D_x v + k2 v^2 = m g tan(tilt_max)."""
    k1, k2, F = params["drag_lin_Nspm"][0], params["drag_quad_Nsm2m2"], params["mass_kg"] * G * np.tan(params["max_tilt_rad"])
    return (-k1 + np.sqrt(k1 * k1 + 4 * k2 * F)) / (2 * k2)


def _fixture(path, steps=200, n=20):
    """Fixed scenario for the JS parity test: mixed commands, wind on (from wind.py), some dead."""
    import wind as W
    veh = Vehicle(n)
    i = np.arange(n)
    veh.reset(np.c_[i * 2.0, np.zeros(n), np.full(n, 20.0)], np.c_[np.zeros(n), np.sin(i), np.zeros(n)])
    wnd = W.Wind(n, {"mean_mps": [2.0, -1.0, 0.0], "turbulence": "moderate", "shared_fraction": 0.5}, 7, 0.02)
    att = {"fraction": 0.2, "t_s": 1.0}
    out = {"params": PARAMS, "n": n, "dt": 0.02, "steps": steps, "seed": 7, "wind_cfg": wnd.cfg, "attrition": att,
           "p0": veh.p.ravel().tolist(), "v0": veh.v.ravel().tolist(), "p": [], "v": [], "w": [], "alive": [], "acmd": []}
    for k in range(steps):
        t = k * 0.02
        alive = W.attrition_mask(n, att, 7, t)
        v_des = np.c_[3 * np.cos(0.3 * t + i), 3 * np.sin(0.2 * t + i), 0.5 * np.sin(t + i)] * (1 + (i % 3))[:, None]
        a = velocity_controller(v_des, veh.v)
        if k % 50 == 0: a[i % 4 == 0] = [30.0, -20.0, 50.0]   # saturating kicks
        w = wnd.step(k, veh.p)
        veh.step(a, w, alive, 0.02)
        for key, val in (("p", veh.p), ("v", veh.v), ("w", w), ("acmd", a)): out[key].append(val.ravel().tolist())
        out["alive"].append(alive.astype(int).tolist())
    Path(path).write_text(json.dumps(out))


if __name__ == "__main__":
    import sys, time
    P, m = PARAMS, PARAMS["mass_kg"]
    dt, n = 0.02, 4
    z = np.zeros((n, 3)); live = np.ones(n, bool)
    # 1. hover is an equilibrium
    veh = Vehicle(n); veh.reset(np.c_[np.zeros(n), np.zeros(n), np.full(n, 20.0)])
    for _ in range(500): veh.step(z, z, live, dt)
    acc = (veh.T[:, None] * veh.b / m - [0, 0, G])
    assert np.abs(acc).max() < 1e-9 and np.abs(veh.v).max() < 1e-9, acc
    # 2. step response to 3 m/s (x), limits never exceeded
    veh = Vehicle(1); vd = np.array([[3.0, 0, 0]]); t90 = None; max_tilt = max_T = 0.0
    for k in range(1000):
        veh.step(velocity_controller(vd, veh.v), np.zeros((1, 3)), np.ones(1, bool), dt)
        max_tilt, max_T = max(max_tilt, veh.tilt[0]), max(max_T, veh.T[0])
        if t90 is None and veh.v[0, 0] >= 2.7: t90 = (k + 1) * dt
    vss, zdrift = veh.v[0, 0], abs(veh.p[0, 2])
    assert t90 is not None and 0.3 < t90 < 3.0 and abs(vss - 3.0) < 0.05 and zdrift < 0.05, (t90, vss, zdrift)
    # 3. saturating commands: tilt <= max_tilt, T in [T_min, T_max] for any dt
    veh = Vehicle(64); rs = np.random.default_rng(0)  # test inputs only, not model randomness
    for k in range(400):
        veh.step(rs.normal(0, 40, (64, 3)), rs.normal(0, 5, (64, 3)), np.ones(64, bool), [0.02, 0.005, 0.1][k % 3])
        assert veh.tilt.max() <= P["max_tilt_rad"] + 1e-12 and veh.T.max() <= P["thrust_max_N"] + 1e-12
        assert veh.T.min() >= P["thrust_min_N"] - 1e-12
    # 4. terminal speed at max tilt with altitude hold = analytic drag limit
    veh = Vehicle(1)
    for _ in range(3000): veh.step(np.array([[100.0, 0, 0]]), np.zeros((1, 3)), np.ones(1, bool), dt)
    vt = terminal_speed()
    assert abs(veh.v[0, 0] - vt) < 1e-6 and abs(veh.v[0, 2]) < 1e-9 and P["v_max_mps"] < vt, (veh.v, vt)
    # 5. wind pushes a hovering (a_cmd = 0) drone downwind; it ends drifting at the wind speed
    veh = Vehicle(1); w = np.array([[2.0, -1.0, 0]])
    for _ in range(3000): veh.step(np.zeros((1, 3)), w, np.ones(1, bool), dt)
    assert veh.p[0, 0] > 10 and veh.p[0, 1] < -5 and np.abs(veh.v[0, :2] - w[0, :2]).max() < 1e-6
    # 6. energy: with hover thrust held (a_cmd = 0 -> f = mg z_hat) and still air, drag only removes energy
    veh = Vehicle(1); veh.reset([[0, 0, 20.0]], [[4.0, 0, 1.0]]); ke = [0.5 * m * (veh.v ** 2).sum()]
    for _ in range(500):
        veh.step(np.zeros((1, 3)), np.zeros((1, 3)), np.ones(1, bool), dt); ke.append(0.5 * m * (veh.v ** 2).sum())
    assert np.all(np.diff(ke) <= 1e-15) and ke[-1] < 1e-2 * ke[0]
    # 7. dead drones are frozen
    veh = Vehicle(3); veh.reset(np.ones((3, 3)), np.ones((3, 3))); al = np.array([True, False, True])
    veh.step(np.ones((3, 3)), np.zeros((3, 3)), al, dt)
    assert np.all(veh.p[1] == 1) and np.all(veh.v[1] == 0) and np.all(veh.p[0] != 1)
    # timings
    tim = {}
    for nn in (50, 100):
        veh = Vehicle(nn); a, w0, al = np.ones((nn, 3)), np.zeros((nn, 3)), np.ones(nn, bool)
        al_d = al.copy(); al_d[::7] = False
        for lab, msk in (("all alive", al), ("some dead", al_d)):
            t0 = time.perf_counter()
            for _ in range(3000): veh.step(a, w0, msk, dt)
            tim[(nn, lab)] = (time.perf_counter() - t0) / 3000 * 1e6
        t0 = time.perf_counter()
        for _ in range(3000): velocity_controller(a, w0)
        tim[(nn, "velocity_controller")] = (time.perf_counter() - t0) / 3000 * 1e6
    if len(sys.argv) > 1: _fixture(sys.argv[1])
    print(f"vehicle self-check OK: hover |a| {np.abs(acc).max():.1e}; 3 m/s step t90 {t90:.2f} s, v_ss {vss:.4f}, "
          f"max tilt {np.degrees(max_tilt):.1f} deg, max T {max_T:.4f} N; terminal speed {vt:.3f} m/s (analytic) "
          f"at {np.degrees(P['max_tilt_rad']):.0f} deg; energy monotone; wind drift OK")
    print("timings (us/step): " + ", ".join(f"n={k[0]} {k[1]} {v:.1f}" for k, v in tim.items()))
