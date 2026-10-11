"""Silent-flock model: a radio-jammed, camera-only quadrotor flock with anchors, spoofed
GNSS drones and uninformed followers. Python reference; app/sim/flock.js mirrors it line by line.
Run: uv run python scripts/flock.py   (self-check + timing)

Modelling simplifications (quoted in the app's "Model" panel):
 1. Roles are fixed by index: [0, A) anchors, [A, A+G) GNSS drones, rest followers. Positions at launch are
    random (rng S_INIT), so roles are randomly placed in the launch blob. Drones start hovering (v = 0).
 2. Anchors know the true goal bearing without GNSS (landmark / visual-odometry navigation, not modelled
    in detail): their goal direction is unit(goal - p_true). No navigation drift.
 3. GNSS drones believe p_hat = p_true + s(t), s = -offset (common mode: every GNSS drone gets the same
    s; linear ramp over ramp_s). So they steer to the fake goal = goal + offset. They do not know it.
 4. Followers know nothing about the goal; they only copy and stay near what they see.
 5. Own velocity is known without GNSS (IMU + optical flow / VIO); altitude from a barometer (exact).
 6. Neighbours come only from vision.perceive (range, FOV, sphere occlusion, noise) at CTRL_HZ = 10. Spacing
    terms use the raw detections; neighbour velocities come from vision.Tracker (alpha-beta) run in each
    drone's odometry frame (detections shifted by its own VIO displacement, assumed drift-free), so its own
    accelerations do not leak into them. Control is held (zero-order) between ticks; the vehicle's velocity
    loop runs every dynamics step (dt).
 7. Desired-velocity flocking (replaces the toy's constant-speed, unit-normalised rule, which fragmented).
    Drone i uses neighbours U_i (k nearest visible; stable argsort on measured distance, ties -> lower index),
    r_ij = measured p_j - p_i, d_ij = |r_ij|, u_ij = r_ij / d_ij, v_j = tracked neighbour velocity:
      separation  sep = -sum_j K_SEP (D0 - d_ij) u_ij  over d_ij < D0        (summed: more crowding, more push)
      cohesion    c = mean_j r_ij;  coh = sat_VCOH(K_COH max(|c| - C_DEAD, 0) unit(c))   (toward the centroid,
                  with a deadband so edge drones are not dragged back by the flock's own shape)
      alignment   vbar = (W_SELF v_i + sum_j v_j) / (W_SELF + |U_i|)
      heading     h = A_LEAK vbar / V_CRUISE + w_i unit_xy(goal_believed - p_believed)
      propulsion  prop = V_CRUISE h / max(1, |h|), then |prop| <= min(mean_j |v_j| + V_UP, min_j |v_j| + V_DOWN)
                  (speed matching: the flock accelerates as one body, nobody outruns the slowest neighbour
                  it uses, so informed drones cannot leave and braking propagates through the flock)
      v_des = sat_VMAX(prop + coh + sep) + K_ALT (z_home - z) z_hat
    w_i = stubbornness (anchors), W_GNSS (GNSS drones), 0 (followers). A_LEAK < 1 is a slow velocity leak
    (memory ~ 0.1 / (1 - A_LEAK) s): without it a leaderless group coasts forever after the informed stop.
    Informed drones also cap |prop| <= V_CRUISE clip(d_goal / R, 0, 1), R = R_SLOW while approaching faster
    than 0.3 m/s, else R_HOLD: they brake gently toward their believed goal, then hold it and hover there.
    The goal term is horizontal; altitude is a barometric hold to home z (the flock is a flattened 3D blob).
 8. Lost drone (sees nobody): followers fly the last seen mean neighbour velocity plus V_SEEK toward the
    last seen neighbour centroid bearing, for LOST_TIMEOUT s, then hover. Informed drones keep flying their
    own goal term. Positions of the last sighting are not dead-reckoned (bearing kept in world frame).
 9. Rule variants: 'topological' (k nearest visible), 'metric' (all visible within radius_m), 'zones'
    (Couzin-style bands: repulsion < R_REP overrides everything; alignment from R_REP..R_ORI; cohesion
    from R_ORI..range).
10. Adaptive defence (anchor.adaptive, anchor.adaptive_rule): 'anchor_gain' - anchors raise their weight
    w = stubbornness * min(1 + ADAPT_GAIN * dis, ADAPT_CAP), dis = (1 - cos(own goal dir, neighbour mean
    velocity)) / 2; 'follower_filter' - followers weight each neighbour's velocity by clip(cos(v_j, own
    consensus heading), 0, 1)^2 (neighbours slower than 0.3 m/s keep weight 1).
11. Baselines (scenario key 'baseline'): 'none' (the flock), 'own_gnss' (no flocking; GNSS drones and anchors
    fly their own navigation, followers hold; vision separation only), 'radio_leader' / 'radio_flock'
    (radio.available: neighbour states by radio - exact, no range/occlusion limit, k nearest - and a
    shared goal term: the leader's (drone 0) or the weighted mean of all informed drones' goal directions;
    radio jammed: radio.fallback = hold | own_gnss | return_home).
12. No collisions, obstacles or terrain; drones are points for dynamics, BODY_R spheres for occlusion.
13. Early termination: at the end of any STILL_T-s window (windows back to back from t = 0) in which no alive
    drone moved more than STILL_V * STILL_T net (mean speed < STILL_V; hover jitter averages out), not before
    STILL_MIN_T (the flock starts from hover and accelerates slowly).
"""
import numpy as np
import rng
import scenario as sc
import vision
import vehicle
import wind

CTRL_HZ = 10
V_CRUISE, V_MAX = 2.0, 3.0      # m/s; Crazyflie-class cruise (well under its flight envelope)
D0 = 2.5                        # m, preferred spacing (10 cm drones, 4-15 m vision)
K_SEP, K_COH, V_COH = 0.8, 0.3, 2.5
C_DEAD = 1.0                    # m, cohesion deadband on the centroid offset
K_ALT = 0.5                     # 1/s, baro altitude hold
W_GNSS = 1.0                    # goal weight of GNSS drones (fixed; tuned in Phase 0)
R_SLOW = 100.0                  # m, informed propulsion capped at V_CRUISE * d / R_SLOW while approaching their goal
R_HOLD = 25.0                   # m, ... and at V_CRUISE * d / R_HOLD once slow (< 0.3 m/s toward it) or past it
V_SEEK, LOST_TIMEOUT = 1.0, 30.0
W_SELF = 1.0                    # weight of own velocity in the alignment average
A_LEAK = 0.998                  # alignment leak per control tick (velocity memory ~50 s without input)
V_UP, V_DOWN = 0.2, 0.2         # m/s, propulsion cap: min(mean neighbour speed + V_UP, slowest + V_DOWN)
BELIEF_TAU = 2.0                # s, follower belief low-pass
R_REP, R_ORI = 1.0, 5.0         # m, zones rule bands
ADAPT_GAIN, ADAPT_CAP = 4.0, 4.0
ARRIVE_R, HOME_R = 20.0, 20.0
STILL_V, STILL_T, STILL_MIN_T = 0.2, 10.0, 120.0
ROLE_NAMES = ("anchor", "gnss", "follower")

_DEFAULTS = {k: v for k, v in dict(globals()).items() if k.isupper() and isinstance(v, (int, float)) and not isinstance(v, bool)}
_attrition_mask = getattr(wind, "attrition_mask", None) or getattr(vehicle, "attrition_mask")


def _unit(x):
    s = np.sqrt((x * x).sum(-1, keepdims=True))
    return np.where(s > 1e-12, x / np.where(s > 1e-12, s, 1.0), 0.0)


def _sat(x, m):
    s = np.sqrt((x * x).sum(-1, keepdims=True))
    return x * np.minimum(1.0, m / np.maximum(s, 1e-12))


def components(adj):
    """Connected components of an undirected bool adjacency (BFS in index order; JS-portable)."""
    n, lab, c = len(adj), -np.ones(len(adj), int), 0
    for s in range(n):
        if lab[s] >= 0:
            continue
        lab[s], stack = c, [s]
        while stack:
            i = stack.pop()
            for j in np.flatnonzero(adj[i] & (lab < 0)):
                lab[j] = c; stack.append(j)
        c += 1
    return c, lab


class Flock:
    def __init__(self, scenario):
        s = self.s = scenario
        # optional constant overrides for the tuning-sensitivity grid (scenario key "model"); the module
        # constants are reset first, so one process can run many scenarios in sequence.
        # ponytail: module-global swap, one Flock per process at a time; make constants per-instance if that ever changes
        g_ = globals()
        g_.update(_DEFAULTS)
        for k_, v_ in s.get("model", {}).items():
            assert k_ in _DEFAULTS, f"unknown model constant {k_}"
            g_[k_] = v_
        f = s["flock"]
        n = self.n = f["n"]
        self.seed, self.dt = s["seed"], s["mission"]["dt_s"]
        self.every = max(1, int(round(1.0 / (CTRL_HZ * self.dt))))
        self.dt_ctrl = self.every * self.dt
        self.role = np.full(n, 2)
        self.role[:f["anchors"]] = 0
        self.role[f["anchors"]:f["anchors"] + f["gnss"]] = 1
        self.goal, self.home = np.array(s["goal"], float), np.array(s["home"], float)
        self.fake_goal = np.array(sc.fake_goal(s), float)
        self.baseline = s.get("baseline", "none")
        # radio leader: drone 0 ('first', an anchor when anchors > 0) or the first GNSS drone ('gnss')
        self.leader = f["anchors"] if s["radio"].get("leader", "first") == "gnss" and f["gnss"] > 0 else 0
        self.adaptive = s["anchor"].get("adaptive", False)
        self.adaptive_rule = s["anchor"].get("adaptive_rule", "anchor_gain")
        self.w = np.where(self.role == 0, s["anchor"]["stubbornness"], np.where(self.role == 1, W_GNSS, 0.0))
        # launch blob: uniform disc of radius D0 sqrt(n/pi) (mean spacing ~D0), z jitter +-1 m
        u = rng.uniform(self.seed, rng.S_INIT, 0, np.arange(3 * n)).reshape(n, 3)
        r0 = D0 * np.sqrt(n / np.pi) * np.sqrt(u[:, 0])
        th = 2 * np.pi * u[:, 1]
        p0 = self.home + np.stack([r0 * np.cos(th), r0 * np.sin(th), 2.0 * (u[:, 2] - 0.5)], 1)
        self.veh = vehicle.Vehicle(n)
        self.veh.reset(p0, np.zeros((n, 3)))
        self.p0 = p0.copy()
        self.wind = wind.Wind(n, s["wind"], self.seed, self.dt)
        self.tracker = vision.Tracker(n, self.dt_ctrl)
        self.k, self.tick, self.t = 0, 0, 0.0
        self.alive = np.ones(n, bool)
        self.v_des = np.zeros((n, 3))
        self.heading = np.tile([1.0, 0, 0], (n, 1))
        self.belief = np.zeros((n, 3))
        self.last_bear, self.last_vbar = np.zeros((n, 3)), np.zeros((n, 3))
        self.t_seen = np.zeros(n)
        self.visible = np.zeros((n, n), bool)
        self.used = np.zeros((n, n), bool)
        self.p_mark, self.still, self.done = p0.copy(), False, False

    # --- what each drone believes -------------------------------------------------------------
    def spoof_s(self):
        sp = self.s["spoof"]
        if not sp["on"] or self.t < sp["start_s"]:
            return np.zeros(3)
        ramp = 1.0 if sp.get("ramp_s", 0) <= 0 else min(1.0, (self.t - sp["start_s"]) / sp["ramp_s"])
        return -np.array(sp["offset_m"], float) * ramp

    def goal_terms(self, p, target=None):
        """(unit horizontal goal direction (n,3), fade (n,)) from each drone's own belief.
        Anchors: true position; GNSS: spoofed position; followers: zeros. target overrides goal (home)."""
        tgt = self.goal if target is None else target
        p_hat = p + np.where((self.role == 1)[:, None], self.spoof_s(), 0.0)
        d = tgt - p_hat
        d[:, 2] = 0.0
        dist = np.sqrt((d * d).sum(1))
        g = _unit(d) * (self.role < 2)[:, None]
        return g, np.clip(dist / R_SLOW, 0.0, 1.0)

    # --- 10 Hz perception + control ----------------------------------------------------------
    def control(self):
        n, p, v = self.n, self.veh.p, self.veh.v
        a = _attrition_mask(n, self.s["attrition"], self.seed, self.t)
        self.alive &= a
        sp = np.sqrt((v[:, :2] ** 2).sum(1))
        self.heading = np.where((sp > vision.MIN_SPEED)[:, None], _unit(v * [1, 1, 0]), self.heading)
        obs = vision.perceive(p, v, self.alive, self.s["sensing"], self.seed, self.tick, heading=self.heading)
        self.visible = obs["visible"]
        # track in each drone's odometry frame (own displacement from VIO), so own accelerations do not leak
        # into neighbour velocity estimates (tracking raw rel with own_vel added lags and self-excites)
        odo = p - self.p0
        _, vel_nb = self.tracker.update(obs["visible"], obs["rel"] + odo[:, None, :], np.zeros((n, 3)))
        rel = obs["rel"]                                       # spacing terms: raw detections
        radio = self.baseline in ("radio_leader", "radio_flock")
        if radio and self.s["radio"]["available"]:            # exact radio states, all alive drones
            rel = p[None, :, :] - p[:, None, :]
            vel_nb = np.broadcast_to(v[None], (n, n, 3)).copy()
            mask = self.alive[:, None] & self.alive[None, :] & ~np.eye(n, dtype=bool)
        else:
            mask = obs["visible"] & np.isfinite(rel[..., 0])
        dist = np.where(mask, np.sqrt(np.nansum(rel * rel, -1)), np.inf)
        rel = np.where(mask[..., None], rel, 0.0)
        vel_nb = np.where(mask[..., None], vel_nb, 0.0)

        # neighbour selection
        rule = self.s["flock"]["rule"]
        if rule == "topological" or radio:
            k = min(self.s["flock"]["k"], n - 1)
            order = np.argsort(dist, axis=1, kind="stable")[:, :k]   # stable: ties -> lower index first
            U = np.zeros((n, n), bool)
            np.put_along_axis(U, order, True, axis=1)
            U &= mask
        elif rule == "metric":
            U = mask & (dist <= self.s["flock"]["radius_m"])
        else:
            U = mask
        self.used = U
        cnt = U.sum(1)
        d = np.where(U, dist, 1.0)
        ur = rel / d[..., None]
        if rule == "zones" and not radio:
            rep = U & (dist < R_REP)
            ori = U & (dist >= R_REP) & (dist < R_ORI)
            att = U & (dist >= R_ORI)
            any_rep = rep.any(1)
            ori &= ~any_rep[:, None]; att &= ~any_rep[:, None]
            sep = -(np.where(rep, K_SEP * (D0 - d), 0.0)[..., None] * ur).sum(1)
            catt = att.sum(1)
            coh = _sat(K_COH * (att[..., None] * rel).sum(1) / np.maximum(catt, 1)[:, None], V_COH)
            A = ori
        else:
            sep = -(np.where(U & (d < D0), K_SEP * (D0 - d), 0.0)[..., None] * ur).sum(1)
            c = (U[..., None] * rel).sum(1) / np.maximum(cnt, 1)[:, None]
            cn = np.sqrt((c * c).sum(1, keepdims=True))
            coh = _sat(K_COH * np.maximum(cn - C_DEAD, 0.0) * _unit(c), V_COH)   # deadband: no drag on edge drones
            A = U

        # alignment weights (follower_filter down-weights neighbours against own consensus)
        wa = A.astype(float)
        if self.adaptive and self.adaptive_rule == "follower_filter":
            sj = np.sqrt((vel_nb * vel_nb).sum(-1))
            c = (_unit(vel_nb) * _unit(self.belief)[:, None, :]).sum(-1)
            filt = np.where(sj > 0.3, np.clip(c, 0.0, 1.0) ** 2, 1.0)
            has_b = (np.abs(self.belief).sum(1) > 0) & (self.role == 2)
            wa = np.where(has_b[:, None], wa * filt, wa)
        vsum = (wa[..., None] * vel_nb).sum(1)
        vbar = (W_SELF * v + vsum) / (W_SELF + wa.sum(1))[:, None]
        vnb = vsum / np.maximum(wa.sum(1), 1e-12)[:, None]           # neighbours only

        g, fade = self.goal_terms(p)
        w = self.w.copy()
        if self.adaptive and self.adaptive_rule == "anchor_gain":
            snb = np.sqrt((vnb * vnb).sum(1))
            dis = np.where(snb > 0.3, 0.5 * (1 - (g * _unit(vnb)).sum(1)), 0.0)
            w = np.where(self.role == 0, w * np.minimum(1 + ADAPT_GAIN * dis, ADAPT_CAP), w)
        gt = w[:, None] * g
        # informed brake toward their believed goal: gently while approaching (the braking wave must cross
        # the flock), stiffly once slow or past it (so followers' momentum cannot drag them off the goal)
        approach = (v * g).sum(1) > 0.3
        dg = fade * R_SLOW
        brake = np.where(self.role < 2, V_CRUISE * np.minimum(1.0, dg / np.where(approach, R_SLOW, R_HOLD)), np.inf)

        if radio and self.s["radio"]["available"]:
            inf_ = (self.role < 2) & self.alive
            if self.baseline == "radio_leader":
                L = self.leader
                shared, sf = (g[L], fade[L]) if self.alive[L] else (np.zeros(3), 0.0)
            else:
                wsum = (self.w * inf_).sum()
                shared = ((self.w * inf_)[:, None] * g).sum(0) / wsum if wsum > 0 else np.zeros(3)
                sf = (self.w * inf_ * fade).sum() / wsum if wsum > 0 else 0.0
            gt, brake = np.tile(shared, (n, 1)), np.full(n, V_CRUISE * sf)
        h = A_LEAK * vbar / V_CRUISE + gt                             # gt in units of V_CRUISE
        hn = np.sqrt((h * h).sum(1, keepdims=True))
        prop = V_CRUISE * h / np.maximum(1.0, hn)
        # speed matching: accelerate gently above the neighbours' mean speed, never far above the slowest
        spd = np.sqrt((vel_nb * vel_nb).sum(-1))
        s_min = np.where(U, spd, np.inf).min(1)                          # slowest used neighbour
        s_mean = (U * spd).sum(1) / np.maximum(cnt, 1)
        cap = np.minimum(np.where(cnt > 0, np.minimum(s_mean + V_UP, s_min + V_DOWN), V_MAX), brake)
        vd = _sat(prop, cap[:, None]) + coh + sep

        # lost followers: chase the last sighting, then hover
        seen = cnt > 0
        self.last_bear = np.where(seen[:, None], _unit((U[..., None] * rel).sum(1)), self.last_bear)
        self.last_vbar = np.where(seen[:, None], vnb, self.last_vbar)
        self.t_seen = np.where(seen, self.t, self.t_seen)
        lost = ~seen & (self.role == 2)
        chase = self.last_vbar + V_SEEK * self.last_bear
        hover_or_chase = np.where(((self.t - self.t_seen) < LOST_TIMEOUT)[:, None], chase, 0.0)
        vd = np.where(lost[:, None], hover_or_chase, vd)

        # baselines and radio fallbacks override the flock
        mode = self.baseline
        if mode in ("radio_leader", "radio_flock") and not self.s["radio"]["available"]:
            mode = self.s["radio"]["fallback"]
        if mode in ("own_gnss", "hold", "return_home"):
            if mode == "own_gnss":
                gg, ff = g, fade
            elif mode == "return_home":
                gg, ff = self.goal_terms(p, self.home)
            else:
                gg, ff = np.zeros((n, 3)), np.zeros(n)
            vd = V_CRUISE * ff[:, None] * gg + sep          # followers/hold: zero + vision separation
        vd = _sat(vd, V_MAX)
        vd[:, 2] += K_ALT * (self.home[2] - p[:, 2])
        self.v_des = np.where(self.alive[:, None], vd, 0.0)

        # beliefs for the mind-reader panel
        lp = self.dt_ctrl / BELIEF_TAU
        fb = (1 - lp) * self.belief + lp * _unit(vd * [1, 1, 0])
        self.belief = np.where((self.role < 2)[:, None], g, fb)
        self.tick += 1

    # --- stepwise API -------------------------------------------------------------------------
    def step(self):
        if self.k % self.every == 0:
            self.control()
        wv = self.wind.step(self.k, self.veh.p)
        a_cmd = vehicle.velocity_controller(self.v_des, self.veh.v, self.veh.params)
        self.veh.step(a_cmd, wv, self.alive, self.dt)
        self.k += 1
        self.t = self.k * self.dt
        kw = int(round(STILL_T / self.dt))
        if self.k % kw == 0:          # consecutive STILL_T windows: stop if nobody alive moved > STILL_V*STILL_T net
            moved = np.sqrt(((self.veh.p - self.p_mark) ** 2).sum(1))[self.alive]
            self.still = self.t >= STILL_MIN_T and (moved < STILL_V * STILL_T).all()
            self.p_mark = self.veh.p.copy()
        self.done = self.still or self.t >= self.s["mission"]["max_time_s"] - 1e-9

    def metrics(self):
        p, al = self.veh.p, self.alive
        dg = np.sqrt(((p - self.goal) ** 2).sum(1))
        df = np.sqrt(((p - self.fake_goal) ** 2).sum(1))
        dh = np.sqrt(((p - self.home) ** 2).sum(1))
        arr, cap, hom = dg <= ARRIVE_R, df < dg, dh <= HOME_R
        frac = lambda x, m: float(x[m].mean()) if m.any() else float("nan")
        out = {"arrival_frac": frac(arr, al), "capture_frac": frac(cap, al), "return_home_frac": frac(hom, al),
               "arrival_by_role": {}, "capture_by_role": {}}
        for r, name in enumerate(ROLE_NAMES):
            out["arrival_by_role"][name] = frac(arr, al & (self.role == r))
            out["capture_by_role"][name] = frac(cap, al & (self.role == r))
        ns = al & (self.role != 1)
        out["arrival_nonspoofed"], out["capture_nonspoofed"] = frac(arr, ns), frac(cap, ns)
        adj = (self.visible | self.visible.T)[np.ix_(al, al)]
        out["fragments"] = components(adj)[0] if al.any() else 0
        pa = p[al]
        if len(pa) > 1:
            dd = np.sqrt(((pa[None] - pa[:, None]) ** 2).sum(-1)); np.fill_diagonal(dd, np.inf)
            out["mean_nn_dist_m"] = float(dd.min(1).mean())
        else:
            out["mean_nn_dist_m"] = float("nan")
        va = self.veh.v[al]
        out["order"] = float(np.sqrt((_unit(va).mean(0) ** 2).sum())) if len(va) else float("nan")
        out["time_s"] = float(self.t)
        return out

    def state(self):
        e = np.argwhere(self.used)
        return {"t": self.t, "n": self.n, "pos": self.veh.p.copy(), "vel": self.veh.v.copy(), "role": self.role.copy(),
                "alive": self.alive.copy(), "belief": _unit(self.belief), "edges": e, "goal": self.goal.copy(),
                "fake_goal": self.fake_goal.copy(), "home": self.home.copy(), "metrics": self.metrics()}


def simulate(scenario, record_every=None):
    """Run to max_time_s (or early stillness stop). record_every: seconds between trajectory frames."""
    fl = Flock(scenario)
    rec_k = None if record_every is None else max(1, int(round(record_every / fl.dt)))
    traj = {"t": [], "pos": [], "alive": [], "belief": []}
    while not fl.done:
        if rec_k and fl.k % rec_k == 0:
            traj["t"].append(fl.t); traj["pos"].append(fl.veh.p.copy())
            traj["alive"].append(fl.alive.copy()); traj["belief"].append(_unit(fl.belief))
        fl.step()
    out = fl.metrics()
    if rec_k:
        out["traj"] = {"t": np.array(traj["t"]), "pos": np.array(traj["pos"]), "role": fl.role.copy(),
                       "alive": np.array(traj["alive"]), "belief": np.array(traj["belief"])}
    return out


def record(scenario, out, split="train", **extra):
    """Contract 2 run-record row (plus per-role columns)."""
    import subprocess, datetime
    f = scenario["flock"]
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "nogit"
    except Exception:
        git = "nogit"
    row = {"scenario_hash": sc.scenario_hash(scenario), "seed": scenario["seed"], "split": split, "n": f["n"],
           "anchors": f["anchors"], "gnss": f["gnss"], "followers": f["followers"],
           "stubbornness": scenario["anchor"]["stubbornness"], "adaptive": bool(scenario["anchor"]["adaptive"]),
           "rule": f["rule"], "range_m": scenario["sensing"]["range_m"], "occlusion": bool(scenario["sensing"]["occlusion"]),
           "radio": bool(scenario["radio"]["available"]), "fallback": scenario["radio"]["fallback"],
           "arrival_frac": out["arrival_frac"], "capture_frac": out["capture_frac"], "fragments": out["fragments"],
           "mean_nn_dist_m": out["mean_nn_dist_m"], "return_home_frac": out["return_home_frac"], "time_s": out["time_s"],
           "git": git, "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "spoof_on": bool(scenario["spoof"]["on"]), "offset_m": float(np.linalg.norm(scenario["spoof"]["offset_m"])),
           "baseline": scenario.get("baseline", "none"), "arrival_nonspoofed": out["arrival_nonspoofed"],
           "capture_nonspoofed": out["capture_nonspoofed"], "order": out["order"]}
    for r in ROLE_NAMES:
        row[f"arrival_{r}"] = out["arrival_by_role"][r]
        row[f"capture_{r}"] = out["capture_by_role"][r]
    row.update(extra)
    return row


if __name__ == "__main__":
    import time
    mk = sc.make
    # determinism
    s = mk(seed=1, flock={"n": 20, "anchors": 3, "gnss": 5}, mission={"max_time_s": 60.0})
    a, b = simulate(s, record_every=1.0), simulate(s, record_every=1.0)
    assert np.array_equal(a["traj"]["pos"], b["traj"]["pos"]) and a["capture_frac"] == b["capture_frac"]
    # role counts
    fl = Flock(mk(flock={"n": 30, "anchors": 3, "gnss": 10}))
    assert list(np.bincount(fl.role, minlength=3)) == [3, 10, 17]
    # spoof off -> fake goal irrelevant
    s1 = mk(seed=2, flock={"n": 20, "anchors": 2, "gnss": 4}, spoof={"on": False}, mission={"max_time_s": 40.0})
    s2 = sc.merge(s1, {"spoof": {"offset_m": [0.0, -400.0, 0.0]}})
    assert np.array_equal(simulate(s1, 1.0)["traj"]["pos"], simulate(s2, 1.0)["traj"]["pos"])
    # no-spoof N=20 flock with 5 anchors arrives
    r = simulate(mk(seed=3, flock={"n": 20, "anchors": 5, "gnss": 0}, spoof={"on": False}))
    assert r["arrival_frac"] >= 0.9, r
    # own_gnss baseline: all GNSS, no spoof -> arrives 1.00; spoofed -> captured 1.00
    base = dict(flock={"n": 20, "anchors": 0, "gnss": 20}, baseline="own_gnss")
    r0 = simulate(mk(seed=4, spoof={"on": False}, **base)); r1 = simulate(mk(seed=4, **base))
    assert r0["arrival_frac"] == 1.0 and r1["capture_frac"] == 1.0, (r0, r1)
    # every rule / baseline / adaptive option runs and stays finite (30 s smoke runs)
    short = dict(flock={"n": 20, "anchors": 3, "gnss": 5}, mission={"max_time_s": 30.0})
    for extra in [dict(flock={"rule": "metric"}), dict(flock={"rule": "zones"}), dict(sensing={"fov_deg": 120.0}),
                  dict(anchor={"adaptive": True, "adaptive_rule": "anchor_gain"}),
                  dict(anchor={"adaptive": True, "adaptive_rule": "follower_filter"}),
                  dict(baseline="radio_flock", radio={"available": True}), dict(baseline="radio_leader", radio={"available": True}),
                  dict(baseline="radio_leader", radio={"fallback": "hold"}), dict(baseline="radio_leader", radio={"fallback": "return_home"}),
                  dict(attrition={"fraction": 0.3, "t_s": 10.0}), dict(wind={"mean_mps": [1.0, 0.0, 0.0], "turbulence": "moderate"})]:
        fl = Flock(sc.merge(mk(seed=5, **short), extra))
        while not fl.done:
            fl.step()
        st = fl.state()
        assert np.isfinite(st["pos"]).all() and st["edges"].shape[1] == 2, extra
    assert fl.alive.sum() == 20  # wind run: nobody dies
    print("flock self-check OK:",{k: r[k] for k in ("arrival_frac", "fragments", "mean_nn_dist_m", "time_s")})
    for n in (20, 50, 100):
        s = mk(seed=0, flock={"n": n, "anchors": 3, "gnss": 10}, spoof={"on": False})
        t0 = time.perf_counter(); r = simulate(s); el = time.perf_counter() - t0
        print(f"  timing N={n}: {el:.2f} s wall for {r['time_s']:.0f} s simulated "
              f"({el / r['time_s'] * 600:.2f} s per 600 s mission); arrival {r['arrival_frac']:.2f}")
