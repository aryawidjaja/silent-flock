"""Pre-registered sweeps (Phase 2). Each grid is a list of jobs (cell, scenario, split); a shard runs every
job with index % n_shards == shard and writes results/runs/<grid>__<shard>.parquet (Contract 2 rows).
Values fixed in Phase 0/1 (anchor stubbornness, adaptive rule, ...) come from results/sweep_config.json,
frozen before any test seed is run.
Run:  uv run python scripts/sweep.py <grid> [--shard i --n-shards k] [--seeds a:b] [--dry]
      uv run python scripts/sweep.py --selfcheck"""
import argparse, json, os, platform, sys, time
from multiprocessing import Pool
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flock
import scenario as sc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_PATH = os.path.join(ROOT, "results", "sweep_config.json")
TEST = list(range(1000, 1200))     # held-out seeds (pre-registered)
TRAIN = list(range(0, 100))


def cfg():
    with open(CFG_PATH) as f:
        return json.load(f)


def off(d):
    return [0.0, float(d), 0.0]


def h1(c):
    """Capture vs followers: followers x offsets x anchor counts, plus spoof-off and no-anchor controls."""
    jobs, w = [], c["stubbornness_h1"]
    for a in c["h1_anchors"]:
        for u in c["h1_followers"]:
            base = dict(flock={"anchors": a, "gnss": c["h1_gnss"], "followers": u, "n": a + c["h1_gnss"] + u},
                        anchor={"stubbornness": w})
            for d in c["h1_offsets"]:
                jobs.append((f"a{a}_u{u}_d{d}", sc.merge(base, {"spoof": {"offset_m": off(d)}})))
            jobs.append((f"a{a}_u{u}_nospoof", sc.merge(base, {"spoof": {"on": False}})))
    for u in c["h1_followers"]:            # no-anchor control: the spoofed drones alone with followers
        jobs.append((f"a0_u{u}_d200", dict(flock={"anchors": 0, "gnss": c["h1_gnss"], "followers": u,
                                                   "n": c["h1_gnss"] + u}, spoof={"offset_m": off(200)})))
    return jobs


def h2(c):
    """omega* grid: anchors x spoofed x N x stubbornness (offset 200 m). Infeasible cells skipped."""
    jobs = []
    for n in c["h2_n"]:
        for a in c["h2_anchors"]:
            for g in c["h2_spoofed"]:
                if a + g > n:
                    continue
                for w in c["h2_omega"]:
                    jobs.append((f"n{n}_a{a}_s{g}_w{w}", dict(flock={"n": n, "anchors": a, "gnss": g, "followers": n - a - g},
                                                              anchor={"stubbornness": w})))
    return jobs


def h3(c):
    """Adaptive vs fixed: 3 anchors vs 20 spoofed, N = 50, plus spoof-off controls."""
    jobs = []
    for ad in (False, True):
        for on in (True, False):
            jobs.append((f"adaptive{int(ad)}_spoof{int(on)}",
                         dict(flock={"n": 50, "anchors": 3, "gnss": 20, "followers": 27},
                              anchor={"stubbornness": c["stubbornness_h3"], "adaptive": ad, "adaptive_rule": c["adaptive_rule"]},
                              spoof={"on": on})))
    return jobs


def robust(c):
    """H1 key contrast (followers lo vs hi) under each perturbation; H3 adaptive under the same."""
    pert = {"ref": {}, "wind_shared": {"wind": {"mean_mps": [0.0, 1.0, 0.0], "turbulence": "light", "shared_fraction": 1.0}},
            "gust_each": {"wind": {"turbulence": "light", "shared_fraction": 0.0}},
            "attrition30": {"attrition": {"fraction": 0.3, "t_s": c["attrition_t_s"]}},
            "occl_off": {"sensing": {"occlusion": False}}, "fov120": {"sensing": {"fov_deg": 120.0}}}
    for r in (4.0, 6.0, 15.0):
        pert[f"range{int(r)}"] = {"sensing": {"range_m": r}}
    jobs = []
    for name, p in pert.items():
        for u in c["robust_followers"]:
            base = dict(flock={"anchors": 3, "gnss": 10, "followers": u, "n": 13 + u}, anchor={"stubbornness": c["stubbornness_h1"]})
            jobs.append((f"{name}_u{u}", sc.merge(base, p)))
            jobs.append((f"{name}_u{u}_nospoof", sc.merge(base, sc.merge(p, {"spoof": {"on": False}}))))
        jobs.append((f"{name}_h3adaptive", sc.merge(dict(flock={"n": 50, "anchors": 3, "gnss": 20, "followers": 27},
                                                         anchor={"stubbornness": c["stubbornness_h3"], "adaptive": True,
                                                                 "adaptive_rule": c["adaptive_rule"]}), p)))
    return jobs


def baselines(c):
    """Non-flock and radio baselines at the H1 reference cell (3 anchors, 10 GNSS, 37 followers)."""
    f = {"anchors": 3, "gnss": 10, "followers": 37, "n": 50}
    jobs = []
    for on in (True, False):
        tag = f"spoof{int(on)}"
        jobs.append((f"own_gnss_{tag}", dict(flock=f, baseline="own_gnss", spoof={"on": on})))
        jobs.append((f"radio_flock_{tag}", dict(flock=f, baseline="radio_flock", radio={"available": True}, spoof={"on": on})))
        for fb in ("hold", "own_gnss", "return_home"):
            jobs.append((f"radio_leader_jam_{fb}_{tag}", dict(flock=f, baseline="radio_leader",
                                                               radio={"available": False, "fallback": fb}, spoof={"on": on})))
        jobs.append((f"radio_leader_ok_{tag}", dict(flock=f, baseline="radio_leader", radio={"available": True}, spoof={"on": on})))
        jobs.append((f"flock_{tag}", dict(flock=f, anchor={"stubbornness": c["stubbornness_h1"]}, spoof={"on": on})))
    return jobs


TUNINGS = {"ref": {}, "kcoh06": {"K_COH": 0.6}, "kcoh015": {"K_COH": 0.15}, "d0_4": {"D0": 4.0}, "cdead05": {"C_DEAD": 0.5}}


def sens(c):
    """Amendment 2: is the H1 outcome a property of the Phase 0 tuning? H1 reference cell under alternative
    flock-rule constants, with the no-spoof gate checked per tuning."""
    jobs = []
    for name, m in TUNINGS.items():
        for u in (5, 17, 30):
            base = dict(flock={"anchors": 3, "gnss": 10, "followers": u, "n": 13 + u}, anchor={"stubbornness": c["stubbornness_h1"]}, model=m)
            jobs.append((f"{name}_u{u}", base))
            jobs.append((f"{name}_u{u}_nospoof", sc.merge(base, {"spoof": {"on": False}})))
    return jobs


def h2ctl(c):
    """Amendment 2: spoof-off control for every H2 composition (omega = 2)."""
    jobs = []
    for n in c["h2_n"]:
        for a in c["h2_anchors"]:
            for g in c["h2_spoofed"]:
                if a + g <= n:
                    jobs.append((f"n{n}_a{a}_s{g}_nospoof", dict(flock={"n": n, "anchors": a, "gnss": g, "followers": n - a - g},
                                                                anchor={"stubbornness": 2.0}, spoof={"on": False})))
    return jobs


def extra(c):
    """Amendment 2: baselines the first grid lacked (all-GNSS own-navigation fallback, no-anchor flock,
    GNSS-led radio leader, radio flock at the flock's omega) and H3's no-anchor / all-honest controls."""
    f = {"anchors": 3, "gnss": 10, "followers": 37, "n": 50}
    w = {"stubbornness": c["stubbornness_h1"]}
    jobs = []
    for on in (True, False):
        t = f"spoof{int(on)}"
        jobs.append((f"own_gnss_all_{t}", dict(flock={"anchors": 0, "gnss": 50, "followers": 0, "n": 50}, baseline="own_gnss", spoof={"on": on})))
        jobs.append((f"noanchor_flock_{t}", dict(flock={"anchors": 0, "gnss": 10, "followers": 40, "n": 50}, spoof={"on": on})))
        jobs.append((f"radio_flock_w2_{t}", dict(flock=f, anchor=w, baseline="radio_flock", radio={"available": True}, spoof={"on": on})))
        jobs.append((f"radio_leader_gnss_ok_{t}", dict(flock=f, anchor=w, baseline="radio_leader", radio={"available": True, "leader": "gnss"}, spoof={"on": on})))
        for fb in ("hold", "own_gnss", "return_home"):
            jobs.append((f"radio_leader_gnss_jam_{fb}_{t}", dict(flock=f, anchor=w, baseline="radio_leader",
                                                                  radio={"available": False, "fallback": fb, "leader": "gnss"}, spoof={"on": on})))
    jobs.append(("h3_noanchor", dict(flock={"n": 50, "anchors": 0, "gnss": 20, "followers": 30})))
    jobs.append(("h3_allhonest", dict(flock={"n": 50, "anchors": 0, "gnss": 23, "followers": 27}, spoof={"on": False})))
    return jobs


def robust2(c):
    """Amendment 2: perturbations the first robustness grid lacked: detection noise 0.1 / 1.0 m, wind
    opposite to the spoof direction, and the 4 m vision case with a 1,200 s horizon."""
    pert = {"noise01": {"sensing": {"pos_noise_m": 0.1}}, "noise10": {"sensing": {"pos_noise_m": 1.0}},
            "wind_opp": {"wind": {"mean_mps": [0.0, -1.0, 0.0], "turbulence": "light", "shared_fraction": 1.0}}}
    jobs = []
    for name, p in pert.items():
        for u in c["robust_followers"]:
            base = dict(flock={"anchors": 3, "gnss": 10, "followers": u, "n": 13 + u}, anchor={"stubbornness": c["stubbornness_h1"]})
            jobs.append((f"{name}_u{u}", sc.merge(base, p)))
            jobs.append((f"{name}_u{u}_nospoof", sc.merge(base, sc.merge(p, {"spoof": {"on": False}}))))
    long4 = {"sensing": {"range_m": 4.0}, "mission": {"max_time_s": 1200.0}}
    jobs.append(("range4long_a5_nospoof", sc.merge(dict(flock={"anchors": 5, "gnss": 0, "followers": 45, "n": 50}, spoof={"on": False}), long4)))
    jobs.append(("range4long_u30", sc.merge(dict(flock={"anchors": 3, "gnss": 10, "followers": 30, "n": 43}, anchor={"stubbornness": c["stubbornness_h1"]}), long4)))
    jobs.append(("range4long_u30_nospoof", sc.merge(dict(flock={"anchors": 3, "gnss": 10, "followers": 30, "n": 43}, anchor={"stubbornness": c["stubbornness_h1"]},
                                                         spoof={"on": False}), long4)))
    return jobs


GRIDS = {"h1": h1, "h2": h2, "h3": h3, "robust": robust, "baselines": baselines,
         "sens": sens, "h2ctl": h2ctl, "extra": extra, "robust2": robust2}


def expand(grid, c, seeds, split):
    return [(cell, sc.make(seed=s, **over), split) for cell, over in GRIDS[grid](c) for s in seeds]


def _run(job):
    cell, s, split = job
    t0 = time.perf_counter()
    out = flock.simulate(s)
    return flock.record(s, out, split=split, cell=cell, wall_s=time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid", nargs="?")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--seeds", default=None, help="a:b range (default: the grid's seed set from the config)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    c = cfg()
    if a.seeds:
        lo, hi = map(int, a.seeds.split(":")); seeds = list(range(lo, hi))
    else:
        seeds = c["seeds"][a.grid]
        seeds = list(range(*seeds)) if isinstance(seeds, list) and len(seeds) == 2 else seeds
    jobs = expand(a.grid, c, seeds, a.split)
    mine = jobs[a.shard::a.n_shards]
    print(f"{a.grid}: {len(jobs)} jobs total, shard {a.shard}/{a.n_shards} runs {len(mine)}", flush=True)
    if a.dry:
        return
    t0 = time.perf_counter()
    with Pool(a.procs) as p:
        rows = []
        for i, r in enumerate(p.imap_unordered(_run, mine, chunksize=1)):
            rows.append(r)
            if (i + 1) % max(1, len(mine) // 20) == 0:
                print(f"  {i + 1}/{len(mine)}  {time.perf_counter() - t0:.0f} s", flush=True)
    df = pd.DataFrame(rows)
    df["grid"], df["python"], df["numpy"] = a.grid, platform.python_version(), np.__version__
    df["config"] = json.dumps(c, sort_keys=True)
    os.makedirs(os.path.join(ROOT, "results", "runs"), exist_ok=True)
    out = os.path.join(ROOT, "results", "runs", f"{a.grid}__{a.shard:03d}.parquet")
    df.to_parquet(out, index=False)
    print(f"wrote {out}: {len(df)} rows in {time.perf_counter() - t0:.0f} s", flush=True)


def selfcheck():
    c = {"stubbornness_h1": 2.0, "stubbornness_h3": 2.0, "adaptive_rule": "anchor_gain", "attrition_t_s": 100.0,
         "h1_anchors": [3], "h1_gnss": 10, "h1_followers": [0, 17], "h1_offsets": [200],
         "h2_n": [20, 50], "h2_anchors": [3], "h2_spoofed": [10, 20], "h2_omega": [1.0, 2.0], "robust_followers": [0, 17]}
    for g in GRIDS:
        jobs = expand(g, c, [1000, 1001], "test")
        assert jobs and all(sc.validate(s) is None for _, s, _ in jobs), g
        cells = [j[0] for j in jobs]
        assert len(set(cells)) * 2 == len(cells), f"{g}: duplicate cell ids"
    h2j = expand("h2", c, [0], "test")
    assert not any(s["flock"]["n"] == 20 and s["flock"]["gnss"] == 20 for _, s, _ in h2j), "infeasible H2 cell kept"
    a0 = [s for cell, s, _ in expand("h1", c, [0], "test") if cell.startswith("a0_")]
    assert a0 and all(s["flock"]["anchors"] == 0 and s["spoof"]["on"] for s in a0)
    shards = [expand("h1", c, [0, 1], "test")[i::3] for i in range(3)]
    assert sum(map(len, shards)) == len(expand("h1", c, [0, 1], "test"))
    print("sweep self-check OK:", {g: len(expand(g, c, [0], "test")) for g in GRIDS}, "cells per seed (selfcheck config)")


if __name__ == "__main__":
    main()
