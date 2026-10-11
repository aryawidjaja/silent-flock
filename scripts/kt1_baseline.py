"""KT1 (H0 gate): no spoofing, radio jammed, default sensing (10 m, 360 deg, occlusion, 0.3 m noise), N = 50.
Arrival vs informed count 3/5/10/15 (anchors), plus the same with informed = GNSS drones (spoof off; must match),
plus the own-GNSS baseline (5 and 10 GNSS drones fly alone, followers hold). Training seeds only.
Gate: arrival >= 0.85 at 5 and 10 informed, and flat across 3-15.
Run: uv run python scripts/kt1_baseline.py [n_seeds=50]"""
import sys
import numpy as np
import kt_common as kc

INFORMED = (3, 5, 10, 15)
METRICS = ["arrival_frac", "capture_frac", "fragments", "mean_nn_dist_m", "time_s", "order"]

if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    seeds = list(range(n_seeds))
    off = {"spoof": {"on": False}}
    cells = [(f"anchors={k}", dict(flock={"n": 50, "anchors": k, "gnss": 0}, **off)) for k in INFORMED]
    cells += [(f"gnss={k}", dict(flock={"n": 50, "anchors": 0, "gnss": k}, **off)) for k in (5, 10)]
    cells += [(f"own_gnss gnss={k}", dict(flock={"n": 50, "anchors": 0, "gnss": k}, baseline="own_gnss", **off)) for k in (5, 10)]
    df, wall = kc.run(cells, seeds, "kt1_baseline")
    S = kc.summarize(df, METRICS)
    for k in (5, 10):   # anchors vs GNSS (spoof off) are the same dynamics: identical per seed
        a, g = df[df.cond == f"anchors={k}"].set_index("seed"), df[df.cond == f"gnss={k}"].set_index("seed")
        S[f"gnss={k}"]["identical_to_anchors"] = bool(np.allclose(a.arrival_frac, g.arrival_frac) and np.allclose(a.time_s, g.time_s))
    lo, hi = df[df.cond == "anchors=3"].set_index("seed").arrival_frac, df[df.cond == "anchors=15"].set_index("seed").arrival_frac
    S["flatness_diff_15_minus_3"] = kc.boot_diff(lo, hi)
    gate = all(S[f"anchors={k}"]["arrival_frac"][0] >= 0.85 for k in (5, 10))
    flat = all(S[f"anchors={k}"]["arrival_frac"][1] >= 0.85 - 0.1 for k in INFORMED)
    S["gate"] = {"arrival_ge_0.85_at_5_and_10": gate, "flat_3_to_15_lowerCI_ge_0.75": flat}
    kc.write("kt1_baseline", S, kc.meta(seeds, wall, {"test": "KT1 / H0", "n": 50, "spoof": "off", "radio": "jammed",
                                                       "sensing": {"range_m": 10, "fov_deg": 360, "occlusion": True, "pos_noise_m": 0.3}}))
    print(f"{'condition':22s} {'arrival':22s} {'fragments':22s} {'time_s':22s}")
    for tag, _ in cells:
        r = S[tag]; print(f"{tag:22s} {kc.fmt(r['arrival_frac']):22s} {kc.fmt(r['fragments']):22s} {r['time_s'][0]:.0f}")
    print("15 minus 3 informed (paired):", kc.fmt(S["flatness_diff_15_minus_3"]), "| gate:", S["gate"])
    assert len(df) == len(cells) * n_seeds and df.split.eq("train").all() and df.seed.max() < 100
    assert all(S[f"gnss={k}"]["identical_to_anchors"] for k in (5, 10)), "anchor and unspoofed-GNSS informed should match"
