"""KT2 (H1 pilot): 3 anchors vs 10 GNSS drones all spoofed common-mode (200 m north), followers 0/5/10/17/30,
anchor stubbornness 1/2/4. Controls in the same table: spoof off (stubbornness 2), all-honest (no anchors, 13 GNSS
unspoofed), no anchors (10 spoofed + followers); "no followers" is the followers = 0 column. Training seeds only.
Kill H1 if capture does not rise with followers (CI at 0 vs 30 followers overlapping) where anchors win at 0 followers.
Run: uv run python scripts/kt2_anchor.py [n_seeds=50] [n_control_seeds=n_seeds]   (kt3 reuses build())"""
import sys
import numpy as np
import pandas as pd
import kt_common as kc

FOLLOWERS = (0, 5, 10, 17, 30)
OMEGAS = (1.0, 2.0, 4.0)
METRICS = ["capture_follower", "capture_nonspoofed", "capture_anchor", "capture_gnss", "arrival_nonspoofed",
           "arrival_follower", "arrival_anchor", "arrival_frac", "capture_frac", "fragments", "mean_nn_dist_m", "time_s"]
OFFSET = [0.0, 200.0, 0.0]


def build(sensing=None, omegas=OMEGAS, followers=FOLLOWERS, controls=True):
    sens = {"sensing": sensing} if sensing else {}
    sp = {"on": True, "offset_m": OFFSET}
    cell = lambda a, g, f, w, spon: dict(flock={"n": a + g + f, "anchors": a, "gnss": g, "followers": f},
                                         spoof={**sp, "on": spon}, anchor={"stubbornness": w}, **sens)
    cells = [(f"w={w:g} f={f}", cell(3, 10, f, w, True)) for w in omegas for f in followers]
    if controls:
        cells += [(f"spoof_off w=2 f={f}", cell(3, 10, f, 2.0, False)) for f in followers]
        cells += [(f"all_honest g=13 f={f}", cell(0, 13, f, 1.0, False)) for f in followers]
        cells += [(f"no_anchor g=10 f={f}", cell(0, 10, f, 1.0, True)) for f in followers]
    return cells


def slope(df, prefix, metric, followers):
    """Bootstrap (over seeds, paired across follower counts) of the OLS slope of mean metric vs follower count."""
    piv = df[df.cond.str.startswith(prefix)].pivot(index="seed", columns="followers", values=metric)[list(followers)].dropna()
    x, X = np.array(followers, float), piv.values
    idx = np.random.default_rng(2).integers(0, len(X), (kc.B, len(X)))
    s = np.polyfit(x, X[idx].mean(1).T, 1)[0]
    return [float(np.polyfit(x, X.mean(0), 1)[0]), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)), len(X)]


def report(df, cells, omegas=OMEGAS, followers=FOLLOWERS):
    S = kc.summarize(df, METRICS)
    S["tests"] = {}
    f_hi = max(followers)
    for w in omegas:
        p = f"w={w:g} "
        get = lambda f, m: df[df.cond == f"{p}f={f}"].set_index("seed")[m]
        S["tests"][f"w={w:g}"] = {
            "capture_nonspoofed f0->f30 (paired diff)": kc.boot_diff(get(0, "capture_nonspoofed"), get(f_hi, "capture_nonspoofed")),
            "capture_follower f5->f30 (paired diff)": kc.boot_diff(get(5, "capture_follower"), get(f_hi, "capture_follower")),
            "slope capture_nonspoofed per follower": slope(df, p, "capture_nonspoofed", followers),
            "anchors win at f=0 (arrival_anchor)": S[f"{p}f=0"]["arrival_anchor"],
        }
    print(f"{'condition':26s} {'capture(followers)':20s} {'capture(anch+fol)':20s} {'arrival(anch+fol)':20s} {'capture(anchors)':20s} {'fragments':18s}")
    for tag, _ in cells:
        r = S[tag]
        print(f"{tag:26s} {kc.fmt(r['capture_follower']):20s} {kc.fmt(r['capture_nonspoofed']):20s} {kc.fmt(r['arrival_nonspoofed']):20s} "
              f"{kc.fmt(r['capture_anchor']):20s} {kc.fmt(r['fragments']):18s}")
    for k, v in S["tests"].items():
        print(k, {kk: (kc.fmt(vv) if len(vv) >= 3 else vv) for kk, vv in v.items()})
    return S


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 50          # main grid
    n_ctrl = int(sys.argv[2]) if len(sys.argv) > 2 else n_seeds      # controls (fewer seeds to fit the Mac budget)
    seeds = list(range(n_seeds))
    cells = build()
    main, ctrl = cells[:len(OMEGAS) * len(FOLLOWERS)], cells[len(OMEGAS) * len(FOLLOWERS):]
    df1, w1 = kc.run(main, seeds, "kt2_anchor")
    df2, w2 = kc.run(ctrl, list(range(n_ctrl)), "kt2_anchor_controls")
    df, wall = pd.concat([df1, df2], ignore_index=True), w1 + w2
    df.to_parquet(f"{kc.ROOT}/results/runs/kt2_anchor.parquet", index=False)
    import os; os.remove(f"{kc.ROOT}/results/runs/kt2_anchor_controls.parquet")
    S = report(df, cells)
    kc.write("kt2_anchor", S, kc.meta(seeds, wall, {"control_seeds": list(range(n_ctrl)), "test": "KT2 / H1 pilot", "anchors": 3, "gnss_spoofed": 10,
                                                     "offset_m": OFFSET, "followers": FOLLOWERS, "stubbornness": OMEGAS}))
    assert len(df) == len(main) * n_seeds + len(ctrl) * n_ctrl and df.seed.max() < 100
