"""KT3: does the KT2 follower effect survive shorter vision, and does the flock still hold together?
At vision range 6 m and 4 m (360 deg, occlusion on, 0.3 m noise at 10 m): 3 anchors vs 10 spoofed with followers 0/17/30
at one stubbornness (the KT2 level where anchors win at 0 followers), the same counts with spoof off, and the H0 flock
(N = 50, 5 or 10 informed anchors, spoof off). The 10 m point is KT2/KT1 itself. Training seeds only.
"Realistic vision breaks the flock" if no-spoof arrival < 0.5.
Run: uv run python scripts/kt3_vision.py [n_seeds=30] [stubbornness=2] [ranges=6,4]"""
import sys
import kt_common as kc
import kt2_anchor as k2

FOLLOWERS = (0, 17, 30)

if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    w = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    ranges = [float(r) for r in sys.argv[3].split(",")] if len(sys.argv) > 3 else [6.0, 4.0]
    seeds = list(range(n_seeds))
    cells = []
    for r in ranges:
        sens = {"range_m": r, "fov_deg": 360.0, "occlusion": True, "pos_noise_m": 0.3}
        for tag, kw in k2.build(sens, omegas=(w,), followers=FOLLOWERS, controls=False):
            cells.append((f"R={r:g} {tag}", kw))
        for f in FOLLOWERS:
            cells.append((f"R={r:g} spoof_off w={w:g} f={f}", dict(flock={"n": 13 + f, "anchors": 3, "gnss": 10, "followers": f},
                                                                  spoof={"on": False}, anchor={"stubbornness": w}, sensing=sens)))
        for k in (5, 10):
            cells.append((f"R={r:g} H0 anchors={k}", dict(flock={"n": 50, "anchors": k, "gnss": 0}, spoof={"on": False}, sensing=sens)))
    df, wall = kc.run(cells, seeds, "kt3_vision")
    S = kc.summarize(df, k2.METRICS + ["order"])
    S["tests"] = {}
    for r in ranges:
        get = lambda f, m: df[df.cond == f"R={r:g} w={w:g} f={f}"].set_index("seed")[m]
        S["tests"][f"R={r:g}"] = {
            "capture_nonspoofed f0->f30 (paired diff)": kc.boot_diff(get(0, "capture_nonspoofed"), get(30, "capture_nonspoofed")),
            "capture_nonspoofed f0->f17 (paired diff)": kc.boot_diff(get(0, "capture_nonspoofed"), get(17, "capture_nonspoofed")),
            "capture_follower f17->f30 (paired diff)": kc.boot_diff(get(17, "capture_follower"), get(30, "capture_follower")),
            "no-spoof arrival H0 anchors=5": S[f"R={r:g} H0 anchors=5"]["arrival_frac"],
            "no-spoof arrival H0 anchors=10": S[f"R={r:g} H0 anchors=10"]["arrival_frac"],
        }
    print(f"{'condition':30s} {'capture(fol)':20s} {'capture(anch+fol)':20s} {'arrival(anch+fol)':20s} {'arrival(all)':20s} {'fragments':18s}")
    for tag, _ in cells:
        q = S[tag]
        print(f"{tag:30s} {kc.fmt(q['capture_follower']):20s} {kc.fmt(q['capture_nonspoofed']):20s} {kc.fmt(q['arrival_nonspoofed']):20s} "
              f"{kc.fmt(q['arrival_frac']):20s} {kc.fmt(q['fragments']):18s}")
    for k, v in S["tests"].items():
        print(k, {kk: kc.fmt(vv) for kk, vv in v.items()})
    kc.write("kt3_vision", S, kc.meta(seeds, wall, {"test": "KT3 vision", "ranges_m": ranges, "stubbornness": w,
                                                     "followers": FOLLOWERS, "offset_m": k2.OFFSET}))
    assert len(df) == len(cells) * n_seeds and df.seed.max() < 100
