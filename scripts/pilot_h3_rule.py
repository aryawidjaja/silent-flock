"""Phase 0: choose the H3 adaptive rule on TRAINING seeds (0-19) only, before any test seed.
3 anchors vs 20 spoofed GNSS, N = 50, stubbornness 2. Candidates: fixed (control), anchor_gain, follower_filter.
Rule chosen = higher arrival over non-spoofed drones; written to results/kt/pilot_h3_rule.json.
Run: uv run python scripts/pilot_h3_rule.py"""
import json, os, sys, platform
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flock, scenario as sc

SEEDS = list(range(20))
CONDS = {"fixed": {"adaptive": False}, "anchor_gain": {"adaptive": True, "adaptive_rule": "anchor_gain"},
         "follower_filter": {"adaptive": True, "adaptive_rule": "follower_filter"}}

def job(a):
    name, s = a
    sc_ = sc.make(seed=s, flock={"n": 50, "anchors": 3, "gnss": 20, "followers": 27}, anchor={"stubbornness": 2.0, **CONDS[name]})
    return flock.record(sc_, flock.simulate(sc_), split="train", cond=name)

if __name__ == "__main__":
    with Pool(8) as p:
        rows = p.map(job, [(c, s) for c in CONDS for s in SEEDS])
    out = {"meta": {"seeds": SEEDS, "python": platform.python_version(), "numpy": np.__version__}, "summary": {}}
    for c in CONDS:
        r = [x for x in rows if x["cond"] == c]
        out["summary"][c] = {k: float(np.nanmean([x[k] for x in r])) for k in
                             ("arrival_nonspoofed", "capture_nonspoofed", "capture_follower", "capture_anchor", "arrival_frac", "fragments")}
    cand = {c: out["summary"][c]["arrival_nonspoofed"] for c in ("anchor_gain", "follower_filter")}
    out["chosen"] = max(cand, key=cand.get)
    os.makedirs("results/kt", exist_ok=True)
    json.dump(out, open("results/kt/pilot_h3_rule.json", "w"), indent=1)
    print(json.dumps(out["summary"], indent=1), "\nchosen:", out["chosen"])
    assert len(rows) == 60 and out["chosen"] in cand
