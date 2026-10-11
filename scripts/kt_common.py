"""Shared helpers for the Phase 1 kill tests (kt1-kt3): run a grid of scenarios over seeds in a process pool,
write Contract 2 rows to results/runs/<name>.parquet and a summary with bootstrap 95% CIs to results/kt/<name>.json.
Run: uv run python scripts/kt_common.py   (self-check of the bootstrap)"""
import json, os, platform, sys, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
import scenario as sc
import flock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = 10_000
PROCS = min(8, os.cpu_count() or 1)


def _job(a):
    tag, s = a
    t0 = time.perf_counter()
    out = flock.simulate(s)
    return flock.record(s, out, split="train", cond=tag, wall_s=time.perf_counter() - t0)


def run(cells, seeds, name, probe=None):
    """cells: list of (tag, scenario-without-seed kwargs dict). Prints a timing extrapolation from a probe first."""
    jobs = [(tag, sc.make(seed=sd, **kw)) for tag, kw in cells for sd in seeds]
    probe = probe or jobs[len(jobs) // 2]
    t0 = time.perf_counter(); _job(probe); t1 = time.perf_counter() - t0
    print(f"[{name}] {len(cells)} cells x {len(seeds)} seeds = {len(jobs)} runs; probe {t1:.1f} s/run "
          f"-> ~{len(jobs) * t1 / PROCS / 60:.1f} min on {PROCS} processes (upper bound: early stops are faster)", flush=True)
    t0 = time.perf_counter()
    with Pool(PROCS) as p:
        rows = p.map(_job, jobs, chunksize=1)
    wall = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    os.makedirs(f"{ROOT}/results/runs", exist_ok=True)
    df.to_parquet(f"{ROOT}/results/runs/{name}.parquet", index=False)
    print(f"[{name}] done in {wall / 60:.1f} min; {len(df)} rows -> results/runs/{name}.parquet", flush=True)
    return df, wall


def boot_ci(x, b=B, seed=0):
    """Mean and percentile 95% CI over seeds (resampling seeds with replacement)."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return [float("nan")] * 3
    idx = np.random.default_rng(seed).integers(0, len(x), (b, len(x)))
    m = x[idx].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def boot_diff(a, b_, b=B, seed=1):
    """Paired difference mean(b_ - a) over common seeds (a, b_ Series indexed by seed), with 95% CI."""
    j = pd.concat([a, b_], axis=1, join="inner").dropna()
    return boot_ci(j.iloc[:, 1] - j.iloc[:, 0], b, seed) + [len(j)]


def summarize(df, metrics, by="cond"):
    out = {}
    for tag, g in df.groupby(by, sort=False):
        out[tag] = {"n_seeds": int(len(g))}
        for m in metrics:
            out[tag][m] = boot_ci(g[m].values)
    return out


def meta(seeds, wall, extra=None):
    import scipy
    m = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
         "platform": platform.platform(), "seeds": list(map(int, seeds)), "split": "train", "bootstrap_resamples": B,
         "ci": "percentile 95%, resampling seeds", "wall_min": round(wall / 60, 2), "procs": PROCS,
         "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "constants": {k: getattr(flock, k) for k in dir(flock) if k.isupper() and isinstance(getattr(flock, k), (int, float, str))}}
    m.update(extra or {})
    return m


def write(name, summary, meta_):
    os.makedirs(f"{ROOT}/results/kt", exist_ok=True)
    with open(f"{ROOT}/results/kt/{name}.json", "w") as f:
        json.dump({"meta": meta_, "summary": summary}, f, indent=1, default=float)
    print(f"[{name}] summary -> results/kt/{name}.json")


def fmt(ci):
    return f"{ci[0]:.2f} [{ci[1]:.2f}, {ci[2]:.2f}]"


if __name__ == "__main__":
    m, lo, hi = boot_ci(np.r_[np.zeros(50), np.ones(50)])
    assert m == 0.5 and 0.39 < lo < 0.42 and 0.58 < hi < 0.61, (lo, hi)
    d = boot_diff(pd.Series([0.0, 0, 0, 0], index=[0, 1, 2, 3]), pd.Series([1.0, 1, 1], index=[1, 2, 3]))
    assert d[:3] == [1.0, 1.0, 1.0] and d[3] == 3
    print("kt_common self-check OK")
