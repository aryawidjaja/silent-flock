"""Phase 2 analysis: every hypothesis test on the held-out sweeps -> results/analysis.json.
Reads results/runs/<grid>__*.parquet (h1, h2, h3, robust, baselines). Bootstrap = resampling seeds
(paired across cells: one resample of the seed list is applied to every cell), 10,000 resamples.
Run: uv run python scripts/analyze.py [--boot 10000]      self-check: uv run python scripts/analyze.py --selfcheck"""
import argparse, glob, json, os, platform
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "runs")
H2_HELDOUT = lambda a, s, n: a == 2 or (n == 100 and s == 10)      # pre-registered held-out cells


def load(grid):
    fs = sorted(glob.glob(os.path.join(RUNS, f"{grid}__*.parquet")))
    if not fs:
        return None
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    dup = d.duplicated(["cell", "seed"])
    assert not dup.any(), f"{grid}: duplicate (cell, seed) rows"
    return d


def boot_idx(seeds, B, rs=0):
    """B x len(seeds) resampled positions into the sorted seed list (shared across cells => paired)."""
    return np.random.default_rng(rs).integers(0, len(seeds), (B, len(seeds)))


def cell_matrix(d, cells, col):
    """cells x seeds matrix of `col` (NaN where missing); seeds sorted."""
    p = d[d.cell.isin(cells)].pivot(index="cell", columns="seed", values=col).reindex(cells)
    return p.to_numpy(float), list(p.columns)


def ci_of(samples):
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))] if len(s) else [np.nan, np.nan]


def mean_ci(M, I):
    """M: cells x seeds; I: B x seeds. Returns per-cell mean, lo, hi (nanmean over seeds)."""
    est = np.nanmean(M, 1)
    bs = np.stack([np.nanmean(M[:, ix], 1) for ix in I])        # B x cells
    lo, hi = np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)
    return est, lo, hi, bs


def logit_slope(x, k, m):
    """Grouped binomial logistic regression of k successes / m trials on x (Newton). Returns (b0, b1)."""
    b = np.zeros(2)
    X = np.c_[np.ones_like(x), x]
    for _ in range(50):
        p = 1 / (1 + np.exp(-(X @ b)))
        g = X.T @ (k - m * p)
        H = (X * (m * p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-10:
            break
    return b


def ci(est, lo, hi, n=None, src=None):
    out = {"mean": float(est), "lo": float(lo), "hi": float(hi)}
    if n is not None:
        out["n_seeds"] = int(n)
    if src:
        out["_source"] = src
    return out


def crossing(xs, ys, thr, below=False):
    """First x where y crosses thr (y >= thr, or y <= thr if below), linear interpolation; nan = censored."""
    ys = np.asarray(ys, float)
    ok = ys <= thr if below else ys >= thr
    if not ok.any():
        return np.nan
    j = int(np.argmax(ok))
    if j == 0:
        return float(xs[0])
    x0, x1, y0, y1 = xs[j - 1], xs[j], ys[j - 1], ys[j]
    return float(x0 + (thr - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x1)


def fit_law(rows):
    """log w = log a + b log(S/A) + c log N, OLS on uncensored rows [(A, S, N, w)]."""
    r = [x for x in rows if np.isfinite(x[3]) and x[3] > 0]
    if len(r) < 4:
        return None
    A, S, N, W = map(np.array, zip(*r))
    X = np.c_[np.ones(len(r)), np.log(S / A), np.log(N)]
    coef, *_ = np.linalg.lstsq(X, np.log(W), rcond=None)
    return coef


def law_pred(coef, A, S, N):
    return np.exp(coef[0] + coef[1] * np.log(S / A) + coef[2] * np.log(N))


def r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < 2:
        return np.nan
    return float(1 - ((y[m] - yhat[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum())


# ---------------------------------------------------------------- H1
def analyze_h1(d, B):
    out = {"by_cell": {}, "slopes": {}, "composition": {}}
    seeds = sorted(d.seed.unique()); I = boot_idx(seeds, B, 1)
    cells = sorted(d.cell.unique())
    for col in ("capture_follower", "capture_nonspoofed", "arrival_frac", "capture_anchor", "fragments", "arrival_nonspoofed"):
        M, _ = cell_matrix(d, cells, col)
        est, lo, hi, _ = mean_ci(M, I)
        for c, e, l, h in zip(cells, est, lo, hi):
            out["by_cell"].setdefault(c, {})[col] = ci(e, l, h, len(seeds))
    meta = d.drop_duplicates("cell").set_index("cell")
    for c in cells:
        out["by_cell"][c].update({"anchors": int(meta.loc[c, "anchors"]), "followers": int(meta.loc[c, "followers"]),
                                  "gnss": int(meta.loc[c, "gnss"]), "offset_m": float(meta.loc[c, "offset_m"]),
                                  "spoof_on": bool(meta.loc[c, "spoof_on"])})
    # primary: follower capture slope on U (U >= 5), per anchors x offset, seed-paired bootstrap
    groups = meta[meta.spoof_on & (meta.followers >= 5) & (meta.anchors > 0)].groupby(["anchors", "offset_m"])
    for (a, off), g in groups:
        cs = list(g.sort_values("followers").index); U = g.loc[cs, "followers"].to_numpy(float)
        F, _ = cell_matrix(d, cs, "capture_follower")
        K, M = np.nan_to_num(F) * U[:, None], np.where(np.isfinite(F), 1.0, 0.0) * U[:, None]
        b1 = logit_slope(U, K.sum(1), M.sum(1))[1]
        bs = np.array([logit_slope(U, K[:, ix].sum(1), M[:, ix].sum(1))[1] for ix in I])
        out["slopes"][f"a{a}_d{int(off)}"] = {"anchors": int(a), "offset_m": float(off), "slope_logodds_per_follower": float(b1),
                                              "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                                              "p_one_sided": float((bs <= 0).mean()), "followers": U.tolist()}
    # secondary: composition null for the combined (anchors + followers) capture
    for (a, off), g in groups:
        cs = list(g.sort_values("followers").index); U = g.loc[cs, "followers"].to_numpy(float)
        cf = np.nanmean(cell_matrix(d, cs, "capture_follower")[0]); ca = np.nanmean(cell_matrix(d, cs, "capture_anchor")[0])
        pred = (U * cf + a * ca) / (a + U)
        obs = np.nanmean(cell_matrix(d, cs, "capture_nonspoofed")[0], 1)
        out["composition"][f"a{a}_d{int(off)}"] = {"followers": U.tolist(), "observed": obs.tolist(), "predicted": pred.tolist(),
                                                   "c_follower": float(cf), "c_anchor": float(ca),
                                                   "max_abs_dev": float(np.nanmax(np.abs(obs - pred)))}
    return out


# ---------------------------------------------------------------- H1b headcount split
def analyze_h1b(h1, h2, B):
    """Follower capture vs S/(A+S) over cells where the anchors escape (anchor capture <= 0.5)."""
    pts = []
    for grid, d in (("h1", h1), ("h2", h2)):
        if d is None:
            continue
        sub = d[(d.spoof_on) & (d.anchors > 0) & (d.followers >= 5)]
        if grid == "h1":
            sub = sub[sub.offset_m == 200]
        for c, g in sub.groupby("cell"):
            if np.nanmean(g.capture_anchor) > 0.5:
                continue
            a, s = int(g.anchors.iloc[0]), int(g.gnss.iloc[0])
            pts.append({"grid": grid, "cell": c, "anchors": a, "spoofed": s, "n": int(g.n.iloc[0]),
                        "stubbornness": float(g.stubbornness.iloc[0]), "share": s / (a + s),
                        "seeds": g.seed.to_numpy(), "cap": g.capture_follower.to_numpy(float)})
    if len(pts) < 3:
        return None
    x = np.array([p["share"] for p in pts]); y = np.array([np.nanmean(p["cap"]) for p in pts])
    coef = np.polyfit(x, y, 1)
    rs = np.random.default_rng(2); bs = []
    for _ in range(B):
        yb = np.array([np.nanmean(p["cap"][rs.integers(0, len(p["cap"]), len(p["cap"]))]) for p in pts])
        bs.append(np.polyfit(x, yb, 1))
    bs = np.array(bs)
    wta = np.where(x > 0.5, 1.0, np.where(x < 0.5, 0.0, 0.5))
    return {"slope": float(coef[0]), "slope_ci": ci_of(bs[:, 0]), "intercept": float(coef[1]), "intercept_ci": ci_of(bs[:, 1]),
            "r2_proportional": r2(y, x), "r2_winner_take_all": r2(y, wta), "r2_linear_fit": r2(y, np.polyval(coef, x)),
            "supported": bool(ci_of(bs[:, 0])[0] <= 1 <= ci_of(bs[:, 0])[1] and ci_of(bs[:, 1])[0] <= 0 <= ci_of(bs[:, 1])[1]),
            "points": [{k: (v if not isinstance(v, np.ndarray) else None) for k, v in p.items() if k not in ("seeds", "cap")}
                       | {"follower_capture": float(np.nanmean(p["cap"]))} for p in pts]}


# ---------------------------------------------------------------- H2
def analyze_h2(d, B):
    seeds = sorted(d.seed.unique()); I = boot_idx(seeds, B, 3)
    meta = d.drop_duplicates("cell").set_index("cell")
    cells, laws = [], {}
    for (n, a, s), g in meta.groupby(["n", "anchors", "gnss"]):
        cs = list(g.sort_values("stubbornness").index); W = g.loc[cs, "stubbornness"].to_numpy(float)
        Arr, _ = cell_matrix(d, cs, "arrival_frac"); Anc, _ = cell_matrix(d, cs, "capture_anchor")
        Fol, _ = cell_matrix(d, cs, "capture_follower")
        ws = crossing(W, np.nanmean(Arr, 1), 0.5); we = crossing(W, np.nanmean(Anc, 1), 0.5, below=True)
        bws = np.array([crossing(W, np.nanmean(Arr[:, ix], 1), 0.5) for ix in I[:2000]])
        bwe = np.array([crossing(W, np.nanmean(Anc[:, ix], 1), 0.5, below=True) for ix in I[:2000]])
        cells.append({"n": int(n), "anchors": int(a), "spoofed": int(s), "heldout": bool(H2_HELDOUT(a, s, n)),
                      "omega": W.tolist(), "arrival": np.nanmean(Arr, 1).tolist(), "anchor_capture": np.nanmean(Anc, 1).tolist(),
                      "follower_capture": np.nanmean(Fol, 1).tolist(),
                      "omega_star": {"mean": ws, "lo": ci_of(bws)[0], "hi": ci_of(bws)[1], "censored_frac": float(np.isnan(bws).mean())},
                      "omega_esc": {"mean": we, "lo": ci_of(bwe)[0], "hi": ci_of(bwe)[1], "censored_frac": float(np.isnan(bwe).mean())},
                      "censored_star": bool(np.isnan(ws)), "censored_esc": bool(np.isnan(we))})
    for key in ("omega_star", "omega_esc"):
        tr = [(c["anchors"], c["spoofed"], c["n"], c[key]["mean"]) for c in cells if not c["heldout"]]
        te = [(c["anchors"], c["spoofed"], c["n"], c[key]["mean"]) for c in cells if c["heldout"]]
        coef = fit_law(tr)
        if coef is None:
            laws[key] = {"fit": None, "verdict": "null: too few uncensored training cells to fit",
                         "n_train_uncensored": sum(np.isfinite(t[3]) for t in tr), "n_heldout_uncensored": sum(np.isfinite(t[3]) for t in te)}
            continue
        ptr = [law_pred(coef, a, s, n) for a, s, n, _ in tr]; pte = [law_pred(coef, a, s, n) for a, s, n, _ in te]
        r2te = r2([t[3] for t in te], pte)
        laws[key] = {"fit": {"a": float(np.exp(coef[0])), "b": float(coef[1]), "c": float(coef[2])},
                     "r2_train": r2([t[3] for t in tr], ptr), "r2_heldout": r2te,
                     "r2_heldout_log": r2(np.log([t[3] for t in te]), np.log(pte)),
                     "n_train_uncensored": sum(np.isfinite(t[3]) for t in tr), "n_heldout_uncensored": sum(np.isfinite(t[3]) for t in te),
                     "verdict": ("law holds (R2 >= 0.8 on held-out cells)" if np.isfinite(r2te) and r2te >= 0.8 else "null (R2 < 0.8 or undefined)")}
    return {"cells": cells, "laws": laws, "criterion": "pre-registered: R2 >= 0.8 on held-out cells, omega* scale", "n_seeds": len(seeds)}


# ---------------------------------------------------------------- H3, robustness, baselines
def contrast_table(d, B, cols=("arrival_frac", "capture_frac", "arrival_nonspoofed", "capture_nonspoofed", "capture_follower", "fragments")):
    seeds = sorted(d.seed.unique()); I = boot_idx(seeds, B, 4); cells = sorted(d.cell.unique())
    out = {c: {} for c in cells}
    for col in cols:
        M, _ = cell_matrix(d, cells, col); est, lo, hi, _ = mean_ci(M, I)
        for c, e, l, h in zip(cells, est, lo, hi):
            out[c][col] = ci(e, l, h, len(seeds))
    return out, I, seeds


def paired(d, a, b, col, I):
    cells = [a, b]; M, _ = cell_matrix(d, cells, col); diff = M[1] - M[0]
    bs = np.array([np.nanmean(diff[ix]) for ix in I])
    return {"diff": float(np.nanmean(diff)), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "p_two_sided": float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))}


def analyze_h3(d, B):
    tab, I, seeds = contrast_table(d, B)
    ad, fx = "adaptive1_spoof1", "adaptive0_spoof1"
    res = {"cells": tab, "n_seeds": len(seeds),
           "adaptive_minus_fixed": {col: paired(d, fx, ad, col, I) for col in ("arrival_frac", "capture_frac", "arrival_nonspoofed")}}
    a = tab[ad]
    res["criterion"] = {"arrival_ge_0.6": bool(a["arrival_frac"]["mean"] >= 0.6), "capture_le_0.2": bool(a["capture_frac"]["mean"] <= 0.2)}
    res["verdict"] = "supported" if all(res["criterion"].values()) else "not supported"
    return res


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()


def main(B):
    out = {"meta": {"boot": B, "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}, "files": {}}
    D = {g: load(g) for g in ("h1", "h2", "h3", "robust", "baselines")}
    for g, d in D.items():
        if d is not None:
            out["files"][g] = {"rows": len(d), "seeds": [int(d.seed.min()), int(d.seed.max())], "n_seeds": int(d.seed.nunique()),
                               "split": sorted(d.split.unique().tolist()), "git": sorted(d.git.unique().tolist()),
                               "config": d.config.iloc[0] if "config" in d else None}
    if D["h1"] is not None:
        out["h1"] = analyze_h1(D["h1"], B)
    out["h1b"] = analyze_h1b(D["h1"], D["h2"], min(B, 2000))
    if D["h2"] is not None:
        out["h2"] = analyze_h2(D["h2"], B)
    if D["h3"] is not None:
        out["h3"] = analyze_h3(D["h3"], B)
    for g in ("robust", "baselines"):
        if D[g] is not None:
            out[g] = contrast_table(D[g], B)[0]
    if D["robust"] is not None:            # per perturbation: follower capture hi vs lo followers (paired)
        d = D["robust"]; seeds = sorted(d.seed.unique()); I = boot_idx(seeds, B, 5); us = sorted(d.followers.unique())
        out["robust_contrast"] = {}
        for name in sorted({c.rsplit("_u", 1)[0] for c in d.cell if "_u" in c}):
            lo_c, hi_c = f"{name}_u{min(us)}", f"{name}_u{max(us)}"
            if lo_c in set(d.cell) and hi_c in set(d.cell):
                out["robust_contrast"][name] = {"capture_follower_hi_minus_lo": paired(d, lo_c, hi_c, "capture_follower", I),
                                                "capture_nonspoofed_hi_minus_lo": paired(d, lo_c, hi_c, "capture_nonspoofed", I)}
    # Holm across the H1 primary slopes (one-sided p) and H3 arrival contrast
    ps, names = [], []
    for k, v in out.get("h1", {}).get("slopes", {}).items():
        ps.append(v["p_one_sided"]); names.append(f"h1_slope_{k}")
    if "h3" in out:
        ps.append(out["h3"]["adaptive_minus_fixed"]["arrival_frac"]["p_two_sided"]); names.append("h3_arrival")
    if ps:
        out["holm"] = dict(zip(names, holm(np.array(ps))))
    path = os.path.join(ROOT, "results", "analysis.json")
    json.dump(out, open(path, "w"), indent=1, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else str(o))
    print("wrote", path, {k: (v["rows"]) for k, v in out["files"].items()})
    return out


def selfcheck():
    # logistic slope recovers a known slope
    x = np.array([5, 10, 17, 25, 30, 40.0]); m = np.full(6, 1e6); p = 1 / (1 + np.exp(-(-1 + 0.05 * x)))
    b = logit_slope(x, p * m, m); assert abs(b[1] - 0.05) < 1e-6 and abs(b[0] + 1) < 1e-5, b
    assert crossing([1, 2, 4], [0.2, 0.4, 0.8], 0.5) == 2.5 and np.isnan(crossing([1, 2], [0.1, 0.2], 0.5))
    assert crossing([1, 2, 4], [0.9, 0.6, 0.1], 0.5, below=True) == 2.4
    A, S, N = np.array([1, 3, 5, 1, 3, 5.0]), np.array([10, 10, 20, 5, 20, 5.0]), np.array([20, 50, 100, 50, 20, 100.0])
    W = 0.8 * (S / A) ** 0.7 * N ** 0.15
    coef = fit_law(list(zip(A, S, N, W))); assert np.allclose([np.exp(coef[0]), coef[1], coef[2]], [0.8, 0.7, 0.15])
    assert np.allclose(holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])
    assert H2_HELDOUT(2, 5, 20) and H2_HELDOUT(3, 10, 100) and not H2_HELDOUT(3, 10, 50)
    print("analyze self-check OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--boot", type=int, default=10000); ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    selfcheck() if a.selfcheck else main(a.boot)
