"""Shared counter-based RNG. Python reference; app/sim/rng.js is the bit-identical twin.

Every random number is a pure function of (seed, stream, step, idx), so Python and JS draw the same
numbers regardless of call order. Mixer: Chris Wellons' "lowbias32" integer hash (public domain,
https://github.com/skeeto/hash-prospector), chained over the keys:
    h = 0x811C9DC5;  for k in keys: h = lowbias32(((h ^ k) + 0x9E3779B9) mod 2^32)
All ops are 32-bit (uint32 wrap in numpy; Math.imul and >>> 0 in JS). Keys are reduced mod 2^32.
uniform = (h + 0.5) / 2^32, in the open interval (0, 1) (32-bit resolution).
normal  = Box-Muller cos branch, u1 from sub-counter 2*idx, u2 from 2*idx+1:
    z = sqrt(-2 ln u1) * cos(2 pi u2)    (|z| <= 6.66 because u1 >= 2^-33)
Not cryptographic; adequate statistical quality for simulation noise (checked below).
Run: uv run python scripts/rng.py"""
import numpy as np

S_INIT, S_VISION, S_WIND_SHARED, S_WIND_EACH, S_ATTRITION, S_ACT, S_CTRL = 1, 2, 3, 4, 5, 6, 7
# add new streams below; never renumber

_H0, _GOLD = np.uint32(0x811C9DC5), np.uint32(0x9E3779B9)
_M1, _M2 = np.uint32(0x7FEB352D), np.uint32(0x846CA68B)
_INV = 1.0 / 4294967296.0


def _u32(k):
    a = np.asarray(k)
    return a.astype(np.int64).astype(np.uint32) if a.dtype != np.uint32 else a  # mod 2^32, negatives wrap


def _lowbias32(x):
    x = x ^ (x >> np.uint32(16)); x = x * _M1
    x = x ^ (x >> np.uint32(15)); x = x * _M2
    return x ^ (x >> np.uint32(16))


def _lowbias32_int(x):  # same on a Python int (scalar fast path)
    x ^= x >> 16; x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15; x = (x * 0x846CA68B) & 0xFFFFFFFF
    return x ^ (x >> 16)


def hash32(*keys):
    """uint32 hash of broadcast uint32 keys (ints or integer arrays)."""
    h, i = 0x811C9DC5, 0
    while i < len(keys) and np.ndim(keys[i]) == 0:  # scalar prefix in exact int arithmetic
        h = _lowbias32_int(((h ^ (int(keys[i]) & 0xFFFFFFFF)) + 0x9E3779B9) & 0xFFFFFFFF); i += 1
    h = np.uint32(h)
    with np.errstate(over="ignore"):
        for k in keys[i:]:
            h = _lowbias32((h ^ _u32(k)) + _GOLD)
        return np.asarray(h, dtype=np.uint32)


def uniform(seed, stream, step, idx):
    """float64 in (0, 1), shape of broadcast(seed, stream, step, idx)."""
    return (hash32(seed, stream, step, idx).astype(np.float64) + 0.5) * _INV


def normal(seed, stream, step, idx):
    """Standard normal (Box-Muller, cos branch) from sub-counters 2*idx and 2*idx+1."""
    i2 = _u32(idx).astype(np.int64) * 2
    u = uniform(seed, stream, step, np.stack(np.broadcast_arrays(i2, i2 + 1)))
    u1, u2 = u[0], u[1]
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def _dump_fixture(path):
    """Expected values for the JS twin (scripts/test_c_js.mjs)."""
    import json
    seeds, streams, steps = [0, 1, 7, 12345, 2**31 + 5], [1, 2, 3, 4, 5, 6, 7], [0, 1, 99, 30000]
    rows = []
    for s in seeds:
        for st in streams:
            for k in steps:
                idx = np.r_[np.arange(16), 149, 150, 7499, 2**20 + 3, 2**31 - 1, 2**31, 2**32 - 2, 2**32 - 1]
                rows.append({"seed": s, "stream": st, "step": k, "idx": idx.tolist(),
                             "h": hash32(s, st, k, idx).tolist(),
                             "u": uniform(s, st, k, idx).tolist(), "z": normal(s, st, k, idx).tolist()})
    with open(path, "w") as f:
        json.dump(rows, f)
    return sum(len(r["idx"]) for r in rows)


if __name__ == "__main__":
    import sys, warnings
    warnings.simplefilter("error")  # any overflow warning fails the check
    idx = np.arange(1_000_000)
    u, z = uniform(3, S_INIT, 0, idx), normal(3, S_INIT, 0, idx)
    assert u.dtype == np.float64 and u.shape == idx.shape
    assert u.min() > 0 and u.max() < 1
    assert abs(u.mean() - 0.5) < 2e-3 and abs(u.var() - 1 / 12) < 1e-3, (u.mean(), u.var())
    assert abs(z.mean()) < 5e-3 and abs(z.var() - 1) < 5e-3, (z.mean(), z.var())
    assert abs(np.mean(z**3)) < 1e-2 and abs(np.mean(z**4) - 3) < 3e-2  # skew, kurtosis
    assert abs(np.corrcoef(u[:-1], u[1:])[0, 1]) < 3e-3  # neighbouring counters uncorrelated
    assert abs(np.corrcoef(u, uniform(4, S_INIT, 0, idx))[0, 1]) < 3e-3  # neighbouring seeds
    assert abs(np.corrcoef(u, uniform(3, S_INIT, 1, idx))[0, 1]) < 3e-3  # neighbouring steps
    assert np.array_equal(u, uniform(3, S_INIT, 0, idx))  # determinism
    assert not np.array_equal(u[:100], uniform(3, S_VISION, 0, idx[:100]))
    b = uniform(1, 2, np.arange(4)[:, None], np.arange(5)[None, :])  # broadcasting
    assert b.shape == (4, 5) and b[2, 3] == uniform(1, 2, 2, 3)
    assert np.ndim(uniform(1, 2, 3, 4)) == 0 and np.isfinite(normal(1, 2, 3, 2**32 - 1))
    assert uniform(0, 1, 0, -1) == uniform(0, 1, 0, 2**32 - 1)  # mod 2^32 key reduction
    ar = lambda x: np.array([x])  # scalar fast path == array path
    assert all(hash32(*k)[()] == hash32(*map(ar, k))[0] for k in [(0, 1, 2, 3), (2**31 + 5, 7, 30000, 2**32 - 1), (-3, 0, 5, 9)])
    n_fix = _dump_fixture(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f"rng self-check OK: uniform mean {u.mean():.5f} var {u.var():.5f}; normal mean {z.mean():.5f} "
          f"var {z.var():.5f} kurt {np.mean(z**4):.4f}; min/max u {u.min():.3e}/{1-u.max():.3e}"
          + (f"; fixture {n_fix} draws -> {sys.argv[1]}" if n_fix else ""))
