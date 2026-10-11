"""Scenario contract (Contract 1): defaults, deep merge, stable hash, light validation.
Every run starts from DEFAULT and overrides keys: scenario.make(seed=3, flock={"followers": 17}).
Run: uv run python scripts/scenario.py"""
import copy, hashlib, json

DEFAULT = {
    "seed": 0,
    "world": {"size_m": [800, 800], "landscape": "meadow", "obstacle_density": 0.0},
    "home": [0.0, 0.0, 20.0], "goal": [600.0, 0.0, 20.0],
    "flock": {"n": 50, "anchors": 3, "gnss": 10, "followers": 37, "rule": "topological", "k": 7, "radius_m": 15.0},
    "spoof": {"on": True, "offset_m": [0.0, 200.0, 0.0], "start_s": 0.0, "ramp_s": 0.0},
    "anchor": {"stubbornness": 1.0, "adaptive": False, "adaptive_rule": "anchor_gain"},
    "sensing": {"range_m": 10.0, "fov_deg": 360.0, "occlusion": True, "pos_noise_m": 0.3},
    "radio": {"available": False, "fallback": "own_gnss"},
    "wind": {"mean_mps": [0.0, 0.0, 0.0], "turbulence": "none", "shared_fraction": 1.0},
    "attrition": {"fraction": 0.0, "t_s": 60.0},
    "mission": {"max_time_s": 600.0, "dt_s": 0.02},
}
ENUMS = {("world", "landscape"): {"meadow", "forest", "urban"}, ("flock", "rule"): {"topological", "metric", "zones"},
         ("radio", "fallback"): {"own_gnss", "hold", "return_home"}, ("anchor", "adaptive_rule"): {"anchor_gain", "follower_filter"},
         ("wind", "turbulence"): {"none", "light", "moderate", "severe"}}

def merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out

def make(**over):
    """DEFAULT overridden by keyword sections; followers default to n - anchors - gnss unless given."""
    s = merge(DEFAULT, over)
    f = s["flock"]
    if "followers" not in over.get("flock", {}):
        f["followers"] = f["n"] - f["anchors"] - f["gnss"]
    if "n" not in over.get("flock", {}):
        f["n"] = f["anchors"] + f["gnss"] + f["followers"]
    validate(s)
    return s

def validate(s):
    f = s["flock"]
    assert min(f["anchors"], f["gnss"], f["followers"]) >= 0, f
    assert f["anchors"] + f["gnss"] + f["followers"] == f["n"], f"role counts must sum to n: {f}"
    for (a, b), allowed in ENUMS.items():
        assert s[a][b] in allowed, f"{a}.{b}={s[a][b]!r} not in {sorted(allowed)}"
    assert 0 <= s["wind"]["shared_fraction"] <= 1 and 0 <= s["attrition"]["fraction"] <= 1
    assert s["mission"]["dt_s"] > 0 and s["sensing"]["range_m"] > 0

def fake_goal(s):
    return [g + o for g, o in zip(s["goal"], s["spoof"]["offset_m"])]

def scenario_hash(s):
    """Hash of everything except the seed, so paired seeds share a condition hash."""
    body = {k: v for k, v in s.items() if k != "seed"}
    return hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]

if __name__ == "__main__":
    s = make(seed=3, flock={"n": 30, "anchors": 3, "gnss": 10})
    assert s["flock"]["followers"] == 17 and s["seed"] == 3 and DEFAULT["flock"]["n"] == 50
    assert scenario_hash(s) == scenario_hash(make(seed=9, flock={"n": 30, "anchors": 3, "gnss": 10}))
    assert scenario_hash(s) != scenario_hash(make(flock={"n": 30, "anchors": 3, "gnss": 10}, spoof={"on": False}))
    assert fake_goal(make()) == [600.0, 200.0, 20.0]
    try: make(flock={"rule": "bogus"}); raise SystemExit("validate missed a bad enum")
    except AssertionError: pass
    print("scenario self-check OK")
