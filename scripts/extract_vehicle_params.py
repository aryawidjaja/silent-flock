"""Vehicle parameters for the flock (Crazyflie 2.x class).
1. results/vehicle_params.json = ../mutant-brain-drone/results/vehicle_params.json copied UNCHANGED (operating
   manual: one parameter file for both projects). Base values: mass, inertia, arm, Förster thrust map, motor lag, rotor drag.
2. results/vehicle_params_flock.json = the extra values the translational flock model needs and the sister file
   does not hold (tilt limit, attitude lag, velocity gain, body drag, speeds); every entry has value/unit/source/tag/note.
vehicle.load_params() merges the two into generic flat keys (derivations there).
Raw inputs opened for (2): data/raw/cf_firmware/defaults.h (Bitcraze firmware @ FW_COMMIT), Bitcraze PWM-to-thrust page.
Run: uv run python scripts/extract_vehicle_params.py"""
import json, math, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE, EXTRA = ROOT / "results/vehicle_params.json", ROOT / "results/vehicle_params_flock.json"
SISTER = ROOT.parent / "mutant-brain-drone/results/vehicle_params.json"
FW_COMMIT = "f6e0f0a3b526861328caf3296a7ecf00bb915d39"
FW_URL = f"https://github.com/bitcraze/crazyflie-firmware/blob/{FW_COMMIT}/src/platform/interface/platform_defaults.h"
BITCRAZE_PWM = "https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/pwm-to-thrust/"
G = 9.81


def P(value, unit, source, tag, note=""):
    return {"value": value, "unit": unit, "source": source, "tag": tag, "note": note}


def extras():
    return {
        "_meta": {"about": "Flock-model additions to results/vehicle_params.json (sister-project file, unchanged). "
                           "Tags: [opened] read from the opened source; DERIVED computed from opened numbers; "
                           "ASSUMPTION engineering choice.", "made_by": "scripts/extract_vehicle_params.py"},
        "max_tilt_rad": P(round(math.radians(20.0), 6), "rad", FW_URL + " PID_VEL_ROLL_MAX = PID_VEL_PITCH_MAX = 20.0f",
                          "[opened]", "roll/pitch limit of the stock velocity controller"),
        "tau_att_s": P(round(1 / 6.0, 5), "s", FW_URL + " PID_ROLL_KP 6.0 (attitude P) over PID_ROLL_RATE_KP 250.0",
                       "[opened]/DERIVED", "closed-loop attitude ~ first order with tau = 1/Kp when the inner rate loop "
                                           "is much faster; approximation, not a measured step response"),
        "kv_per_s": P(round(25.0 * math.pi / 180 * G, 5), "1/s", FW_URL + " PID_VEL_X_KP 25.0f (deg per m/s)",
                      "[opened]/DERIVED", "small-angle conversion a = g*theta; P only (firmware also has KI 1.0)"),
        "drag_quad_Nsm2m2": P(round(0.5 * 1.225 * 1.0 * 1.5e-3, 7), "N s^2/m^2",
                              "ASSUMPTION: 0.5*rho*Cd*A, rho 1.225 kg/m^3, Cd 1.0, A 1.5e-3 m^2", "ASSUMPTION",
                              "body pressure drag |v|v on top of Förster's linear rotor drag (fitted at 2.25-4.25 m/s, so "
                              "partly double-counted there); it bounds the terminal speed"),
        "v_cruise_mps": P(3.0, "m/s", "ASSUMPTION", "ASSUMPTION",
                          "firmware position mode defaults to 1 m/s (PID_POS_VEL_X_MAX 1.0f, [opened]); 3 m/s finishes a 600 m leg in 200 s"),
        "v_max_mps": P(5.0, "m/s", "ASSUMPTION", "ASSUMPTION", "below the drag-limited terminal speed at max tilt (vehicle.py self-check)"),
        "thrust_max_check_N": P(round(57.9e-3 * G, 5), "N", BITCRAZE_PWM + " ('at 93.75% PWM the trust is about 58g')",
                                "[opened]", "cross-check only: whole-vehicle thrust at 93.75 % PWM; the model uses 4 f_max "
                                            "from the sister file (Förster eq. 3.3 at full command)"),
    }


if __name__ == "__main__":
    assert SISTER.exists(), f"{SISTER} missing: the base parameter file comes from the sister project"
    shutil.copyfile(SISTER, BASE)
    assert BASE.read_bytes() == SISTER.read_bytes()
    e = extras()
    assert all({"value", "unit", "source", "tag", "note"} <= set(v) for k, v in e.items() if k != "_meta")
    EXTRA.write_text(json.dumps(e, indent=1, ensure_ascii=False) + "\n")
    print(f"copied {SISTER} -> {BASE} (unchanged); wrote {EXTRA}")
    for k, v in e.items():
        if k != "_meta": print(f"  {k:20s} {v['value']!s:12s} {v['tag']}")
