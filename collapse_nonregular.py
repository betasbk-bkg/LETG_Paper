"""
========================================================================
Non-Regular Trajectory Collapse Verification (LE-TG generalization)
------------------------------------------------------------------------
Tests x* = v*sqrt(tau) scaling collapse on THREE non-regular paths:
  1) Lemniscate    -- engine builtin, self-intersecting figure-8
  2) Ellipse       -- smooth continuous-curvature (curvature varies
                      between minor/major axis)
  3) Suzuka GP (real centerline)-- mixed straights + sweeping curves + a hairpin
                      (representative of real-world racing-game paths)

Why these three:
  - Lemniscate     : highest-risk test (self-intersection, sign change
                     in curvature) -- if collapse survives here, the law
                     is robust well beyond polygons.
  - Ellipse        : smooth but non-uniform curvature -- bridges Circle
                     and polygon cases, isolates "is the constant-
                     curvature assumption necessary?".
  - Suzuka GP (real centerline) : real-world-ish closed loop with multiple corner
                     scales -- directly answers the editor critique
                     "generalization to arbitrary trajectories
                     requires additional validation".

All three use the SAME interface as sm.Circle / sm.Square in the engine:
  .name, .circ, closest(p) -> (point, arclen), at(arc) -> point,
  start() -> point

Method (identical to corner_freq_experiment.measure_vstar):
  - monkey-patch sm.DELAY_F to tau_f
  - sweep SPEEDS, collect run_condition(...)['rmse_mean']
  - parabolic 3-point v* estimator
  - compute x* = v* * sqrt(tau_s) per tau, CV across tau,
    and power-law fit v* = C * tau^(-beta) for R^2

Verdict logic (per trajectory):
  - COLLAPSE PRESERVED       : beta in [0.30, 0.70] AND CV(x*) < 15%
  - COLLAPSE PRESERVED (x* shifted) : same but |x* - 1.347|/1.347 > 0.25
  - COLLAPSE DEGRADED        : otherwise -> report as scope limit

Usage:
  python collapse_nonregular.py
Output: collapse_nonregular_results.json
========================================================================
"""

import json
import numpy as np
import simulation_main as sm


# ----------------------------------------------------------------------
# Shared sweep configuration (matches the manuscript's grids)
# ----------------------------------------------------------------------
TAU_ALL = {100: 6, 200: 12, 300: 18, 433: 26, 600: 36, 1000: 60}
SPEEDS  = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
           2.25, 2.5, 3.0, 4.0, 5.0]
TROLL   = 0.05
N       = 150
MC      = 15

SMOKE_TEST = False
if SMOKE_TEST:
    TAU_ALL = {100: 6, 433: 26, 1000: 60}
    SPEEDS  = [0.5, 1.0, 1.5, 2.0, 3.0]
    MC      = 3


# ======================================================================
# NEW TRAJECTORY CLASSES
# (Lemniscate is already in simulation_main.py; we use sm.Lemniscate.)
# ======================================================================
class Ellipse:
    """Ellipse with semi-axes a (along x) and b (along y).
    Discretized into n points sampled by arc length so .at(arc) is
    uniform in path length, matching engine convention for Lemniscate.
    Closed curve, smooth, curvature varies continuously between
    1/a^2*b at minor-axis vertex and 1/a*b^2 at major-axis vertex.
    """

    def __init__(self, a=12.0, b=6.0, n=800, name=None):
        self.name = name or "ellipse"
        self.a = float(a)
        self.b = float(b)
        # Parametric sample
        ts = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        self.pts = np.column_stack([a * np.cos(ts), b * np.sin(ts)])
        dl = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        # close the loop
        last_dl = np.linalg.norm(self.pts[0] - self.pts[-1])
        self.arcs = np.concatenate([[0.0], np.cumsum(dl)])
        self.circ = float(self.arcs[-1] + last_dl)

    def closest(self, p):
        d = np.linalg.norm(self.pts - p, axis=1)
        i = int(np.argmin(d))
        return self.pts[i].copy(), float(self.arcs[i])

    def at(self, arc):
        a = arc % self.circ
        i = min(int(np.searchsorted(self.arcs, a)), len(self.pts) - 1)
        return self.pts[i].copy()

    def start(self):
        return self.pts[0].copy()


class SuzukaTrack:
    """Suzuka GP circuit, scaled so perimeter matches Circle(R=10).

    Centerline coordinates are loaded from a CSV with columns (x, y).
    Source: real Suzuka GP centerline (~5.807 km in raw units), which
    we rescale so its perimeter equals Circle(R=10).circ = 62.83 m for
    a fair comparison with the manuscript trajectories. Shape is
    preserved exactly; only the size is normalised.

    Famous features visible in the rescaled track:
      - S-curves + Degner (upper left)
      - Hairpin (centre-left)
      - Spoon (lower right)
      - 130R + final chicane (right)
    """

    CSV_PATH = "suzuka_track.csv"  # place CSV next to this script

    def __init__(self, csv_path=None, target_perimeter=None, name="suzuka_gp"):
        self.name = name
        path = csv_path or self.CSV_PATH
        raw = np.loadtxt(path, delimiter=",", skiprows=1)

        # Centre about origin
        raw = raw - raw.mean(axis=0)

        # Raw perimeter (treat as open polyline + closing gap)
        seg_raw = np.diff(raw, axis=0)
        perim_raw = float(np.linalg.norm(seg_raw, axis=1).sum())
        gap_raw = float(np.linalg.norm(raw[0] - raw[-1]))
        total_raw = perim_raw + gap_raw

        # Rescale to match Circle(R=10) by default
        if target_perimeter is None:
            target_perimeter = 2.0 * np.pi * 10.0  # 62.832...
        scale = target_perimeter / total_raw
        self.pts = raw * scale

        # Arc-length indexing (same convention as sm.Lemniscate)
        dl = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        last_dl = float(np.linalg.norm(self.pts[0] - self.pts[-1]))
        self.arcs = np.concatenate([[0.0], np.cumsum(dl)])
        self.circ = float(self.arcs[-1] + last_dl)

    def closest(self, p):
        d = np.linalg.norm(self.pts - p, axis=1)
        i = int(np.argmin(d))
        return self.pts[i].copy(), float(self.arcs[i])

    def at(self, arc):
        a = arc % self.circ
        i = min(int(np.searchsorted(self.arcs, a)), len(self.pts) - 1)
        return self.pts[i].copy()

    def start(self):
        return self.pts[0].copy()



# ======================================================================
# v* estimator and metrics
# ======================================================================
def measure_vstar(traj, tau_f, troll=TROLL):
    """Exact replica of corner_freq_experiment.measure_vstar."""
    orig = sm.DELAY_F
    sm.DELAY_F = tau_f
    try:
        rmses = []
        for v in SPEEDS:
            r = sm.run_condition(traj, N, troll, MC,
                                 method='fixed', speed_override=v,
                                 seed_base=31, seed_offset=N)
            rmses.append(r['rmse_mean'])
    finally:
        sm.DELAY_F = orig

    idx = int(np.argmin(rmses))
    vstar = float(SPEEDS[idx])
    if 1 <= idx <= len(SPEEDS) - 2:
        try:
            c = np.polyfit(SPEEDS[idx - 1:idx + 2],
                           rmses[idx - 1:idx + 2], 2)
            if c[0] > 0:
                vf = -c[1] / (2 * c[0])
                if SPEEDS[idx - 1] <= vf <= SPEEDS[idx + 1]:
                    vstar = float(vf)
        except Exception:
            pass
    return vstar, float(min(rmses)), rmses


def collapse_metrics(tau_ms_list, vstar_list):
    tau_s = np.array([t / 1000.0 for t in tau_ms_list])
    vs = np.array(vstar_list)
    x_star = vs * np.sqrt(tau_s)
    x_mean = float(np.mean(x_star))
    x_std = float(np.std(x_star, ddof=1)) if len(x_star) > 1 else 0.0
    x_cv = (x_std / x_mean) if x_mean != 0 else float('nan')

    lx, ly = np.log(tau_s), np.log(vs)
    A = np.vstack([np.ones_like(lx), -lx]).T
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    C, beta = float(np.exp(coef[0])), float(coef[1])
    pred = A @ coef
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float('nan')

    return {
        "x_star_per_tau": x_star.tolist(),
        "x_star_mean": x_mean,
        "x_star_std": x_std,
        "x_star_cv_pct": float(100.0 * x_cv),
        "C_fit": C,
        "beta_fit": beta,
        "r2_powerlaw": r2,
        "manuscript_xstar": 1.347,
        "manuscript_cv_pct": 6.4,
        "manuscript_beta": 0.511,
    }


def run_one_trajectory(traj, label):
    print("-" * 68)
    print("Trajectory: %s  (perimeter=%.2f m)" % (label, traj.circ))
    print("-" * 68)
    vstars, rmins, curves = [], [], []
    for tau_ms, tau_f in TAU_ALL.items():
        v, rm, curve = measure_vstar(traj, tau_f)
        vstars.append(v)
        rmins.append(rm)
        curves.append(curve)
        print("  tau=%4d ms (delay=%2d f)  v*=%.3f  RMSE_min=%.3f"
              % (tau_ms, tau_f, v, rm))

    m = collapse_metrics(list(TAU_ALL.keys()), vstars)
    print("\n  COLLAPSE METRICS")
    print("  ----------------")
    print("  x* mean = %.3f  (manuscript: 1.347)" % m['x_star_mean'])
    print("  x* CV   = %.2f %% (manuscript: 6.4%%)" % m['x_star_cv_pct'])
    print("  C fit   = %.3f" % m['C_fit'])
    print("  beta    = %.3f  (manuscript: 0.511)" % m['beta_fit'])
    print("  R^2     = %.3f" % m['r2_powerlaw'])

    return {
        "label": label,
        "perimeter_m": traj.circ,
        "vstar_by_tau": vstars,
        "rmse_min_by_tau": rmins,
        "rmse_curves_by_tau": curves,
        "metrics": m,
    }


def verdict(m):
    cv = m["x_star_cv_pct"]
    x_mean = m["x_star_mean"]
    beta = m["beta_fit"]
    cv_ok = cv < 15.0
    beta_ok = 0.30 <= beta <= 0.70
    x_close = abs(x_mean - 1.347) / 1.347 < 0.25
    if cv_ok and beta_ok:
        return ("COLLAPSE PRESERVED" if x_close
                else "COLLAPSE PRESERVED (x* shifted)")
    return "COLLAPSE DEGRADED - report as scope limit"


def main():
    print("=" * 68)
    print("NON-REGULAR TRAJECTORY COLLAPSE VERIFICATION")
    print("MC=%d  speeds=%d  taus=%d%s"
          % (MC, len(SPEEDS), len(TAU_ALL),
             "  [SMOKE]" if SMOKE_TEST else ""))
    print("Trajectories: Lemniscate (self-intersecting) | Ellipse "
          "(smooth) | F1-track (mixed)")
    print("=" * 68)

    out = {
        "config": {
            "tau_ms": list(TAU_ALL.keys()),
            "speeds": SPEEDS, "troll": TROLL, "N": N, "MC": MC,
            "smoke_test": SMOKE_TEST,
            "manuscript_reference": {
                "x_star": 1.347, "cv_pct": 6.4,
                "beta": 0.511, "r2_collapse": 0.956,
                "trajectories_in_paper": ["Circle", "Hexagon",
                                          "Square", "Triangle"],
            },
        },
        "results": {},
    }

    # 1) Lemniscate (engine builtin), perimeter-matched to Circle(R=10)
    # for fair comparison (a=12 -> perimeter ~62.83 m, identical to
    # Circle). The smaller a=7 variant was excluded to avoid mixing
    # shape effects with path-length effects.
    out["results"]["lemniscate"] = run_one_trajectory(
        sm.Lemniscate(a=12),
        "Lemniscate (a=12, perimeter-matched)")

    # 2) Ellipse -- a=12, b=6 gives an eccentric ellipse with continuous
    # but non-uniform curvature; perimeter ~ 58 m (close to Circle).
    out["results"]["ellipse_12_6"] = run_one_trajectory(
        Ellipse(a=12.0, b=6.0), "Ellipse(a=12, b=6)")

    # 3) Suzuka GP (real centerline) -- scale chosen so perimeter is in the same
    # ballpark as Circle(R=10).
    out["results"]["suzuka"] = run_one_trajectory(
        SuzukaTrack(), "Suzuka GP (real centerline, rescaled)")

    print("\n" + "=" * 68)
    print("SUMMARY  (manuscript: x*=1.347, CV=6.4%, beta=0.511)")
    print("=" * 68)
    for key, r in out["results"].items():
        m = r["metrics"]
        v = verdict(m)
        print("  %-50s  %s" % (r["label"], v))
        print("    x*=%.3f  CV=%.1f%%  beta=%.3f  R2=%.3f"
              % (m["x_star_mean"], m["x_star_cv_pct"],
                 m["beta_fit"], m["r2_powerlaw"]))
    print("=" * 68)

    with open("collapse_nonregular_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote collapse_nonregular_results.json")


if __name__ == "__main__":
    main()
