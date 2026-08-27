#!/usr/bin/env python3
"""The three numerical checks the Methods report, run rather than asserted.

1. Conservation. Across all 9,000 steps of the single-population run, with and
   without an alert, does S + E + I + R stay at one, and does any compartment go
   negative?
2. Agreement with an adaptive reference. The model is integrated with explicit
   forward Euler at a fixed step. Integrating the same equations with an adaptive
   stiff solver at tight tolerances, splitting at the two discontinuities of the
   rectangular alert, gives an independent answer to compare against.
3. Step refinement in the two-community model, on attack rates, on the targeting
   advantage, and on the dose totals.

The stepper in `_trace` is a copy of sim_calibrated.run_single that also records
the compartments. Check 0 below proves the copy is exact before anything is
concluded from it: if it ever diverges, this script stops.
"""
import numpy as np
from scipy.integrate import solve_ivp

from sim_calibrated import CAL, run_single, win
import two_sigma_report as R

P = dict(CAL)
TIMELY = lambda t: win(t, 6, 26, 0.12)
ALERT_LO, ALERT_HI, ALERT_AMP = 6.0, 26.0, 0.12
POP = 100_000


def _trace(p, M=lambda t: 0.0, T=180.0, dt=0.02):
    """sim_calibrated.run_single, with the compartments recorded."""
    n = int(T / dt)
    S = p['S0']; E = 0.0; I = p['I0']; Rc = 1 - p['S0'] - p['I0']
    Rep = 0.0; A = 0.0; Pr = 0.0; V = 0.0; C = 0.0
    tot = np.empty(n); mins = np.empty(n)
    for k in range(n):
        t = k * dt
        beta_eff = p['R0'] * p['gamma'] * (1 - p['phi'] * V)
        inc = beta_eff * S * I
        vac = p['nu'] * V * S
        onset = p['sigma_lat'] * E
        g = p['mMod'] * Pr * Pr / (Pr * Pr + p['kh'] ** 2)
        dS = -inc - vac; dE = inc - onset; dI = onset - p['gamma'] * I
        dR = p['gamma'] * I + vac
        dRep = (onset - Rep) / p['tau_rep']
        dA = p['aIn'] * Rep + p['aR1'] * Pr - p['aDec'] * A
        dP = p['sigma'] * p['kP'] * A * (1 - Pr) - p['pDec'] * Pr + M(t) * (1 - Pr)
        dV = p['kV'] * g * (1 - V) - p['vWane'] * V
        tot[k] = S + E + I + Rc
        mins[k] = min(S, E, I, Rc)
        S += dS * dt; E += dE * dt; I += dI * dt; Rc += dR * dt
        Rep += dRep * dt; A += dA * dt; Pr += dP * dt; V += dV * dt; C += inc * dt
    return dict(total=tot, minimum=mins, attack=C, steps=n)


def _rhs(t, y, p, amp):
    S, E, I, Rc, Rep, A, Pr, V, C = y
    beta_eff = p['R0'] * p['gamma'] * (1 - p['phi'] * V)
    inc = beta_eff * S * I
    vac = p['nu'] * V * S
    onset = p['sigma_lat'] * E
    g = p['mMod'] * Pr * Pr / (Pr * Pr + p['kh'] ** 2)
    return [-inc - vac, inc - onset, onset - p['gamma'] * I, p['gamma'] * I + vac,
            (onset - Rep) / p['tau_rep'],
            p['aIn'] * Rep + p['aR1'] * Pr - p['aDec'] * A,
            p['sigma'] * p['kP'] * A * (1 - Pr) - p['pDec'] * Pr + amp * (1 - Pr),
            p['kV'] * g * (1 - V) - p['vWane'] * V,
            inc]


def adaptive(p, alerted, T=180.0):
    """Adaptive stiff reference, integrated in three legs so the rectangular
       alert is never differentiated across its own discontinuity."""
    y = [p['S0'], 0.0, p['I0'], 1 - p['S0'] - p['I0'], 0.0, 0.0, 0.0, 0.0, 0.0]
    legs = [(0.0, ALERT_LO, 0.0), (ALERT_LO, ALERT_HI, ALERT_AMP if alerted else 0.0),
            (ALERT_HI, T, 0.0)]
    for t0, t1, amp in legs:
        sol = solve_ivp(_rhs, (t0, t1), y, args=(p, amp), method="Radau",
                        rtol=1e-10, atol=1e-12, dense_output=False)
        y = list(sol.y[:, -1])
    return y[8]


def main():
    print("=" * 78)
    print("CHECK 0. IS THE TRACED STEPPER IDENTICAL TO THE ONE USED FOR THE RESULTS?")
    print("=" * 78)
    worst = 0.0
    for lbl, M in (("no alert", lambda t: 0.0), ("timely alert", TIMELY)):
        a = run_single(P, M=M)["attack"]; b = _trace(P, M=M)["attack"]
        worst = max(worst, abs(a - b))
        print(f"  {lbl:14s} run_single {a:.15f}   traced {b:.15f}   diff {abs(a-b):.3e}")
    if worst != 0.0:
        raise SystemExit("STOP: the traced stepper is not identical, nothing below is valid")
    print("  identical to the last bit; the checks below are on the same model\n")

    print("=" * 78)
    print("CHECK 1. CONSERVATION")
    print("=" * 78)
    for lbl, M in (("no alert", lambda t: 0.0), ("timely alert", TIMELY)):
        r = _trace(P, M=M)
        dev = float(np.max(np.abs(r["total"] - 1.0)))
        neg = float(np.min(r["minimum"]))
        print(f"  {lbl:14s} steps {r['steps']}   max |S+E+I+R - 1| = {dev:.2e}   "
              f"smallest compartment = {neg:.3e}")

    print("\n" + "=" * 78)
    print("CHECK 2. AGREEMENT WITH AN ADAPTIVE STIFF REFERENCE (Radau, rtol 1e-10)")
    print("=" * 78)
    fixed_base = run_single(P)["attack"]
    fixed_tim = run_single(P, M=TIMELY)["attack"]
    ad_base = adaptive(P, alerted=False)
    ad_tim = adaptive(P, alerted=True)
    d1, d2 = abs(fixed_base - ad_base), abs(fixed_tim - ad_tim)
    av_fixed = 100 * (fixed_base - fixed_tim) / fixed_base
    av_ad = 100 * (ad_base - ad_tim) / ad_base
    print(f"  cumulative incidence, no alert : fixed {fixed_base:.8f}  adaptive {ad_base:.8f}  diff {d1:.2e}")
    print(f"  cumulative incidence, alerted  : fixed {fixed_tim:.8f}  adaptive {ad_tim:.8f}  diff {d2:.2e}")
    print(f"  largest difference             : {max(d1, d2):.2e}")
    print(f"  percent averted, fixed step    : {av_fixed:.4f}%")
    print(f"  percent averted, adaptive      : {av_ad:.4f}%")
    print(f"  difference on percent averted  : {abs(av_fixed - av_ad):.4f} percentage points")

    print("\n" + "=" * 78)
    print("CHECK 3. STEP REFINEMENT IN THE TWO-COMMUNITY MODEL  (sigA 0.12, sigB 0.04)")
    print("=" * 78)
    BC = R.DOSE_BC; TG = BC / R.FB
    arms = {"none": (lambda t: 0.0, lambda t: 0.0),
            "broadcast": (lambda t: win(t, R.WIN_LO, R.WIN_HI, BC),
                          lambda t: win(t, R.WIN_LO, R.WIN_HI, BC)),
            "targeted": (lambda t: 0.0, lambda t: win(t, R.WIN_LO, R.WIN_HI, TG))}

    def run(dt, MA, MB):
        return R.run_ts(P, 0.12, 0.04, M_A=MA, M_B=MB, dt=dt)

    print(f"  {'arm':10s} {'dt':>7s} {'attack A %':>11s} {'attack B %':>11s} {'doses A':>9s} {'doses B':>9s}")
    ref, fine = {}, {}
    for dt, store in ((0.02, ref), (0.0025, fine)):
        for k, (MA, MB) in arms.items():
            r = run(dt, MA, MB)
            store[k] = (r["attackA"] * 100, r["attackB"] * 100,
                        r["dosesA"] * R.FA * POP, r["dosesB"] * R.FB * POP)
            print(f"  {k:10s} {dt:7.4f} {store[k][0]:11.5f} {store[k][1]:11.5f} "
                  f"{store[k][2]:9.2f} {store[k][3]:9.2f}")
    da = max(max(abs(ref[k][i] - fine[k][i]) for i in (0, 1)) for k in arms)
    adv_ref = 100 * (ref["none"][1] - ref["broadcast"][1]) / ref["none"][1]
    adv_fine = 100 * (fine["none"][1] - fine["broadcast"][1]) / fine["none"][1]
    tg_ref = 100 * (ref["none"][1] - ref["targeted"][1]) / ref["none"][1]
    tg_fine = 100 * (fine["none"][1] - fine["targeted"][1]) / fine["none"][1]
    dadv = abs((tg_ref - adv_ref) - (tg_fine - adv_fine))
    print(f"\n  largest change in any attack rate, 0.02 to 0.0025 : {da:.4f} percentage points")
    print(f"  change in the targeting advantage                 : {dadv:.4f} percentage points")

    fine2 = {}
    for k, (MA, MB) in arms.items():
        r = run(0.001, MA, MB)
        fine2[k] = (r["dosesA"] * R.FA * POP, r["dosesB"] * R.FB * POP)
    dd = max(max(abs(ref[k][2 + i] - fine2[k][i]) for i in (0, 1)) for k in arms)
    dd2 = max(max(abs(ref[k][2 + i] - fine[k][2 + i]) for i in (0, 1)) for k in arms)
    print(f"  largest change in any dose total, 0.02 to 0.0025  : {dd2:.2f} per 100,000")
    print(f"  largest change in any dose total, 0.02 to 0.001   : {dd:.2f} per 100,000")


if __name__ == "__main__":
    main()
