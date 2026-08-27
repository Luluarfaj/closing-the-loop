#!/usr/bin/env python3
"""Vaccine dose allocation under broadcast versus targeted messaging.

Reviewer request (Branch-Elliman): "show vaccine allocation to high versus low risk based on
a targeted versus general messaging approach", and the related counterpoint that "distributed
messaging and vaccination actually protects no one because people who aren't really at risk get
a limited resource."

The two-community model in sim_calibrated.run_two already computes the per-community
vaccination flow vac[i] = nu * V[i] * S[i], but it only returns attack rates. This script
re-implements the SAME integration (identical equations, identical Euler step, identical
parameters) and additionally accumulates cumulative doses delivered in each community:

    D[i] = integral over t of  nu * V[i](t) * S[i](t) dt      [fraction of community i]

D[i] is a per-capita fraction of community i, so counts are D[0]*80,000 (A) and
D[1]*20,000 (B) in a population of 100,000.

Scenarios, budget-equal exactly as build_equity_sensitivity.py sets them up:
  none      : no campaign
  broadcast : DOSE_BC = 0.010 per-capita alert intensity to EVERYONE, window [6, 26) days
  targeted  : the same total budget concentrated on B, i.e. DOSE_BC / fB = 0.050 to B only

This file does NOT modify sim_calibrated.py. It imports CAL and run_two from it only to
verify that the re-implementation reproduces the published attack rates exactly.

Run:  python3 dose_allocation.py
"""
import numpy as np

from sim_calibrated import CAL, run_two, win

# ---- base parameters used in the paper (Section 4.2.1 / Appendix A) ----
P = dict(CAL)
FB = 0.20
FA = 1.0 - FB
SA0 = 0.10
SB0 = 0.60
ASSORT = 0.70
DOSE_BC = 0.010          # broadcast per-capita intensity; total budget = DOSE_BC
DOSE_TG = DOSE_BC / FB   # same budget concentrated on B  -> 0.050
WIN_LO, WIN_HI = 6.0, 26.0

POP = 100_000
POP_A = int(FA * POP)    # 80,000
POP_B = int(FB * POP)    # 20,000

# published attack rates to check against (Table 2 of the manuscript), percent
PUBLISHED = {
    "none":      (5.9, 59.2),
    "broadcast": (2.5, 42.3),
    "targeted":  (5.0, 8.9),
}


def run_two_with_doses(p, M_A=lambda t: 0.0, M_B=lambda t: 0.0, T=180.0, dt=0.02,
                       fA=FA, fB=FB, SA0=SA0, SB0=SB0, assort=ASSORT):
    """Byte-for-byte the same dynamics as sim_calibrated.run_two, plus cumulative doses.

    Returns attackA/attackB (cumulative incidence as a fraction of each community) and
    dosesA/dosesB (cumulative vaccine doses delivered, as a fraction of each community).
    """
    n = int(T / dt)
    S = [SA0, SB0]; E = [0.0, 0.0]; I = [p['I0'], p['I0']]
    V = [0.0, 0.0]; Pr = [0.0, 0.0]
    Rep = 0.0; A = 0.0
    C = [0.0, 0.0]          # cumulative incidence
    D = [0.0, 0.0]          # cumulative doses  <-- the new quantity
    f = [fA, fB]
    tsB = np.empty(n); incB = np.empty(n)
    doseA_t = np.empty(n); doseB_t = np.empty(n)   # cumulative dose trajectories
    for k in range(n):
        t = k * dt
        Ibar = f[0] * I[0] + f[1] * I[1]
        inc = [0.0, 0.0]; vac = [0.0, 0.0]; onset = [0.0, 0.0]
        for i in (0, 1):
            beta_eff = p['R0'] * p['gamma'] * (1 - p['phi'] * V[i])
            force = assort * I[i] + (1 - assort) * Ibar
            inc[i] = beta_eff * S[i] * force
            vac[i] = p['nu'] * V[i] * S[i]
            onset[i] = p['sigma_lat'] * E[i]
        onset_tot = f[0] * onset[0] + f[1] * onset[1]
        Pbar = f[0] * Pr[0] + f[1] * Pr[1]
        dRep = (onset_tot - Rep) / p['tau_rep']
        dA = p['aIn'] * Rep + p['aR1'] * Pbar - p['aDec'] * A
        Mv = [M_A(t), M_B(t)]
        for i in (0, 1):
            g = p['mMod'] * Pr[i] * Pr[i] / (Pr[i] * Pr[i] + p['kh'] ** 2)
            dS = -inc[i] - vac[i]
            dE = inc[i] - onset[i]
            dI = onset[i] - p['gamma'] * I[i]
            dP = p['sigma'] * p['kP'] * A * (1 - Pr[i]) - p['pDec'] * Pr[i] + Mv[i] * (1 - Pr[i])
            dV = p['kV'] * g * (1 - V[i]) - p['vWane'] * V[i]
            S[i] += dS * dt; E[i] += dE * dt; I[i] += dI * dt
            Pr[i] += dP * dt; V[i] += dV * dt
            C[i] += inc[i] * dt
            D[i] += vac[i] * dt
        Rep += dRep * dt; A += dA * dt
        tsB[k] = t; incB[k] = inc[1]
        doseA_t[k] = D[0]; doseB_t[k] = D[1]
    return dict(attackA=C[0], attackB=C[1], dosesA=D[0], dosesB=D[1],
                t=tsB, incB=incB, doseA_t=doseA_t, doseB_t=doseB_t)


def scenarios():
    kw = dict(fA=FA, fB=FB, SA0=SA0, SB0=SB0, assort=ASSORT)
    return {
        "none": run_two_with_doses(P, **kw),
        "broadcast": run_two_with_doses(
            P,
            M_A=lambda t: win(t, WIN_LO, WIN_HI, DOSE_BC),
            M_B=lambda t: win(t, WIN_LO, WIN_HI, DOSE_BC), **kw),
        "targeted": run_two_with_doses(
            P,
            M_A=lambda t: 0.0,
            M_B=lambda t: win(t, WIN_LO, WIN_HI, DOSE_TG), **kw),
    }


def reference_attack_rates():
    """Same three scenarios through the UNMODIFIED sim_calibrated.run_two."""
    kw = dict(fA=FA, fB=FB, SA0=SA0, SB0=SB0, assort=ASSORT)
    return {
        "none": run_two(P, **kw),
        "broadcast": run_two(P,
                             M_A=lambda t: win(t, WIN_LO, WIN_HI, DOSE_BC),
                             M_B=lambda t: win(t, WIN_LO, WIN_HI, DOSE_BC), **kw),
        "targeted": run_two(P, M_A=lambda t: 0.0,
                            M_B=lambda t: win(t, WIN_LO, WIN_HI, DOSE_TG), **kw),
    }


def main():
    res = scenarios()
    ref = reference_attack_rates()

    label = {"none": "No campaign", "broadcast": "Broadcast, spread evenly",
             "targeted": "Targeted to community B"}

    print("=" * 96)
    print("PARAMETERS: fB=%.2f  SA0=%.2f  SB0=%.2f  assort=%.2f  "
          "DOSE_BC=%.3f  DOSE_TG=%.3f  window=[%g,%g)  dt=0.02  T=180"
          % (FB, SA0, SB0, ASSORT, DOSE_BC, DOSE_TG, WIN_LO, WIN_HI))
    print("Population 100,000:  A = %d (protected majority)   B = %d (under-protected)"
          % (POP_A, POP_B))
    print("=" * 96)

    # ---------- reproduction check ----------
    print("\n[1] REPRODUCTION CHECK against published Table 2 attack rates")
    print("%-26s %-22s %-22s %s" % ("scenario", "A: mine / published", "B: mine / published", "match"))
    all_ok = True
    for key in ("none", "broadcast", "targeted"):
        a = 100 * res[key]["attackA"]; b = 100 * res[key]["attackB"]
        pa, pb = PUBLISHED[key]
        ok = (abs(a - pa) <= 0.05) and (abs(b - pb) <= 0.05)
        all_ok &= ok
        # also confirm the re-implementation equals the untouched run_two
        da = abs(res[key]["attackA"] - ref[key]["attackA"])
        db = abs(res[key]["attackB"] - ref[key]["attackB"])
        assert da < 1e-12 and db < 1e-12, "re-implementation diverges from sim_calibrated.run_two"
        print("%-26s %6.2f%% / %5.1f%%        %6.2f%% / %5.1f%%        %s"
              % (label[key], a, pa, b, pb, "OK" if ok else "MISMATCH"))
    print("re-implementation == sim_calibrated.run_two to < 1e-12 for all three scenarios: YES")
    print("REPRODUCTION: %s" % ("PASSED (all within 0.05 pp of published)" if all_ok
                                else "*** FAILED - see mismatches above ***"))

    # ---------- dose allocation ----------
    print("\n[2] CUMULATIVE VACCINE DOSES, as a fraction of each community")
    print("%-26s %12s %12s" % ("scenario", "A (of 100%)", "B (of 100%)"))
    for key in ("none", "broadcast", "targeted"):
        print("%-26s %11.4f%% %11.4f%%"
              % (label[key], 100 * res[key]["dosesA"], 100 * res[key]["dosesB"]))

    print("\n[3] CUMULATIVE VACCINE DOSES, counts per 100,000 people")
    print("%-26s %10s %10s %10s %14s" % ("scenario", "A doses", "B doses", "total", "share to B"))
    rows = []
    for key in ("none", "broadcast", "targeted"):
        da = res[key]["dosesA"] * POP_A
        db = res[key]["dosesB"] * POP_B
        tot = da + db
        share = 100 * db / tot if tot > 1e-9 else float("nan")
        rows.append((key, da, db, tot, share))
        print("%-26s %10.0f %10.0f %10.0f %13.1f%%" % (label[key], da, db, tot, share))

    print("\n[4] DOSES PER INFECTION AVERTED (efficiency of the limited resource)")
    inf = lambda k: res[k]["attackA"] * POP_A + res[k]["attackB"] * POP_B
    base_inf = inf("none")
    for key in ("broadcast", "targeted"):
        averted = base_inf - inf(key)
        extra_doses = (rows[[r[0] for r in rows].index(key)][3]
                       - rows[0][3])
        print("%-26s infections averted %7.0f   extra doses vs no campaign %7.0f   "
              "doses per infection averted %6.2f"
              % (label[key], averted, extra_doses,
                 extra_doses / averted if averted > 1e-9 else float("nan")))

    print("\n[5] INFECTIONS per 100,000 (context, matches published Table 3)")
    print("%-26s %10s %10s %10s" % ("scenario", "A", "B", "total"))
    for key in ("none", "broadcast", "targeted"):
        a = res[key]["attackA"] * POP_A; b = res[key]["attackB"] * POP_B
        print("%-26s %10.0f %10.0f %10.0f" % (label[key], a, b, a + b))

    return res


if __name__ == "__main__":
    main()
