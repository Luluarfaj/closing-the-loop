#!/usr/bin/env python3
"""Two-community model with COMMUNITY-SPECIFIC attenuation (sigA, sigB).

Why this file exists
--------------------
In sim_calibrated.run_two the perceived-risk equation is

    dP[i] = sigma * kP * A * (1 - P[i]) - pDec * P[i] + M[i](t) * (1 - P[i])

Both the attention state A and the attenuation gain sigma are GLOBAL. With a
symmetric message the two communities therefore have bit-identical perceived
risk. The paper's argument is that community B sits in the downplayed regime,
which means its attention-to-perceived-risk link is ATTENUATED relative to
community A's. The original code does not represent that.

This file copies run_two and splits sigma into sigA and sigB. Nothing else in
the dynamics changes. sim_calibrated.py is NOT modified.

Setting sigA = sigB = p['sigma'] must reproduce the published numbers exactly.
That check is the first thing __main__ runs.

Run:  python3 sim_two_sigma.py
"""
import numpy as np

from sim_calibrated import CAL, run_two, win

# ---- call-site settings used to produce the published Table 2 / Table 3 ----
# (identical to make_figures.py lines 66-68 and to build_equity_sensitivity.py)
FB = 0.20
FA = 1.0 - FB
SA0 = 0.10
SB0 = 0.60
ASSORT = 0.70
DOSE_BC = 0.010            # broadcast per-capita intensity, everyone
DOSE_TG = DOSE_BC / FB     # same total budget concentrated on B -> 0.050
WIN_LO, WIN_HI = 6.0, 26.0

POP = 100_000
POP_A = int(FA * POP)      # 80,000
POP_B = int(FB * POP)      # 20,000


def run_two_sigma(p, sigA, sigB, M_A=lambda t: 0.0, M_B=lambda t: 0.0,
                  T=180.0, dt=0.02, fA=FA, fB=FB, SA0=SA0, SB0=SB0, assort=ASSORT):
    """run_two with a separate attenuation gain per community.

    sigA, sigB replace the single global p['sigma'] in the perceived-risk
    equation. Every other line is identical to sim_calibrated.run_two,
    including the Euler step and the ordering of the updates.

    Returns attack rates, the community-B incidence trace, and the perceived-risk
    traces for both communities so the peak perceived risk can be reported.
    """
    n = int(T / dt)
    S = [SA0, SB0]; E = [0.0, 0.0]; I = [p['I0'], p['I0']]
    V = [0.0, 0.0]; P = [0.0, 0.0]
    Rep = 0.0; A = 0.0; C = [0.0, 0.0]
    f = [fA, fB]
    sig = [sigA, sigB]
    tsB = np.empty(n); incB = np.empty(n); incA = np.empty(n)
    PA = np.empty(n); PB = np.empty(n); As = np.empty(n)
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
        Pbar = f[0] * P[0] + f[1] * P[1]
        dRep = (onset_tot - Rep) / p['tau_rep']
        dA = p['aIn'] * Rep + p['aR1'] * Pbar - p['aDec'] * A
        Mv = [M_A(t), M_B(t)]
        tsB[k] = t; incB[k] = inc[1]; incA[k] = inc[0]
        PA[k] = P[0]; PB[k] = P[1]; As[k] = A
        for i in (0, 1):
            g = p['mMod'] * P[i] * P[i] / (P[i] * P[i] + p['kh'] ** 2)
            dS = -inc[i] - vac[i]
            dE = inc[i] - onset[i]
            dI = onset[i] - p['gamma'] * I[i]
            dP = sig[i] * p['kP'] * A * (1 - P[i]) - p['pDec'] * P[i] + Mv[i] * (1 - P[i])
            dV = p['kV'] * g * (1 - V[i]) - p['vWane'] * V[i]
            S[i] += dS * dt; E[i] += dE * dt; I[i] += dI * dt
            P[i] += dP * dt; V[i] += dV * dt; C[i] += inc[i] * dt
        Rep += dRep * dt; A += dA * dt
    return dict(attackA=C[0], attackB=C[1], t=tsB, incB=incB, incA=incA,
                PA=PA, PB=PB, A=As, peakPA=float(PA.max()), peakPB=float(PB.max()))


def three_scenarios(p, sigA, sigB, **kw):
    """no campaign / broadcast / targeted, on an equal messaging budget.

    The targeted dose is DERIVED from fB (dose_tg = DOSE_BC / fB) so the total
    budget stays equal when fB is swept. Pinning it at 0.050 would silently
    hand the targeted arm a different budget at every fB other than 0.20.
    """
    fB_here = kw.get('fB', FB)
    dose_tg = DOSE_BC / fB_here
    none = run_two_sigma(p, sigA, sigB, **kw)
    bc = run_two_sigma(p, sigA, sigB,
                       M_A=lambda t: win(t, WIN_LO, WIN_HI, DOSE_BC),
                       M_B=lambda t: win(t, WIN_LO, WIN_HI, DOSE_BC), **kw)
    tg = run_two_sigma(p, sigA, sigB,
                       M_A=lambda t: 0.0,
                       M_B=lambda t: win(t, WIN_LO, WIN_HI, dose_tg), **kw)
    return none, bc, tg


def averted_B(none, r):
    if none['attackB'] <= 1e-9:
        return 0.0
    return 100.0 * (none['attackB'] - r['attackB']) / none['attackB']


def _table(tag, none, bc, tg):
    print(f"\n--- {tag} ---")
    print(f"{'scenario':<12}{'A %':>9}{'B %':>9}{'whole %':>10}"
          f"{'A n':>10}{'B n':>10}{'total n':>10}")
    for name, r in (("none", none), ("broadcast", bc), ("targeted", tg)):
        a = r['attackA']; b = r['attackB']
        whole = FA * a + FB * b
        print(f"{name:<12}{a*100:9.2f}{b*100:9.2f}{whole*100:10.2f}"
              f"{a*POP_A:10.0f}{b*POP_B:10.0f}{a*POP_A + b*POP_B:10.0f}")
    avbc = averted_B(none, bc); avtg = averted_B(none, tg)
    print(f"percent of B's outbreak averted: broadcast {avbc:.2f}  targeted {avtg:.2f}"
          f"  advantage {avtg - avbc:.2f} pts")
    print(f"peak perceived risk: A {none['peakPA']:.4f}  B {none['peakPB']:.4f} (no campaign)")
    return avbc, avtg


if __name__ == "__main__":
    P = dict(CAL)
    S = P['sigma']

    # ---------- 1. reproduction check against sim_calibrated.run_two ----------
    print("=" * 74)
    print(f"REPRODUCTION CHECK  sigA = sigB = {S}")
    print("=" * 74)
    kw = dict(fA=FA, fB=FB, SA0=SA0, SB0=SB0, assort=ASSORT)
    old_none = run_two(P, **kw)
    old_bc = run_two(P, M_A=lambda t: win(t, WIN_LO, WIN_HI, DOSE_BC),
                     M_B=lambda t: win(t, WIN_LO, WIN_HI, DOSE_BC), **kw)
    old_tg = run_two(P, M_A=lambda t: 0.0,
                     M_B=lambda t: win(t, WIN_LO, WIN_HI, DOSE_TG), **kw)
    new_none, new_bc, new_tg = three_scenarios(P, S, S)
    worst = 0.0
    print(f"{'scenario':<12}{'field':<9}{'sim_calibrated':>16}{'sim_two_sigma':>16}{'abs diff':>14}")
    for name, o, nw in (("none", old_none, new_none), ("broadcast", old_bc, new_bc),
                        ("targeted", old_tg, new_tg)):
        for fld in ("attackA", "attackB"):
            d = abs(o[fld] - nw[fld]); worst = max(worst, d)
            print(f"{name:<12}{fld:<9}{o[fld]*100:16.10f}{nw[fld]*100:16.10f}{d:14.3e}")
    print(f"worst absolute difference: {worst:.3e}"
          f"   -> {'EXACT MATCH' if worst == 0.0 else 'MISMATCH, STOP'}")
    _table(f"baseline, sigA = sigB = {S}", new_none, new_bc, new_tg)

    # ---------- 2. community-specific attenuation ----------
    # The gains used in the paper. Every simulation number reported in the manuscript
    # is produced by two_sigma_report.py, which calls run_two_sigma at these values.
    SIG_A, SIG_B = 0.12, 0.04
    print("\n" + "=" * 74)
    print(f"COMMUNITY-SPECIFIC ATTENUATION  sigA = {SIG_A}  sigB = {SIG_B}")
    print("=" * 74)
    cs_none, cs_bc, cs_tg = three_scenarios(P, SIG_A, SIG_B)
    _table(f"sigA = {SIG_A}, sigB = {SIG_B}", cs_none, cs_bc, cs_tg)
    print(f"peak perceived risk under broadcast: A {cs_bc['peakPA']:.4f}  B {cs_bc['peakPB']:.4f}")
    print(f"peak perceived risk under targeted : A {cs_tg['peakPA']:.4f}  B {cs_tg['peakPB']:.4f}")

    # ---------- 3. sweep sigB, holding sigA fixed ----------
    print("\n" + "=" * 74)
    print(f"SWEEP  sigB from 0.02 to 0.12, sigA held at {SIG_A}")
    print("=" * 74)
    print(f"{'sigB':>7}{'peakP_B':>10}{'A none':>9}{'B none':>9}{'A bc':>8}{'B bc':>8}"
          f"{'A tg':>8}{'B tg':>8}{'avert bc':>10}{'avert tg':>10}{'adv pts':>9}")
    for sigB in [round(0.02 + 0.005 * i, 3) for i in range(21)]:
        n_, b_, t_ = three_scenarios(P, SIG_A, sigB)
        avbc = averted_B(n_, b_); avtg = averted_B(n_, t_)
        print(f"{sigB:7.3f}{n_['peakPB']:10.4f}{n_['attackA']*100:9.2f}{n_['attackB']*100:9.2f}"
              f"{b_['attackA']*100:8.2f}{b_['attackB']*100:8.2f}"
              f"{t_['attackA']*100:8.2f}{t_['attackB']*100:8.2f}"
              f"{avbc:10.2f}{avtg:10.2f}{avtg - avbc:9.2f}")

    # ---------- 4. sensitivity to sigA as well ----------
    print("\n" + "=" * 74)
    print("SWEEP  sigA from 0.12 to 0.60, sigB held at 0.04")
    print("=" * 74)
    print(f"{'sigA':>7}{'peakP_A':>10}{'A none':>9}{'B none':>9}{'avert bc':>10}"
          f"{'avert tg':>10}{'adv pts':>9}")
    for sigA in [0.12, 0.20, 0.28, 0.36, 0.40, 0.44, 0.52, 0.60]:
        n_, b_, t_ = three_scenarios(P, sigA, 0.04)
        avbc = averted_B(n_, b_); avtg = averted_B(n_, t_)
        print(f"{sigA:7.3f}{n_['peakPA']:10.4f}{n_['attackA']*100:9.2f}{n_['attackB']*100:9.2f}"
              f"{avbc:10.2f}{avtg:10.2f}{avtg - avbc:9.2f}")

    # ---------- 5. isolating which sigma drives the targeting advantage ----------
    print("\n" + "=" * 74)
    print("ISOLATION  is the advantage driven by the A/B difference, or by sigB alone?")
    print("=" * 74)
    print(f"{'sigA':>7}{'sigB':>7}{'peakP_A':>10}{'peakP_B':>10}{'A none':>9}{'B none':>9}"
          f"{'avert bc':>10}{'avert tg':>10}{'adv pts':>9}")
    for sa, sb in ((0.12, 0.12), (0.40, 0.40), (0.04, 0.04), (0.40, 0.04), (0.04, 0.40)):
        n_, b_, t_ = three_scenarios(P, sa, sb)
        avbc = averted_B(n_, b_); avtg = averted_B(n_, t_)
        print(f"{sa:7.2f}{sb:7.2f}{n_['peakPA']:10.4f}{n_['peakPB']:10.4f}"
              f"{n_['attackA']*100:9.2f}{n_['attackB']*100:9.2f}"
              f"{avbc:10.2f}{avtg:10.2f}{avtg - avbc:9.2f}")

    # ---------- 6. equity sensitivity sweeps, budget-equal, under both settings ----------
    print("\n" + "=" * 74)
    print("EQUITY SENSITIVITY (Figure 6b sweeps) under both settings")
    print("=" * 74)
    grids = (("assort", [dict(assort=float(a)) for a in np.linspace(0.50, 0.90, 9)]),
             ("SA0", [dict(SA0=float(s)) for s in np.linspace(0.10, 0.55, 10)]),
             ("fB", [dict(fA=1 - float(f), fB=float(f)) for f in np.linspace(0.10, 0.35, 11)]))
    for tag, (sa, sb) in (("published sigma 0.12/0.12", (0.12, 0.12)),
                          ("community-specific 0.40/0.04", (0.40, 0.04))):
        print(f"\n  {tag}")
        allpairs = []
        for lab, kws in grids:
            pairs = []
            for kwx in kws:
                n_, b_, t_ = three_scenarios(P, sa, sb, **kwx)
                pairs.append((averted_B(n_, b_), averted_B(n_, t_)))
            allpairs += pairs
            a_ = [t - b for b, t in pairs]
            print(f"    {lab:<8} advantage {min(a_):5.1f} to {max(a_):5.1f} pts")
        a_ = [t - b for b, t in allpairs]
        wins = sum(1 for b, t in allpairs if t > b + 2)
        print(f"    OVERALL  advantage {min(a_):5.1f} to {max(a_):5.1f} pts; "
              f"targeting wins by more than 2 pts in {wins}/{len(allpairs)} settings")

    # ---------- 7. numerical robustness ----------
    print("\n" + "=" * 74)
    print("NUMERICAL CHECK  step-size sensitivity at sigA = 0.40, sigB = 0.04")
    print("=" * 74)
    print(f"{'dt':>9}{'A none':>10}{'B none':>10}{'B bc':>10}{'B tg':>10}{'adv pts':>10}")
    for dtx in (0.02, 0.01, 0.005, 0.0025):
        n_, b_, t_ = three_scenarios(P, 0.40, 0.04, dt=dtx)
        print(f"{dtx:9.4f}{n_['attackA']*100:10.4f}{n_['attackB']*100:10.4f}"
              f"{b_['attackB']*100:10.4f}{t_['attackB']*100:10.4f}"
              f"{averted_B(n_, t_) - averted_B(n_, b_):10.4f}")
