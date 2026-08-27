#!/usr/bin/env python3
"""Every number the manuscript needs under the community-specific attenuation fix.

Dynamics are IDENTICAL to sim_calibrated.run_two except that the single global
attenuation gain p['sigma'] is replaced by a per-community pair (sigA, sigB).
Cumulative vaccine doses are accumulated exactly as in dose_allocation.py:
    D[i] = integral of nu * V[i](t) * S[i](t) dt.

Nothing in sim_calibrated.py or dose_allocation.py is modified.
"""
import numpy as np
from sim_calibrated import CAL, run_two, win

FB = 0.20; FA = 1.0 - FB
SA0 = 0.10; SB0 = 0.60; ASSORT = 0.70
DOSE_BC = 0.010
WIN_LO, WIN_HI = 6.0, 26.0
POP = 100_000; POP_A = int(FA * POP); POP_B = int(FB * POP)
P = dict(CAL)


def run_ts(p, sigA, sigB, M_A=lambda t: 0.0, M_B=lambda t: 0.0, T=180.0, dt=0.02,
           fA=FA, fB=FB, SA0=SA0, SB0=SB0, assort=ASSORT):
    n = int(T / dt)
    S = [SA0, SB0]; E = [0.0, 0.0]; I = [p['I0'], p['I0']]
    V = [0.0, 0.0]; Pr = [0.0, 0.0]
    Rep = 0.0; A = 0.0; C = [0.0, 0.0]; D = [0.0, 0.0]
    f = [fA, fB]; sig = [sigA, sigB]
    PA = np.empty(n); PB = np.empty(n)
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
        PA[k] = Pr[0]; PB[k] = Pr[1]
        for i in (0, 1):
            g = p['mMod'] * Pr[i] * Pr[i] / (Pr[i] * Pr[i] + p['kh'] ** 2)
            dS = -inc[i] - vac[i]
            dE = inc[i] - onset[i]
            dI = onset[i] - p['gamma'] * I[i]
            dP = sig[i] * p['kP'] * A * (1 - Pr[i]) - p['pDec'] * Pr[i] + Mv[i] * (1 - Pr[i])
            dV = p['kV'] * g * (1 - V[i]) - p['vWane'] * V[i]
            S[i] += dS * dt; E[i] += dE * dt; I[i] += dI * dt
            Pr[i] += dP * dt; V[i] += dV * dt
            C[i] += inc[i] * dt; D[i] += vac[i] * dt
        Rep += dRep * dt; A += dA * dt
    return dict(attackA=C[0], attackB=C[1], dosesA=D[0], dosesB=D[1],
                peakPA=float(PA.max()), peakPB=float(PB.max()))


def three(p, sigA, sigB, **kw):
    fB_here = kw.get('fB', FB)
    dose_tg = DOSE_BC / fB_here
    none = run_ts(p, sigA, sigB, **kw)
    bc = run_ts(p, sigA, sigB, M_A=lambda t: win(t, WIN_LO, WIN_HI, DOSE_BC),
                M_B=lambda t: win(t, WIN_LO, WIN_HI, DOSE_BC), **kw)
    tg = run_ts(p, sigA, sigB, M_A=lambda t: 0.0,
                M_B=lambda t: win(t, WIN_LO, WIN_HI, dose_tg), **kw)
    return none, bc, tg


def avB(none, r):
    return 100.0 * (none['attackB'] - r['attackB']) / none['attackB'] if none['attackB'] > 1e-9 else 0.0


def tot(r, popA=POP_A, popB=POP_B):
    return r['attackA'] * popA + r['attackB'] * popB


def full(tag, sigA, sigB):
    none, bc, tg = three(P, sigA, sigB)
    print("\n" + "=" * 88)
    print(f"{tag}   sigA = {sigA}  sigB = {sigB}")
    print("=" * 88)
    print(f"{'campaign':<26}{'A %':>8}{'B %':>8}{'whole %':>10}{'A n':>9}{'B n':>9}{'total n':>10}")
    for nm, r in (("No campaign", none), ("Broadcast", bc), ("Targeted to B", tg)):
        w = FA * r['attackA'] + FB * r['attackB']
        print(f"{nm:<26}{r['attackA']*100:8.2f}{r['attackB']*100:8.2f}{w*100:10.2f}"
              f"{r['attackA']*POP_A:9.0f}{r['attackB']*POP_B:9.0f}{tot(r):10.0f}")
    base = tot(none)
    print(f"\npercent of B's outbreak averted : broadcast {avB(none,bc):6.2f}   targeted {avB(none,tg):6.2f}"
          f"   advantage {avB(none,tg)-avB(none,bc):.2f} pts")
    print(f"percent of ALL infections averted: broadcast {100*(base-tot(bc))/base:6.2f}"
          f"   targeted {100*(base-tot(tg))/base:6.2f}")
    print(f"\n{'campaign':<26}{'peakP A':>10}{'peakP B':>10}{'A/B ratio':>11}")
    for nm, r in (("No campaign", none), ("Broadcast", bc), ("Targeted to B", tg)):
        print(f"{nm:<26}{r['peakPA']:10.4f}{r['peakPB']:10.4f}{r['peakPA']/r['peakPB']:11.2f}")
    print(f"\n{'campaign':<26}{'A doses %':>11}{'B doses %':>11}{'A doses':>9}{'B doses':>9}"
          f"{'total':>9}{'share B':>10}")
    for nm, r in (("No campaign", none), ("Broadcast", bc), ("Targeted to B", tg)):
        da = r['dosesA'] * POP_A; db = r['dosesB'] * POP_B; t_ = da + db
        print(f"{nm:<26}{100*r['dosesA']:11.4f}{100*r['dosesB']:11.4f}{da:9.0f}{db:9.0f}{t_:9.0f}"
              f"{100*db/t_:9.1f}%")
    dn = none['dosesA'] * POP_A + none['dosesB'] * POP_B
    for nm, r in (("Broadcast", bc), ("Targeted to B", tg)):
        av = base - tot(r); extra = (r['dosesA'] * POP_A + r['dosesB'] * POP_B) - dn
        print(f"{nm:<26} infections averted {av:7.0f}  extra doses {extra:7.0f}"
              f"  doses per infection averted {extra/av:5.2f}")
    return none, bc, tg


if __name__ == "__main__":
    print("=" * 88)
    print("REPRODUCTION CHECK vs sim_calibrated.run_two, sigA = sigB = 0.12")
    print("=" * 88)
    kw = dict(fA=FA, fB=FB, SA0=SA0, SB0=SB0, assort=ASSORT)
    olds = (run_two(P, **kw),
            run_two(P, M_A=lambda t: win(t, WIN_LO, WIN_HI, DOSE_BC),
                    M_B=lambda t: win(t, WIN_LO, WIN_HI, DOSE_BC), **kw),
            run_two(P, M_A=lambda t: 0.0,
                    M_B=lambda t: win(t, WIN_LO, WIN_HI, DOSE_BC / FB), **kw))
    news = three(P, 0.12, 0.12)
    worst = 0.0
    for nm, o, nw in zip(("none", "broadcast", "targeted"), olds, news):
        for fld in ("attackA", "attackB"):
            d = abs(o[fld] - nw[fld]); worst = max(worst, d)
            print(f"  {nm:<10}{fld:<9}old {o[fld]*100:.10f}   new {nw[fld]*100:.10f}   diff {d:.3e}")
    print(f"worst abs diff {worst:.3e}  -> {'EXACT MATCH' if worst == 0.0 else 'MISMATCH'}")

    full("PUBLISHED BASELINE (global sigma)", 0.12, 0.12)
    full("CHOSEN FIX  minimal split", 0.12, 0.04)
    full("ALTERNATIVE  symmetric log split (geometric mean 0.12)", 0.36, 0.04)

    print("\n" + "=" * 88)
    print("SWEEP sigB, sigA held at the published 0.12")
    print("=" * 88)
    print(f"{'sigB':>7}{'A/B pk ratio':>14}{'peakP_A':>9}{'peakP_B':>9}{'A none':>8}{'B none':>8}"
          f"{'tot none':>10}{'avert bc':>10}{'avert tg':>10}{'adv pts':>9}{'all bc':>8}{'all tg':>8}")
    for sb in [0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12]:
        n_, b_, t_ = three(P, 0.12, sb)
        base = tot(n_)
        print(f"{sb:7.3f}{n_['peakPA']/n_['peakPB']:14.2f}{n_['peakPA']:9.4f}{n_['peakPB']:9.4f}"
              f"{n_['attackA']*100:8.2f}{n_['attackB']*100:8.2f}{base:10.0f}"
              f"{avB(n_,b_):10.2f}{avB(n_,t_):10.2f}{avB(n_,t_)-avB(n_,b_):9.2f}"
              f"{100*(base-tot(b_))/base:8.2f}{100*(base-tot(t_))/base:8.2f}")

    print("\n" + "=" * 88)
    print("SWEEP sigA, sigB held at 0.04")
    print("=" * 88)
    print(f"{'sigA':>7}{'peakP_A':>9}{'peakP_B':>9}{'A none':>8}{'B none':>8}{'tot none':>10}"
          f"{'avert bc':>10}{'avert tg':>10}{'adv pts':>9}{'all tg':>8}")
    for sa in [0.04,0.08,0.12,0.16,0.20,0.24,0.28,0.32,0.36,0.40,0.48,0.60]:
        n_, b_, t_ = three(P, sa, 0.04)
        base = tot(n_)
        print(f"{sa:7.3f}{n_['peakPA']:9.4f}{n_['peakPB']:9.4f}{n_['attackA']*100:8.2f}"
              f"{n_['attackB']*100:8.2f}{base:10.0f}{avB(n_,b_):10.2f}{avB(n_,t_):10.2f}"
              f"{avB(n_,t_)-avB(n_,b_):9.2f}{100*(base-tot(t_))/base:8.2f}")

    print("\n" + "=" * 88)
    print("EQUITY SENSITIVITY (Figure 6b sweeps), budget-equal, at sigA=0.12 sigB=0.04")
    print("=" * 88)
    grids = (("assort", [dict(assort=float(a)) for a in np.linspace(0.50, 0.90, 9)]),
             ("SA0", [dict(SA0=float(s)) for s in np.linspace(0.10, 0.55, 10)]),
             ("fB", [dict(fA=1-float(f), fB=float(f)) for f in np.linspace(0.10, 0.35, 11)]))
    for tag, (sa, sb) in (("published 0.12/0.12", (0.12, 0.12)),
                          ("fix 0.12/0.04", (0.12, 0.04)),
                          ("alt 0.36/0.04", (0.36, 0.04))):
        allp = []
        print(f"  {tag}")
        for lab, kws in grids:
            pr = []
            for kwx in kws:
                n_, b_, t_ = three(P, sa, sb, **kwx)
                pr.append((avB(n_, b_), avB(n_, t_)))
            allp += pr
            a_ = [t - b for b, t in pr]
            print(f"    {lab:<8} bc {min(b for b,_ in pr):5.1f}-{max(b for b,_ in pr):5.1f}"
                  f"  tg {min(t for _,t in pr):5.1f}-{max(t for _,t in pr):5.1f}"
                  f"  advantage {min(a_):5.1f} to {max(a_):5.1f} pts")
        a_ = [t - b for b, t in allp]
        wins = sum(1 for b, t in allp if t > b)
        print(f"    OVERALL advantage {min(a_):5.1f} to {max(a_):5.1f} pts; targeting wins {wins}/{len(allp)}")

    print("\n" + "=" * 88)
    print("STEP-SIZE CHECK at the chosen sigA=0.12 sigB=0.04")
    print("=" * 88)
    print(f"{'dt':>9}{'A none':>10}{'B none':>10}{'B bc':>10}{'B tg':>10}{'adv pts':>10}")
    for dtx in (0.02, 0.01, 0.005, 0.0025):
        n_, b_, t_ = three(P, 0.12, 0.04, dt=dtx)
        print(f"{dtx:9.4f}{n_['attackA']*100:10.4f}{n_['attackB']*100:10.4f}"
              f"{b_['attackB']*100:10.4f}{t_['attackB']*100:10.4f}{avB(n_,t_)-avB(n_,b_):10.4f}")

    print("\n" + "=" * 88)
    print("BEHAVIOURAL SWEEPS quoted in the Results (grid endpoints, as the manuscript quotes them)")
    print("=" * 88)
    def advpair(sa, sb, p=None, dose=DOSE_BC):
        pp = dict(P) if p is None else p
        n_ = run_ts(pp, sa, sb)
        b_ = run_ts(pp, sa, sb, M_A=lambda t: win(t, WIN_LO, WIN_HI, dose),
                    M_B=lambda t: win(t, WIN_LO, WIN_HI, dose))
        t_ = run_ts(pp, sa, sb, M_A=lambda t: 0.0,
                    M_B=lambda t: win(t, WIN_LO, WIN_HI, dose / FB))
        return avB(n_, t_) - avB(n_, b_)
    print("  kh (perceived risk at which vaccination engages), grid 0.05 to 0.40")
    old_kh = []; new_kh = []
    for kh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        pp = dict(P); pp['kh'] = kh
        o = advpair(0.12, 0.12, p=dict(pp)); nw = advpair(0.12, 0.04, p=dict(pp))
        old_kh.append(o); new_kh.append(nw)
        print(f"    kh {kh:.2f}   old adv {o:6.2f}   new adv {nw:6.2f}")
    print(f"    endpoints old {old_kh[0]:.1f} to {old_kh[-1]:.1f}  new {new_kh[0]:.1f} to {new_kh[-1]:.1f}")
    print(f"    true min-max old {min(old_kh):.1f} to {max(old_kh):.1f}  new {min(new_kh):.1f} to {max(new_kh):.1f}")
    print("  message dose, grid 0.005 to 0.050")
    old_d = []; new_d = []
    for d in [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050]:
        o = advpair(0.12, 0.12, dose=d); nw = advpair(0.12, 0.04, dose=d)
        old_d.append(o); new_d.append(nw)
        print(f"    dose {d:.3f}  old adv {o:6.2f}   new adv {nw:6.2f}")
    print(f"    endpoints old {old_d[-1]:.1f} to {old_d[0]:.1f}  new {new_d[-1]:.1f} to {new_d[0]:.1f}")
    print(f"    true min-max old {min(old_d):.1f} to {max(old_d):.1f}  new {min(new_d):.1f} to {max(new_d):.1f}")
