"""
Calibrated SEIR version of the Closing-the-Loop model.
Anchored to the 2017 Minnesota Somali-American measles outbreak:
  R0 = 15 (measles, band 12-18; Guerra 2017), latent ~10 d, infectious ~8 d (CDC; Vink 2014),
  S0 = 0.60 (MMR coverage in the pocket ~36%; Hall MMWR 2017).
Information side (attention, perceived risk, behavior) unchanged from Appendix A (illustrative).
"""
import numpy as np

# NOTE: these are the PUBLISHED Appendix A values. Keep this dict and the manuscript in sync.
CAL = dict(R0=15.0, gamma=1/8.0, sigma_lat=1/10.0, phi=0.60, nu=0.08,
           S0=0.60, I0=0.001, tau_rep=5.0, aIn=8.0, aR1=0.15, aDec=0.20,
           kP=0.50, pDec=0.25, sigma=0.12, kV=1.0, vWane=0.04, mMod=0.90, kh=0.22)

def win(t, lo, hi, amp): return amp if (lo <= t < hi) else 0.0

def run_single(p, M=lambda t:0.0, T=180.0, dt=0.02):
    n=int(T/dt)
    S=p['S0']; E=0.0; I=p['I0']; R=1-p['S0']-p['I0']
    Rep=0.0; A=0.0; P=0.0; V=0.0; C=0.0
    ts=np.empty(n); Inc=np.empty(n); Ps=np.empty(n); As=np.empty(n); Cs=np.empty(n)
    for k in range(n):
        t=k*dt
        beta_eff=p['R0']*p['gamma']*(1-p['phi']*V)
        inc=beta_eff*S*I
        vac=p['nu']*V*S
        onset=p['sigma_lat']*E
        g=p['mMod']*P*P/(P*P+p['kh']**2)
        dS=-inc-vac; dE=inc-onset; dI=onset-p['gamma']*I; dR=p['gamma']*I+vac
        dRep=(onset-Rep)/p['tau_rep']
        dA=p['aIn']*Rep+p['aR1']*P-p['aDec']*A
        dP=p['sigma']*p['kP']*A*(1-P)-p['pDec']*P+M(t)*(1-P)
        dV=p['kV']*g*(1-V)-p['vWane']*V
        ts[k]=t; Inc[k]=inc; Ps[k]=P; As[k]=A; Cs[k]=C
        S+=dS*dt;E+=dE*dt;I+=dI*dt;R+=dR*dt;Rep+=dRep*dt;A+=dA*dt;P+=dP*dt;V+=dV*dt;C+=inc*dt
    return dict(t=ts, inc=Inc, P=Ps, A=As, C=Cs, attack=C, peak_day=ts[int(np.argmax(Inc))])

def run_two(p, M_A=lambda t:0.0, M_B=lambda t:0.0, T=180.0, dt=0.02,
            fA=0.80, fB=0.20, SA0=0.10, SB0=0.60, assort=0.70):
    n=int(T/dt)
    S=[SA0,SB0]; E=[0.0,0.0]; I=[p['I0'],p['I0']]; V=[0.0,0.0]; P=[0.0,0.0]
    Rep=0.0; A=0.0; C=[0.0,0.0]; f=[fA,fB]
    tsB=np.empty(n); incB=np.empty(n)
    for k in range(n):
        t=k*dt
        Ibar=f[0]*I[0]+f[1]*I[1]
        inc=[0.0,0.0]; vac=[0.0,0.0]; onset=[0.0,0.0]
        for i in (0,1):
            beta_eff=p['R0']*p['gamma']*(1-p['phi']*V[i])
            force=assort*I[i]+(1-assort)*Ibar
            inc[i]=beta_eff*S[i]*force; vac[i]=p['nu']*V[i]*S[i]; onset[i]=p['sigma_lat']*E[i]
        onset_tot=f[0]*onset[0]+f[1]*onset[1]
        Pbar=f[0]*P[0]+f[1]*P[1]
        dRep=(onset_tot-Rep)/p['tau_rep']; dA=p['aIn']*Rep+p['aR1']*Pbar-p['aDec']*A
        Mv=[M_A(t),M_B(t)]
        for i in (0,1):
            g=p['mMod']*P[i]*P[i]/(P[i]*P[i]+p['kh']**2)
            dS=-inc[i]-vac[i]; dE=inc[i]-onset[i]; dI=onset[i]-p['gamma']*I[i]
            dP=p['sigma']*p['kP']*A*(1-P[i])-p['pDec']*P[i]+Mv[i]*(1-P[i])
            dV=p['kV']*g*(1-V[i])-p['vWane']*V[i]
            S[i]+=dS*dt;E[i]+=dE*dt;I[i]+=dI*dt;P[i]+=dP*dt;V[i]+=dV*dt;C[i]+=inc[i]*dt
        Rep+=dRep*dt; A+=dA*dt
        tsB[k]=t; incB[k]=inc[1]
    return dict(attackA=C[0], attackB=C[1], t=tsB, incB=incB)

if __name__=="__main__":
    base=run_single(CAL)
    print(f"BASELINE: attack {base['attack']*100:.1f}%  peak day {base['peak_day']:.0f}")
    # sweep timely windows/amplitudes to find one that averts most (act during early growth)
    print("\nWindow/amplitude search (single population):")
    for (lo,hi,amp,name) in [(8,28,0.07,'timely wide'),(8,25,0.10,'timely'),(10,30,0.07,'timely b'),
                             (5,20,0.12,'timely early strong')]:
        r=run_single(CAL, M=lambda t,lo=lo,hi=hi,amp=amp: win(t,lo,hi,amp))
        av=100*(base['attack']-r['attack'])/base['attack']
        print(f"  {name:20} win[{lo},{hi}] amp {amp}: averted {av:5.1f}%  peakP {r['P'].max():.2f}")
