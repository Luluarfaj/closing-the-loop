#!/usr/bin/env python3
"""
D: the signed divergence between PERCEIVED risk and ACTUAL risk, per US state.

This is the measurement claim of the paper (component C2), computed on real federal
data for the 2025-26 flu season.

    D(state) = vdW(perceived risk) - vdW(actual risk)

    perceived = CDC NIS-FRVM, % of adults concerned about flu disease
    actual    = CDC NSSP, % of ED visits for influenza
    attention = Google Trends 'flu' interest        (NOT part of D; used to prove
                                                     perceived risk is not attention)

    D > 0  the state is more worried than its flu warrants   (over-worried)
    D < 0  the state's flu is worse than its worry           (under-worried)

WHY vdW (van der Waerden / inverse-normal rank scores) AND NOT PLAIN z
  Both legs are skewed and neither is normal. Plain z lets one outlier state drag the
  whole scale. Median/MAD divides by zero on zero-inflated data. Rank-based
  inverse-normal scores are robust to both and keep the two legs on a common,
  symmetric scale so the subtraction is meaningful. This also makes D invariant to
  any monotone rescaling of either leg, which is the point: D should not care whether
  burden is measured in ED-visit percent or ILI percent, only in the ORDERING.

WHAT THIS SCRIPT WILL TELL YOU THAT YOU MAY NOT WANT TO HEAR
  Flu concern is only weakly (and NEGATIVELY) related to actual flu, and is strongly
  related to political identity. The confound test below is not a robustness check
  bolted on at the end. It is the finding. Read it before writing the discussion.

HONEST LIMITS, STATED UP FRONT
  1. The bootstrap propagates uncertainty on the PERCEIVED leg only, because CDC
     publishes CIs for concern but Delphi publishes no CI for NSSP. So the intervals
     below are TOO NARROW. They are a lower bound on the true uncertainty.
  2. This is one cross-section of one flu season. States are not independent
     (regional flu waves, shared media markets), so the p-values are optimistic.
  3. D compares worry to CONTEMPORANEOUS burden. If a state warned people well, they
     worried, took precautions, and burden fell - so effective communication LOOKS
     like over-worry. This is a known open issue in the design, not a bug in the code.

Run:  python3 compute_d.py
Needs: flu_concern_by_state.csv   (concern_by_state.py)
       flu_ed_by_state.csv        (nssp_by_state.py)
       flu_trends_by_state.csv    (trends_by_state.py)
Optional: election_2024_by_state.csv  with columns state,gop_margin
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats

PEAK_WINDOW = "December 28 - January 31"
N_BOOT = 5000
SEED = 20260715

ABBR = {"alabama":"al","alaska":"ak","arizona":"az","arkansas":"ar","california":"ca",
"colorado":"co","connecticut":"ct","delaware":"de","district of columbia":"dc","florida":"fl",
"georgia":"ga","hawaii":"hi","idaho":"id","illinois":"il","indiana":"in","iowa":"ia",
"kansas":"ks","kentucky":"ky","louisiana":"la","maine":"me","maryland":"md",
"massachusetts":"ma","michigan":"mi","minnesota":"mn","mississippi":"ms","missouri":"mo",
"montana":"mt","nebraska":"ne","nevada":"nv","new hampshire":"nh","new jersey":"nj",
"new mexico":"nm","new york":"ny","north carolina":"nc","north dakota":"nd","ohio":"oh",
"oklahoma":"ok","oregon":"or","pennsylvania":"pa","rhode island":"ri","south carolina":"sc",
"south dakota":"sd","tennessee":"tn","texas":"tx","utah":"ut","vermont":"vt","virginia":"va",
"washington":"wa","west virginia":"wv","wisconsin":"wi","wyoming":"wy"}

# The 2024 presidential margins now come from election_2024_by_state.csv, built from
# the FEC's official results by build_election_csv.py. An earlier version of this file
# carried a hand-typed table from memory; it has been deleted. It was wrong: it had
# Oklahoma at +39 when the official two-party margin is +34.9. The confound survived
# the correction unchanged (rho -0.688 both ways), which is exactly why a big effect
# is not a licence to keep sloppy inputs - the next error might not be so forgiving.


def vdw(x):
    """van der Waerden / inverse-normal rank scores. Robust, symmetric, monotone-invariant."""
    x = np.asarray(x, dtype=float)
    r = stats.rankdata(x)                 # average ranks handle ties
    return stats.norm.ppf(r / (len(x) + 1.0))


def hr(t=""):
    print("\n" + "=" * 74)
    if t:
        print(t); print("=" * 74)


def load():
    need = ["flu_concern_by_state.csv", "flu_ed_by_state.csv", "flu_trends_by_state.csv"]
    miss = [f for f in need if not os.path.exists(f)]
    if miss:
        print("missing input files:", ", ".join(miss))
        print("\nrun these first:")
        if "flu_concern_by_state.csv" in miss: print("  python3 concern_by_state.py")
        if "flu_ed_by_state.csv" in miss:      print("  python3 nssp_by_state.py")
        if "flu_trends_by_state.csv" in miss:  print("  python3 trends_by_state.py")
        sys.exit(1)

    c = pd.read_csv("flu_concern_by_state.csv")
    c = c[c["window"] == PEAK_WINDOW][["state", "concern_pct", "ci_low", "ci_high", "sample_size"]]

    a = pd.read_csv("flu_ed_by_state.csv")
    a = a.rename(columns={a.columns[0]: "state", a.columns[1]: "actual"})[["state", "actual"]]

    t = pd.read_csv("flu_trends_by_state.csv"); t.columns = ["geoName", "trends"]
    t["state"] = t["geoName"].str.strip().str.lower().map(ABBR)
    t = t[["state", "trends"]]

    m = c.merge(a, on="state").merge(t, on="state").dropna(subset=["concern_pct", "actual", "trends"])
    return m.reset_index(drop=True)


def load_margins(m):
    """Real FEC data or nothing. No fallback: a made-up number is worse than no number."""
    if not os.path.exists("election_2024_by_state.csv"):
        print("\n  no election_2024_by_state.csv - skipping the confound test.")
        print("  build it:  python3 build_election_csv.py")
        m["gop_margin"] = np.nan
        return m, False
    e = pd.read_csv("election_2024_by_state.csv")
    e.columns = [x.strip().lower() for x in e.columns]
    m = m.merge(e[["state", "gop_margin"]], on="state", how="left")
    return m, True


def separability(m):
    """
    Is perceived risk just attention wearing a different hat?

    DO NOT judge this on the p-value. The question is not 'are they correlated at
    all' - with n=51 almost anything reaches significance. The question is 'are they
    REDUNDANT', which is a question about MAGNITUDE. Two measures sharing 14% of
    their variance are related and distinct. Two sharing 70% are the same variable.
    """
    hr("1. THE SEPARABILITY TEST  (make-or-break: is perceived risk just attention?)")
    r, p = stats.spearmanr(m["concern_pct"], m["trends"])
    shared = r ** 2
    print(f"\n  perceived (CDC concern) vs attention (Google Trends)")
    print(f"    rho = {r:+.3f}   p = {p:.4g}   n = {len(m)}")
    print(f"    shared variance = {shared:.1%}   ->  {1-shared:.0%} of concern is NOT attention")

    print("\n  WHY THIS DECIDES THE PAPER")
    print("  If perceived risk were just attention, D would collapse into")
    print("  attention-minus-actual, which is Gallotti's infodemic volume measure -")
    print("  already published, and the thing this paper argues against.")

    print("\n  READ THE MAGNITUDE, NOT THE P-VALUE")
    print("  The redundancy threshold agreed for this design is |rho| > 0.8 (shared")
    print("  variance > 64%), the point at which the two legs are one variable and R1")
    print("  becomes a tautology. Significance is not the test; with n=51 a trivial")
    print("  correlation clears p<0.05 and tells you nothing about redundancy.")

    if abs(r) > 0.8:
        print(f"\n  FAILS. rho = {r:+.3f}. The legs are redundant. D is not a new measure. Stop.")
    elif abs(r) > 0.5:
        print(f"\n  MARGINAL. rho = {r:+.3f} ({shared:.0%} shared). Below the redundancy")
        print("  threshold but uncomfortably high. Report it prominently and expect a fight.")
    else:
        print(f"\n  PASSES, but state it honestly. rho = {r:+.3f} is significant (p = {p:.3g}),")
        print(f"  so these legs ARE related - just not redundant. They share {shared:.0%} of")
        print(f"  their variance, leaving {1-shared:.0%} of concern unexplained by attention.")
        print("  DO NOT write 'perceived risk and attention are unrelated'. Write that they")
        print("  are modestly correlated and far from redundant. That claim survives review;")
        print("  the stronger one does not.")

    print("\n  THE ARGUMENT THAT DOES NOT DEPEND ON THIS COEFFICIENT AT ALL")
    print("  Concern is a survey proportion: worried respondents over respondents. There")
    print("  is no count anywhere in it, so attention-minus-actual cannot be constructed")
    print("  from it even in principle. That is a by-construction argument and it is")
    print("  stronger than any rho. Lead with it.")
    return r, p


def window_sensitivity(m):
    """
    The window is not a detail. It changed the headline. Show that it did.

    CDC fixes the perceived leg at 2025-12-28 to 2026-01-31, so every other leg has
    to come to that window. An earlier version of this pipeline compared a 13-week
    attention pull against the 5-week concern window and made the two legs look
    unrelated. They are not.
    """
    if not os.path.exists("flu_trends_by_state_WIDEWINDOW.csv"):
        return
    hr("1b. WINDOW SENSITIVITY  (how a mismatch faked the headline)")
    w = pd.read_csv("flu_trends_by_state_WIDEWINDOW.csv"); w.columns = ["geoName", "trends_wide"]
    w["state"] = w["geoName"].str.strip().str.lower().map(ABBR)
    b = m.merge(w[["state", "trends_wide"]], on="state").dropna(subset=["trends_wide"])
    r_align, p_align = stats.spearmanr(b["trends"], b["trends_wide"])
    r_bad, p_bad = stats.spearmanr(b["concern_pct"], b["trends_wide"])
    r_good, p_good = stats.spearmanr(b["concern_pct"], b["trends"])
    print(f"\n  Trends Dec-Feb (13wk) vs Trends Dec28-Jan31 (5wk)   rho = {r_align:+.3f}")
    print(f"\n  separability with MISMATCHED attention window       rho = {r_bad:+.3f}  p = {p_bad:.4g}")
    print(f"  separability with MATCHED attention window          rho = {r_good:+.3f}  p = {p_good:.4g}")
    print("\n  The mismatch cut the correlation roughly in half and flipped it from")
    print("  significant to not. If anyone asks why the number moved, this is why.")
    print("  Window alignment is a finding about method, and it belongs in the paper.")


def spec_curve():
    """
    THE MOST IMPORTANT SECTION IN THIS FILE. Read it before writing a single number
    into the manuscript.

    'Worry does not track flu' (rho -0.268) is NOT a robust finding. It depends on
    two arbitrary choices in the actual-risk leg - which weeks, and mean vs peak -
    and across those choices the correlation ranges from -0.37 to +0.40 with BOTH
    ENDS SIGNIFICANT. The same data supports 'worry is inversely related to burden'
    (p=0.008) and 'worry tracks burden' (p=0.003).

    Nobody cheated to get that. It is the garden of forking paths: a defensible
    choice at each fork, an indefensible result at the end. The only honest response
    is to pre-specify one primary spec, justify it, and report the whole curve so a
    reader can see the fragility instead of discovering it in review.

    PRIMARY SPEC, pre-specified and justified:
        window  = matched to the perceived leg (CDC fixes it at Dec 28 - Jan 31)
        summary = mean
    Justification: matching the window is forced - comparing worry in January to flu
    in February measures a lag nobody theorised. Mean over peak is the conservative
    choice: peak is one week and one outlier week moves a state's whole rank.

    ONE HALF-PATTERN, STATED NO STRONGER THAN IT DESERVES. Peak burden correlates
    POSITIVELY with worry in all four windows (+0.14 to +0.40). Mean burden does not
    have a stable sign (-0.27, +0.12, +0.07, -0.37). So the honest reading is that
    worry may relate to how bad flu GETS more consistently than to how bad it
    typically IS - a hypothesis worth testing, on a suggestion this thin, in a
    dataset built to test it. It is NOT a result, and it must not be harvested as
    the headline after the fact.
    """
    hr("2b. THE SPECIFICATION CURVE  (how fragile is the core finding?)")
    print("\n  Reported here so nobody discovers it in review. The perceived leg is")
    print("  fixed by CDC, so the only free choices are in the actual leg.\n")
    print(f"    {'actual-leg window':30} {'summary':8} {'rho':>8} {'p':>8}")
    print("    " + "-" * 58)
    print(f"    {'MATCHED (Dec 28 - Jan 31)':30} {'mean':8} {-0.268:+8.3f} {0.057:8.3f}   <- PRIMARY")
    print(f"    {'MATCHED (Dec 28 - Jan 31)':30} {'max':8} {+0.143:+8.3f} {0.317:8.3f}")
    print(f"    {'Dec 1 - Feb 28':30} {'mean':8} {+0.117:+8.3f} {0.415:8.3f}")
    print(f"    {'Dec 1 - Feb 28':30} {'max':8} {+0.402:+8.3f} {0.003:8.3f} *")
    print(f"    {'Dec 21 - Jan 31':30} {'mean':8} {+0.066:+8.3f} {0.648:8.3f}")
    print(f"    {'Dec 21 - Jan 31':30} {'max':8} {+0.402:+8.3f} {0.003:8.3f} *")
    print(f"    {'Jan 1 - Feb 28 (lagged)':30} {'mean':8} {-0.366:+8.3f} {0.008:8.3f} *")
    print(f"    {'Jan 1 - Feb 28 (lagged)':30} {'max':8} {+0.140:+8.3f} {0.328:8.3f}")
    print("\n    * p < 0.05   (measured 2026-07-15; rerun spec_curve_live.py to refresh)")
    print("\n  WHAT THIS MEANS")
    print("  The sign is not stable. Do NOT write 'worry does not track flu' as though")
    print("  it were established. Write: across 8 defensible specifications the")
    print("  worry-burden correlation ranges from -0.37 to +0.40, so the relationship")
    print("  is not robustly signed, and report the primary spec with this curve beside it.")
    print("\n  ONE HALF-PATTERN, WORTH A HYPOTHESIS AND NOT A CLAIM")
    print("  Peak burden is positively correlated with worry in ALL FOUR windows")
    print("  (+0.14 to +0.40). Mean burden has no stable sign. So worry may relate to")
    print("  how bad flu GETS more consistently than to how bad it typically IS.")
    print("  That is a hypothesis to test in a study designed for it - not a result,")
    print("  and not something to harvest as a headline after seeing this table.")


def core_correlations(m):
    hr("2. THE THREE LEGS")
    pairs = [("concern_pct", "actual", "perceived vs actual   (does worry track flu?)"),
             ("trends", "actual",      "attention vs actual   (does search track flu?)"),
             ("concern_pct", "trends", "perceived vs attention")]
    print()
    for a, b, lab in pairs:
        r, p = stats.spearmanr(m[a], m[b])
        sig = "*" if p < 0.05 else " "
        print(f"  {lab:52} rho = {r:+.3f}  p = {p:.4g} {sig}")
    print("\n  Read the first line carefully. If worry does not track flu, that is not a")
    print("  failed analysis. That is the paper: the infodemic is decoupled from the epidemic.")


def coverage_diagnostic(m):
    """Was the old +0.32 real, or an artifact of the missing 8 states? Show your work."""
    if not os.path.exists("flu_ili_by_state.csv"):
        return
    hr("3. THE COVERAGE DIAGNOSTIC  (why a published-looking number changed)")
    f = pd.read_csv("flu_ili_by_state.csv")
    f = f.rename(columns={f.columns[0]: "state", f.columns[1]: "ili"})[["state", "ili"]]
    old = m.merge(f, on="state")
    r_old, p_old = stats.spearmanr(old["trends"], old["ili"])
    r_new, p_new = stats.spearmanr(m["trends"], m["actual"])
    sub = m[m["state"].isin(f["state"])]
    r_sub, p_sub = stats.spearmanr(sub["trends"], sub["actual"])
    print(f"\n  attention vs actual, FluView, {len(old)}-state pool  rho = {r_old:+.3f}  p = {p_old:.4g}")
    print(f"  attention vs actual, NSSP,    {len(m)}-state pool  rho = {r_new:+.3f}  p = {p_new:.4g}")
    print(f"  attention vs actual, NSSP, SAME {len(sub)} states    rho = {r_sub:+.3f}  p = {p_sub:.4g}")
    print("\n  INTERPRETATION")
    print(f"  On the same states the two surveillance measures agree ({r_old:+.2f} vs {r_sub:+.2f}).")
    print("  So the change is NOT the measure. It is the 8 states FluView silently dropped")
    print("  (ak, ct, hi, ny, ok, or, sd, ut). Adding them back is what moves it.")
    print("  This HELPS the argument: weaker attention-actual coupling = more room for")
    print("  divergence. But the old number does not get quoted again.")


def confound(m, real_margins):
    hr("4. THE CONFOUND  (the finding nobody went looking for)")
    mm = m.dropna(subset=["gop_margin"])
    if not real_margins or not len(mm):
        print("\n  no margins loaded; skipped. run: python3 build_election_csv.py")
        return
    print("\n  margins: FEC official 2024 results, two-party basis (R-D)/(R+D)*100")

    # The 8 states FluView lost withheld ILI publication this season. If that
    # withholding were political, the missingness would be entangled with the very
    # confound being tested. It is not - but the check has to be run, not assumed.
    withheld = ["ak", "ct", "hi", "ny", "ok", "or", "sd", "ut"]
    w = mm[mm["state"].isin(withheld)]["gop_margin"]
    k = mm[~mm["state"].isin(withheld)]["gop_margin"]
    if len(w) and len(k):
        _, p_mw = stats.mannwhitneyu(w, k)
        print(f"\n  are the 8 ILI-withholding states politically distinctive?")
        print(f"    withheld  (n={len(w)}) mean margin {w.mean():+.1f}")
        print(f"    published (n={len(k)}) mean margin {k.mean():+.1f}   Mann-Whitney p = {p_mw:.3f}")
        print("    " + ("No. The missingness is not entangled with partisanship." if p_mw >= 0.05
                        else "YES - the missingness IS political. Serious problem."))
    print()
    for col, lab in [("concern_pct", "perceived (concern)"),
                     ("actual", "actual (NSSP)"),
                     ("trends", "attention (Trends)")]:
        r, p = stats.spearmanr(mm[col], mm["gop_margin"])
        print(f"  {lab:22} vs 2024 GOP margin    rho = {r:+.3f}  p = {p:.4g}")
    r_pol, _ = stats.spearmanr(mm["concern_pct"], mm["gop_margin"])
    r_act, _ = stats.spearmanr(mm["concern_pct"], mm["actual"])
    print(f"\n  Worry tracks POLITICS at {abs(r_pol):.2f} and FLU at {abs(r_act):.2f}.")
    print("  Flu concern is substantially a political-identity variable.")

    # Does the divergence survive within party? Residualize both legs on partisanship.
    print("\n  DOES THE DIVERGENCE SURVIVE WITHIN PARTY?")
    print("  (residualize both legs on partisan lean, then re-correlate)")
    pol = vdw(mm["gop_margin"])
    res_c = vdw(mm["concern_pct"]) - np.polyval(np.polyfit(pol, vdw(mm["concern_pct"]), 1), pol)
    res_a = vdw(mm["actual"]) - np.polyval(np.polyfit(pol, vdw(mm["actual"]), 1), pol)
    r_res, p_res = stats.spearmanr(res_c, res_a)
    print(f"    residualized concern vs residualized actual   rho = {r_res:+.3f}  p = {p_res:.4g}")
    if r_res < 0:
        print("    Still negative. Even comparing states to their OWN political peers,")
        print("    worry does not track local burden. The decoupling is not just partisanship.")
    print("\n  THE QUESTION THIS RAISES, WHICH IS YOURS AND NOT THE CODE'S:")
    print("  is partisanship a CONFOUND to control away, or is it the MECHANISM - the")
    print("  thing that organizes risk perception instead of local epidemic conditions?")
    print("  If it is the mechanism, that is a stronger paper than a clean D, and it is")
    print("  already your dissertation thesis: the infodemic decoupled from the epidemic.")


def compute(m):
    hr("5. D, THE DIVERGENCE  (with bootstrapped intervals)")
    m = m.copy()
    m["z_perceived"] = vdw(m["concern_pct"])
    m["z_actual"] = vdw(m["actual"])
    m["D_z"] = m["z_perceived"] - m["z_actual"]
    n = len(m)
    m["rank_perceived"] = stats.rankdata(m["concern_pct"])
    m["rank_actual"] = stats.rankdata(m["actual"])
    m["D_rank"] = m["rank_perceived"] - m["rank_actual"]

    # Bootstrap: redraw concern from CDC's published CI, recompute ranks.
    # Treats the published CI as symmetric-normal: sd = (hi - lo) / (2 * 1.96).
    have_ci = m["ci_low"].notna() & m["ci_high"].notna()
    if have_ci.all():
        rng = np.random.default_rng(SEED)
        sd = (m["ci_high"].values - m["ci_low"].values) / (2 * 1.959964)
        draws = rng.normal(m["concern_pct"].values[None, :], sd[None, :], size=(N_BOOT, n))
        boot = np.empty((N_BOOT, n))
        ra = stats.rankdata(m["actual"].values)
        for i in range(N_BOOT):
            boot[i] = stats.rankdata(draws[i]) - ra
        lo = np.percentile(boot, 2.5, axis=0)
        hi = np.percentile(boot, 97.5, axis=0)
        m["D_rank_lo"], m["D_rank_hi"] = lo, hi
        m["excludes_zero"] = (lo > 0) | (hi < 0)
        width = float(np.mean(hi - lo))
        print(f"\n  bootstrap: {N_BOOT} draws from CDC's published CIs, seed {SEED}")
        print(f"  median published CI half-width: {np.median(sd * 1.96):.2f} pp")
        print(f"  mean 95% CI width on D_rank   : {width:.1f} rank positions (out of {n})")
        print(f"  states whose CI excludes zero : {int(m['excludes_zero'].sum())} of {n}")
        print("\n  SO: the MIDDLE of this ranking is noise. Report the TAILS.")
        print("  AND REMEMBER: this propagates uncertainty on the PERCEIVED leg only,")
        print("  because NSSP publishes no CI. The true intervals are WIDER than these.")
    else:
        m["D_rank_lo"] = m["D_rank_hi"] = np.nan
        m["excludes_zero"] = False
        print("\n  (no CIs available; skipping the bootstrap)")

    m = m.sort_values("D_rank", ascending=False).reset_index(drop=True)

    print("\n  OVER-WORRIED  (worry outruns the flu)  D > 0")
    print(f"    {'st':3} {'D_rank':>7} {'95% CI':>16}  {'concern':>8} {'ED%':>6}")
    for _, r in m.head(8).iterrows():
        ci = f"[{r['D_rank_lo']:+.0f}, {r['D_rank_hi']:+.0f}]" if pd.notna(r["D_rank_lo"]) else ""
        mark = "*" if r["excludes_zero"] else " "
        print(f"    {r['state'].upper():3} {r['D_rank']:+7.0f} {ci:>16}{mark} "
              f"{r['concern_pct']:7.1f}% {r['actual']:5.2f}%")

    print("\n  UNDER-WORRIED  (the flu outruns the worry)  D < 0")
    print(f"    {'st':3} {'D_rank':>7} {'95% CI':>16}  {'concern':>8} {'ED%':>6}")
    for _, r in m.tail(8).iloc[::-1].iterrows():
        ci = f"[{r['D_rank_lo']:+.0f}, {r['D_rank_hi']:+.0f}]" if pd.notna(r["D_rank_lo"]) else ""
        mark = "*" if r["excludes_zero"] else " "
        print(f"    {r['state'].upper():3} {r['D_rank']:+7.0f} {ci:>16}{mark} "
              f"{r['concern_pct']:7.1f}% {r['actual']:5.2f}%")
    print("\n    * = 95% CI excludes zero")
    return m


def windows_of(path):
    """Read the epiweeks stamp a collector wrote into its CSV. None if unstamped."""
    try:
        d = pd.read_csv(path)
        return str(d["epiweeks"].iloc[0]) if "epiweeks" in d.columns else None
    except Exception:
        return None


def robustness(m):
    """Do the tails hold under the OTHER actual-risk measure? If yes, say so."""
    if not os.path.exists("flu_ili_by_state.csv"):
        return
    hr("6. ROBUSTNESS  (does D survive swapping the actual leg?)")

    # REFUSE to compare across mismatched windows. Silently mixing a 13-week pull with
    # a 5-week pull is exactly how the separability headline got faked once already,
    # and a wrong number that looks fine is worse than no number.
    w_nssp, w_ili = windows_of("flu_ed_by_state.csv"), windows_of("flu_ili_by_state.csv")
    if w_nssp and w_ili and w_nssp != w_ili:
        print(f"\n  SKIPPED - window mismatch.")
        print(f"    flu_ed_by_state.csv  (NSSP)    : epiweeks {w_nssp}")
        print(f"    flu_ili_by_state.csv (FluView) : epiweeks {w_ili}")
        print("  Comparing these would produce a real-looking, meaningless number.")
        print("  Window choice is NOT innocent: for the SAME NSSP measure, Dec-Feb and")
        print("  Jan rank states at only rho +0.54. Rerun:  python3 fluview_by_state.py")
        print("  (if Delphi rate-limits you, get a free key at")
        print("   api.delphi.cmu.edu/epidata/admin/registration_form and export DELPHI_API_KEY)")
        return
    if not w_ili:
        print("\n  SKIPPED - flu_ili_by_state.csv has no epiweeks stamp, so its window")
        print("  cannot be verified, which means this comparison cannot be trusted.")
        print("  An unverifiable window gets no number: that is the whole lesson here.")
        print("  Rerun:  python3 fluview_by_state.py   (it stamps the window now)")
        print("  (Delphi rate-limits anonymous queries. Free key:")
        print("   api.delphi.cmu.edu/epidata/admin/registration_form -> export DELPHI_API_KEY)")
        return

    f = pd.read_csv("flu_ili_by_state.csv")
    f = f.rename(columns={f.columns[0]: "state", f.columns[1]: "ili"})[["state", "ili"]]
    b = m.merge(f, on="state")
    d_nssp = vdw(b["concern_pct"]) - vdw(b["actual"])
    d_ili = vdw(b["concern_pct"]) - vdw(b["ili"])
    r, p = stats.spearmanr(d_nssp, d_ili)
    r_leg, p_leg = stats.spearmanr(b["actual"], b["ili"])
    print(f"\n  the two ACTUAL-RISK measures themselves, same {len(b)} states, same window")
    print(f"    NSSP (ED visits, influenza-coded) vs FluView (outpatient wILI)")
    print(f"    rho = {r_leg:+.3f}  p = {p_leg:.4g}")
    print(f"\n  D(NSSP) vs D(FluView wILI)")
    print(f"    rho = {r:+.3f}  p = {p:.4g}")

    # The tails are the claim; the middle is noise anyway. Test agreement where it counts.
    b2 = b.assign(dn=d_nssp, di=d_ili)
    for k in (5, 10):
        top = len(set(b2.nlargest(k, "dn")["state"]) & set(b2.nlargest(k, "di")["state"]))
        bot = len(set(b2.nsmallest(k, "dn")["state"]) & set(b2.nsmallest(k, "di")["state"]))
        print(f"    top-{k:<2} over-worried overlap {top}/{k}   "
              f"bottom-{k:<2} under-worried overlap {bot}/{k}")

    print("\n  HOW TO READ THIS")
    print("  These two measures are NOT interchangeable and were never meant to be.")
    print("  wILI is OUTPATIENT visits for influenza-LIKE ILLNESS - a syndrome that is")
    print("  mostly not influenza. NSSP is EMERGENCY-DEPARTMENT visits coded influenza -")
    print("  a diagnosis, in a different care setting. Modest agreement is expected.")
    if r > 0.8:
        print("  They agree strongly. The ranking is not an artifact of the measure.")
    else:
        print("  They agree only modestly, so D depends materially on which measure you")
        print("  pick. This is a REAL limitation and it goes in the paper, stated plainly:")
        print("  'actual risk' is not one thing, and the choice of surveillance measure is")
        print("  a researcher degree of freedom. Report D under both. If the TAILS overlap")
        print("  (see above) the headline states survive even though the ordering does not.")


def main():
    m = load()
    m, real_margins = load_margins(m)
    print(f"Closing the Loop - D on real data. {len(m)} states, flu season 2025-26.")
    print(f"  perceived : CDC NIS-FRVM % concerned about flu   ({PEAK_WINDOW})")
    print(f"  actual    : CDC NSSP % of ED visits for influenza")
    print(f"  attention : Google Trends 'flu'                  (not in D; used to test it)")

    separability(m)
    window_sensitivity(m)
    core_correlations(m)
    spec_curve()
    coverage_diagnostic(m)
    confound(m, real_margins)
    out = compute(m)
    robustness(out)

    cols = ["state", "concern_pct", "ci_low", "ci_high", "sample_size", "actual", "trends",
            "gop_margin", "z_perceived", "z_actual", "D_z", "D_rank",
            "D_rank_lo", "D_rank_hi", "excludes_zero"]
    out[cols].to_csv("divergence_D.csv", index=False)
    hr()
    print("  saved -> divergence_D.csv")
    if not real_margins:
        print("\n  TO DO before any of this is citable:")
        print("   1. Replace the memory-typed election margins (see MARGIN_FROM_MEMORY).")
        print("   2. Read the NIS-FRVM methods note: exact question wording, response")
        print("      scale, survey mode, weighting. The construct claim rests on wording")
        print("      nobody has read yet.")


if __name__ == "__main__":
    main()
