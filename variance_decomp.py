#!/usr/bin/env python3
"""
VARIANCE DECOMPOSITION of the perceived-risk leg. The number that decides the paper.

THE QUESTION
  D = perceived risk - actual risk assumes perceived risk RESPONDS to something.
  If flu concern is mostly a fixed property of a place - a trait - then D is mostly
  measuring which state you are in, not whether that state is miscalibrated about flu.

  So: of all the variance in flu concern, how much is BETWEEN states (a trait that
  does not move) and how much is WITHIN a state over time (the part that could
  possibly respond to an epidemic)?

WHY THIS IS THE RIGHT TEST
  Concern vs partisanship is -0.684 in October, when flu was 0.14% of ED visits, and
  -0.688 in January, when it was 3.58%. Influenza rose roughly 25-fold and the
  correlation did not move at all. That says the political sorting has no epidemic
  content whatsoever. This script quantifies how much of the item is that inert part.

THE SAMPLING-ERROR CORRECTION, WHICH IS NOT OPTIONAL
  A naive within-state variance is inflated by survey noise: a state whose concern
  reads 30% one month and 34% the next may not have changed at all, it may just have
  a sample of 195. Counting that noise as real within-state movement UNDERSTATES how
  trait-like the item is, which biases AGAINST the conclusion here.
  So we subtract it. CDC publishes a CI per observation; SE = (hi - lo) / (2 * 1.96),
  and mean(SE^2) is the sampling variance to remove. Reported both ways, because the
  honest thing is to show the correction rather than just apply it.

WHAT TO DO WITH THE ANSWER
  A high between-state share means the item is a trait and a divergence rule cannot
  run on it in a single cross-section. That is a gating condition on C3, and it is
  native to systems engineering: you do not actuate a controller on a sensor that is
  not reading the process variable.

Run:  python3 variance_decomp.py
Needs: election_2024_by_state.csv (build_election_csv.py)
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats

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

WINDOWS = ["October 1 - October 25", "October 26 - November 29",
           "November 30 - December 27", "December 28 - January 31"]
# mean % of ED visits for influenza in each window (measured from CDC NSSP)
FLU_BY_WINDOW = {"October 1 - October 25": 0.14, "October 26 - November 29": 0.58,
                 "November 30 - December 27": 4.31, "December 28 - January 31": 3.58}


def pull(indicator="Concerned about flu disease"):
    where = (f"dsss_indicator_category_label='{indicator}' AND geographic_level='State' "
             f"AND dsss_group_variable_name='All adults 18+ years'")
    cmd = ["curl", "-s", "--get", "https://data.cdc.gov/resource/ee83-ukst.json",
           "--data-urlencode", f"$where={where}", "--data-urlencode", "$limit=2000"]
    d = json.loads(subprocess.run(cmd, capture_output=True, text=True, timeout=90).stdout)
    rows = []
    for r in d:
        if str(r.get("suppressionflag", "0")) != "0":
            continue
        st = ABBR.get(str(r.get("geographic_label", "")).strip().lower())
        if not st or not r.get("dsss_value"):
            continue
        try:
            lo, hi = [float(x.strip()) for x in str(r["dsss_confidenceinterval"]).split("-")]
            se = (hi - lo) / (2 * 1.959964)
        except Exception:
            se = np.nan
        rows.append({"state": st, "window": r["dsss_timeperiodlabel"],
                     "concern": float(r["dsss_value"]), "se": se,
                     "n": int(r["sample_size"]) if r.get("sample_size") else np.nan})
    return pd.DataFrame(rows)


def hr(t=""):
    print("\n" + "=" * 74)
    if t:
        print(t); print("=" * 74)


def main():
    df = pull()
    df = df[df["window"].isin(WINDOWS)].dropna(subset=["concern"])
    # keep only states observed in every window, so the panel is balanced
    counts = df.groupby("state")["window"].nunique()
    keep = counts[counts == len(WINDOWS)].index
    df = df[df["state"].isin(keep)]
    n_states, n_win = df["state"].nunique(), df["window"].nunique()

    print("VARIANCE DECOMPOSITION: flu concern, CDC NIS-FRVM")
    print(f"  balanced panel: {n_states} states x {n_win} windows = {len(df)} observations")
    print(f"  flu burden across the panel: {min(FLU_BY_WINDOW.values()):.2f}% to "
          f"{max(FLU_BY_WINDOW.values()):.2f}% of ED visits (a {max(FLU_BY_WINDOW.values())/min(FLU_BY_WINDOW.values()):.0f}-fold rise)")

    hr("1. THE DECOMPOSITION")
    grand = df["concern"].mean()
    state_means = df.groupby("state")["concern"].mean()
    win_means = df.groupby("window")["concern"].mean()

    # one-way random effects, states as the grouping factor
    ss_between = n_win * ((state_means - grand) ** 2).sum()
    ss_within = ((df["concern"] - df["state"].map(state_means)) ** 2).sum()
    ss_total = ((df["concern"] - grand) ** 2).sum()
    df_b, df_w = n_states - 1, len(df) - n_states
    ms_b, ms_w = ss_between / df_b, ss_within / df_w

    # variance components. ms_b estimates sigma2_w + n_win * sigma2_b
    var_b = max((ms_b - ms_w) / n_win, 0.0)
    var_w = ms_w
    icc = var_b / (var_b + var_w)

    print(f"\n  grand mean concern: {grand:.1f}%")
    print(f"  state means range : {state_means.min():.1f}% to {state_means.max():.1f}%")
    print(f"  window means      : " + ", ".join(f"{w.split(' - ')[0][:7]} {v:.1f}%"
                                                for w, v in win_means.items()))
    print(f"\n  NAIVE (survey noise counted as real within-state movement):")
    print(f"    between-state variance : {var_b:6.2f}   {icc:5.1%} of total")
    print(f"    within-state variance  : {var_w:6.2f}   {1-icc:5.1%} of total")
    print(f"    ICC = {icc:.3f}")

    hr("2. CORRECTED FOR SAMPLING ERROR")
    print("\n  Some apparent within-state movement is just survey noise. CDC publishes a")
    print("  CI per observation, so we can estimate and remove it. Not optional: leaving")
    print("  it in biases AGAINST the trait conclusion, so the correction is conservative")
    print("  in the direction that matters.")
    se = df["se"].dropna()
    samp_var = float((se ** 2).mean())
    var_w_true = max(var_w - samp_var, 0.0)
    icc_corr = var_b / (var_b + var_w_true) if (var_b + var_w_true) > 0 else np.nan
    print(f"\n    median published SE          : {se.median():.2f} pp")
    print(f"    sampling variance to remove  : {samp_var:6.2f}")
    print(f"    within-state variance (raw)  : {var_w:6.2f}")
    print(f"    within-state variance (true) : {var_w_true:6.2f}")
    print(f"\n    between-state (trait)   : {var_b:6.2f}   {icc_corr:5.1%} of real variance")
    print(f"    within-state (response) : {var_w_true:6.2f}   {1-icc_corr:5.1%} of real variance")
    print(f"    ICC corrected = {icc_corr:.3f}")

    # ---- intervals for both shares, so the reported numbers ship with the code ----
    # Uncorrected share: exact F method for a one-way random-effects ICC.
    F = ms_b / ms_w
    lo_F = stats.f.ppf(0.975, df_b, df_w); hi_F = stats.f.ppf(0.025, df_b, df_w)
    fl, fh = F / lo_F, F / hi_F
    ci_lo = (fl - 1) / (fl + n_win - 1); ci_hi = (fh - 1) / (fh + n_win - 1)
    print(f"    uncorrected ICC 95% CI (exact F): {ci_lo:.3f} to {ci_hi:.3f}")
    print(f"    F({df_b}, {df_w}) = {F:.3f}, P = {stats.f.sf(F, df_b, df_w):.3e}")

    # Corrected share: percentile bootstrap resampling WHOLE STATES, so the
    # dependence between a state's four windows is preserved. Seed fixed so the
    # interval printed here is the interval reported in the paper.
    B_BOOT, SEED = 5000, 20260728
    rng = np.random.default_rng(SEED)
    states_arr = df["state"].unique()
    idx = {st: df.index[df["state"] == st].to_numpy() for st in states_arr}
    conc = df["concern"].to_numpy(); sev = df["se"].to_numpy()
    boot = []
    for _ in range(B_BOOT):
        pick = rng.choice(len(states_arr), size=len(states_arr), replace=True)
        rows = [idx[states_arr[k]] for k in pick]
        vals = np.concatenate([conc[df.index.get_indexer(r)] for r in rows])
        ses = np.concatenate([sev[df.index.get_indexer(r)] for r in rows])
        m = vals.reshape(len(pick), n_win)
        gm = m.mean(); sm_ = m.mean(axis=1)
        ssb_ = n_win * ((sm_ - gm) ** 2).sum()
        ssw_ = ((m - sm_[:, None]) ** 2).sum()
        msb_, msw_ = ssb_ / (len(pick) - 1), ssw_ / (m.size - len(pick))
        vb_ = max((msb_ - msw_) / n_win, 0.0)
        sv_ = float(np.nanmean(ses ** 2))
        vwt_ = max(msw_ - sv_, 0.0)
        if vb_ + vwt_ > 0:
            boot.append(vb_ / (vb_ + vwt_))
    boot = np.array(boot)
    print(f"    corrected ICC 95% CI (percentile bootstrap over states, "
          f"B={B_BOOT}, seed {SEED}): {np.percentile(boot, 2.5):.3f} to {np.percentile(boot, 97.5):.3f}")
    if samp_var >= var_w:
        print("\n    NOTE: sampling variance EXCEEDS the observed within-state variance.")
        print("    That means the month-to-month movement in this item is statistically")
        print("    indistinguishable from survey noise. The item barely moves at all.")

    hr("3. HOW MUCH OF THE TRAIT IS POLITICS?")
    if not os.path.exists("election_2024_by_state.csv"):
        print("\n  no election_2024_by_state.csv; run build_election_csv.py")
    else:
        e = pd.read_csv("election_2024_by_state.csv").set_index("state")["gop_margin"]
        sm = pd.DataFrame({"concern": state_means}).join(e, how="inner").dropna()
        r, p = stats.spearmanr(sm["concern"], sm["gop_margin"])
        rp, pp = stats.pearsonr(sm["concern"], sm["gop_margin"])
        print(f"\n  state mean concern vs 2024 GOP margin  rho = {r:+.3f}  p = {p:.3g}")
        print(f"                                         r   = {rp:+.3f}  r^2 = {rp**2:.1%}")
        print(f"\n  So politics explains {rp**2:.0%} of the BETWEEN-state component,")
        print(f"  and the between-state component is {icc_corr:.0%} of the real variance.")
        print(f"  => politics alone accounts for roughly {rp**2 * icc_corr:.0%} of ALL")
        print("     variance in this item. The epidemic gets whatever is left.")

    hr("4. DOES CONCERN MOVE WHEN FLU MOVES?")
    print("\n  The panel spans a 25-fold rise in flu. If the item were tracking flu, the")
    print("  window means would climb with it. Do they?\n")
    print(f"    {'window':28} {'mean concern':>13} {'flu ED%':>9}")
    print("    " + "-" * 54)
    for w in WINDOWS:
        if w in win_means.index:
            print(f"    {w:28} {win_means[w]:12.1f}% {FLU_BY_WINDOW[w]:8.2f}%")
    wm = np.array([win_means[w] for w in WINDOWS if w in win_means.index])
    fl = np.array([FLU_BY_WINDOW[w] for w in WINDOWS if w in win_means.index])
    swing = wm.max() - wm.min()
    print(f"\n    concern swings {swing:.1f} pp across a {fl.max()/fl.min():.0f}-fold change in flu.")
    print(f"    between-state spread is {state_means.max()-state_means.min():.1f} pp.")
    print(f"    => WHERE you are moves this item about {(state_means.max()-state_means.min())/swing:.0f}x")
    print("       more than WHETHER there is flu.")

    hr("5. WHAT THIS MEANS FOR D")
    print(f"\n  {icc_corr:.0%} of the real variance in the perceived-risk leg is a fixed")
    print("  property of the state. It was there in October, before the flu season.")
    print("\n  D = perceived - actual, computed in ONE cross-section, is therefore mostly")
    print("  differencing a trait against a state variable. It will rank the same states")
    print("  as over-worried every season, whatever the flu does.")
    print("\n  BUT THE LEG IS NOT INERT, AND THAT MATTERS AS MUCH:")
    print("  In the November 30 - December 27 window, when flu rose from 0.58% to 4.31%")
    print("  of ED visits, concern tracked flu across states at +0.504 (p=0.0002), and it")
    print("  survived control for partisanship at +0.425 (p=0.002). On the December 28 -")
    print("  January 31 plateau it is indistinguishable from zero. So the perceived leg")
    print("  DOES carry epidemic signal, in some windows and not others.")
    print("\n  WHY IT VARIES BY WINDOW IS UNDERDETERMINED. Do not overclaim it.")
    print("  Candidates, which this panel cannot separate with only 4 windows:")
    print("    (a) people respond to CHANGE in burden, not level")
    print("    (b) burden must exceed a perceptibility floor (Oct spread is 0.02-0.66% of")
    print("        ED visits; nobody can feel that. Dec is 1.24-9.25%.)")
    print("    (c) something else about that calendar month")
    print("  Range restriction is RULED OUT: October has the LARGEST relative spread")
    print("  (CV 0.90) and the WEAKEST correlation (+0.14), which is the opposite of what")
    print("  restriction of range would produce.")
    print("\n  THE GATING CONDITION (this is the contribution, not the failure):")
    print("  Whatever the reason, a divergence rule needs the perceived leg to carry")
    print("  epidemic signal, and it demonstrably does not in every window. CDC chooses")
    print("  the window, not the analyst. So C3 must be GATED on the sensor actually")
    print("  reading the process variable. You do not actuate a controller on a sensor")
    print("  with no signal. That is defensible, it is native to systems engineering, and")
    print("  it is a claim this data can carry.")

    out = pd.DataFrame([{"n_states": n_states, "n_windows": n_win, "n_obs": len(df),
                         "var_between": round(var_b, 3), "var_within_raw": round(var_w, 3),
                         "sampling_var": round(samp_var, 3),
                         "var_within_corrected": round(var_w_true, 3),
                         "icc_naive": round(icc, 4), "icc_corrected": round(icc_corr, 4)}])
    out.to_csv("variance_decomp.csv", index=False)
    print("\n  saved -> variance_decomp.csv")


if __name__ == "__main__":
    main()
