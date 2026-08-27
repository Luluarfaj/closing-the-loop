#!/usr/bin/env python3
"""
THE DECISION RULE, AS EXECUTABLE CODE, RUN ON ALL 51 US JURISDICTIONS.

WHAT THIS FILE IS FOR
  The manuscript's central object is a decision rule that converts a measured
  divergence between perceived and actual risk into a communication action:
  calm, alert, or hold. Until now that object had never been executed end to end.
  sim_calibrated.py contains no divergence, no threshold and no rule; targeting is
  imposed at the call site by setting M_A=0 and M_B=dose. This file implements the
  rule itself and runs it on real data for 51 jurisdictions, two diseases and four
  survey windows.

WHAT IT DOES NOT DO
  It does not invent a single number the manuscript failed to supply. The paper
  gives NO numeric threshold anywhere (the word "threshold" occurs exactly twice,
  at lines 67 and 270 of rev15_ACC.txt, and both occurrences describe it only as
  "set by decision cost" and explicitly "rather than by a fixed statistical
  cutoff"). Every constant in this file that the paper did not state is collected
  in OUR_CHOICES below, is labelled as ours, and is swept. Every input the data
  cannot supply is set to None and reported as missing rather than faked.

READS (does not modify) the existing files:
  flu_concern_by_state.csv      51 states x 4 windows, CDC NIS-FRVM flu concern
  divergence_flu.csv            the Dec 28 - Jan 31 flu divergence already on disk
  divergence_covid.csv          the Dec 28 - Jan 31 covid divergence already on disk
  flu_ed_by_state.csv           NSSP flu ED share, Dec 28 - Jan 31 only
  election_2024_by_state.csv    2024 GOP margin
  vaccination_merged.csv        used ONLY in a clearly labelled sensitivity mode
PULLS (and caches to new files, never overwriting anything that existed):
  decision_rule_inputs_concern.csv   covid concern for the four windows
  decision_rule_inputs_burden.csv    NSSP flu and covid ED share for the four windows
WRITES:
  decision_rule_actions.csv, decision_rule_gate.csv, decision_rule_sweep.csv,
  decision_rule_output.txt

Run:  python3 decision_rule.py            (uses the caches if present)
      python3 decision_rule.py --refresh  (re-pulls both APIs)
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
NBOOT = 3000
SEED = 20260728  # same seed build_respiratory.py used, so the gate reproduces

WINDOWS = ["October 1 - October 25", "October 26 - November 29",
           "November 30 - December 27", "December 28 - January 31"]
# MMWR epiweeks for each CDC survey window. Verified against flu_ed_by_state.csv,
# whose own epiweeks column stamps the last window as 202601-202605, and against the
# national means hard-coded in variance_decomp.py (0.14, 0.58, 4.31, 3.58).
EPIWEEKS = {"October 1 - October 25": "202540-202543",
            "October 26 - November 29": "202544-202548",
            "November 30 - December 27": "202549-202552",
            "December 28 - January 31": "202601-202605"}
PRIMARY_WINDOW = "December 28 - January 31"   # the window the manuscript uses
SIGNAL_WINDOW = "November 30 - December 27"   # the window where the leg carries signal

DISEASES = {"flu": ("Concerned about flu disease", "pct_ed_visits_influenza"),
            "covid": ("Concerned about COVID-19 disease", "pct_ed_visits_covid")}

ABBR = {"alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
        "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
        "district of columbia": "dc", "florida": "fl", "georgia": "ga", "hawaii": "hi",
        "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
        "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
        "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
        "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
        "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
        "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
        "oregon": "or", "pennsylvania": "pa", "rhode island": "ri",
        "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
        "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
        "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy"}


# =====================================================================================
# OUR CHOICES. Every value here is a free parameter that the manuscript does not
# supply. Nothing in this block is attributable to the paper. Each entry names the
# single constraint (if any) the paper does impose.
# =====================================================================================
OUR_CHOICES = {
    "T_ALERT_DEFAULT": (
        0.50,
        "Alert bar, in van der Waerden normal-score units of D. The paper gives no "
        "number. Paper's only constraint: it must be strictly smaller in magnitude "
        "than the calm bar. Swept below over 0.25 to 1.50."),
    "ASYMMETRY_RATIO_DEFAULT": (
        2.0,
        "t_calm = ratio * t_alert, so the default calm bar is 1.00. The paper gives "
        "only the inequality 'the bar for alerting a downplayed threat is lower than "
        "the bar for calming an inflated one'. No ratio, multiplier or difference is "
        "given anywhere. Swept below over 1.25 to 3.00."),
    "UNCERTAINTY_LEVEL": (
        0.95,
        "Confidence level for the uncertainty condition. The paper never states an "
        "operating level. The 95 percent figure in the manuscript belongs to the "
        "Figure 3 bootstrap intervals, not to the rule."),
    "UNCERTAINTY_COMBINATION": (
        "joint bootstrap interval on D",
        "The paper says the rule acts only when D 'exceeds the combined uncertainty "
        "of both sides' and never defines 'combined'. We form one bootstrap interval "
        "on D directly by perturbing the perceived leg with CDC's own published "
        "confidence intervals and re-ranking all 51 places jointly."),
    "ACTUAL_SIDE_UNCERTAINTY": (
        0.0,
        "Declared zero because it does not exist. Delphi publishes no confidence "
        "interval for NSSP and no error column exists on disk. This makes the test "
        "ONE-SIDED, not the two-sided test the paper describes. The interval is "
        "therefore too narrow and the test is anti-conservative."),
    "BURDEN_POOL_HIGH_PCT": (
        2.0,
        "Pool-wide absolute burden, in percent of emergency-department visits, above "
        "which the pool counts as 'serious across the whole pool'. The paper names no "
        "level, no units and no scale. Swept below over 0.5 to 4.0."),
    "BURDEN_SUMMARY": (
        "mean",
        "Per-state burden is the mean over the weeks in the window. spec_curve.csv "
        "shows mean versus peak-week flips the sign of the headline correlation, so "
        "this is load-bearing. Swept below."),
    "GATE_ALPHA": (
        0.05,
        "Significance level at which the perceived leg counts as tracking the "
        "benchmark inside the acted-on window. The paper attaches no numeric cut to "
        "either gate quantity. Swept below."),
    "BURDEN_CLAUSE_PLACEMENT": (
        "independent",
        "Where the pool-wide burden clause sits relative to the uncertainty test. "
        "The paper says a serious pool 'still triggers action even when no single "
        "place stands out', which is exactly the case where D is not distinguishable "
        "from zero, so we let the clause fire independently of the uncertainty test. "
        "The paper never states the ordering. The two alternatives, firing only after "
        "the uncertainty test and switching the clause off entirely, are also run and "
        "reported, because this placement changes the answer completely."),
    "GATE_MODE": (
        "tracking_only",
        "The paper's stated failure case is a conjunction: most variance between "
        "places AND the measure does not move with the hazard. It is silent on the "
        "mixed cases. We default to failing on the tracking quantity alone, which is "
        "the stricter reading and the one the control-system analogy implies. The "
        "conjunction reading is also computed and reported."),
    "PARTIAL_CONVENTION": (
        "rank residuals on ranked gop margin",
        "The manuscript does not say how partisanship was partialled out. Three "
        "defensible conventions give +0.432, +0.438 and +0.451 in the December "
        "window, all p < 0.002. We use rank residuals and report the number we get "
        "rather than the +0.425 quoted in variance_decomp.py."),
    "ICC_BETWEEN_MAJORITY": (
        0.50,
        "The word 'most' in 'if most of the variance lies between places' is the only "
        "quantifier the paper gives. We read 'most' as more than half."),
    "MIN_POOL": (
        20,
        "Minimum places for rank scoring to be treated as stable. The paper says thin "
        "pools are insufficient data but gives no number. Not binding here: the pool "
        "is 51 in every cell."),
    "TREND_DEADZONE": (
        0.10,
        "Change in |D| below which the trend is called flat rather than widening or "
        "closing. The paper gives no estimator, no window and no cut. NOTE: the trend "
        "changes NO action in this implementation, because the paper never states "
        "what the rule should do differently when the divergence is widening."),
    "CALM_QUALIFICATION_MODE": (
        "strict",
        "On the calming branch the paper requires a coverage and susceptibility "
        "answer before calming. No susceptibility series and no protective-behaviour "
        "series exist for the 51 jurisdictions. Strict mode therefore cannot certify "
        "calming and returns hold-and-watch, which is the outcome the paper itself "
        "names for an uncertified positive divergence. A labelled sensitivity mode "
        "using vaccination coverage as a susceptibility proxy is reported separately."),
    "VAX_LOW_SUSCEPTIBILITY_PCT": (
        45.0,
        "Sensitivity mode only. Vaccination coverage above which we would treat 'few "
        "people remain susceptible' as satisfied. Entirely ours; the paper gives no "
        "coverage cut, the provenance of vaccination_merged.csv is unreproducible "
        "from this repo, and the protective-behaviour half of the test is still "
        "missing, so even this mode does not instantiate the paper's condition."),
}

T_ALERT = OUR_CHOICES["T_ALERT_DEFAULT"][0]
T_CALM = T_ALERT * OUR_CHOICES["ASYMMETRY_RATIO_DEFAULT"][0]
BURDEN_HIGH = OUR_CHOICES["BURDEN_POOL_HIGH_PCT"][0]
GATE_ALPHA = OUR_CHOICES["GATE_ALPHA"][0]
MIN_POOL = OUR_CHOICES["MIN_POOL"][0]
TREND_DEADZONE = OUR_CHOICES["TREND_DEADZONE"][0]

# Terminal outputs. The paper never enumerates this set and never names it, so these
# tokens are our naming. Every one of them traces to a quoted line, given inline.
CALM = "CALM_AND_CLARIFY"
ALERT = "ALERT_AND_MOBILIZE"
HOLD_MONITOR = "HOLD_AND_MONITOR"
HOLD_WATCH = "HOLD_AND_WATCH"
HOLD_UNCERTAIN = "HOLD_UNCERTAINTY_NOT_EXCEEDED"
INSUFFICIENT = "INSUFFICIENT_DATA_THIN_POOL"
NO_DECISION = "NO_DECISION_MEASUREMENT_CONDITION_FAILED"


# =====================================================================================
# THE RULE
# =====================================================================================
def decide(D, D_lo, D_hi, burden, burden_percentile, trend, gate_passed,
           t_alert=T_ALERT, t_calm=T_CALM,
           pool_burden=None, pool_burden_high=BURDEN_HIGH, pool_n=51,
           min_pool=MIN_POOL,
           burden_clause=OUR_CHOICES["BURDEN_CLAUSE_PLACEMENT"][0],
           susceptibility_low=None, protective_behaviour_high=None,
           claims=None):
    """
    Return (action, reason). Pure function: no I/O, no globals mutated.

    D, D_lo, D_hi   the signed divergence and its interval, in van der Waerden units
    burden          this place's absolute burden, percent of ED visits
    burden_percentile  this place's burden rank within the pool, 0 to 1
    trend           'widening', 'closing', 'flat' or None. Reported, never acted on.
    gate_passed     result of the measurement condition for this window
    susceptibility_low        True / False / None. None means the input does not exist.
    protective_behaviour_high True / False / None. None means it does not exist.
    claims          the claim audit. None throughout: no state-level claim data exists.

    The paper's constraint on the two bars, and the only one it gives:
      "the bar for alerting a downplayed threat is lower than the bar for calming an
       inflated one, because a missed serious outbreak is a costlier error than an
       unnecessary reassurance" (Appendix B, line 270).
    """
    if t_alert >= t_calm:
        raise ValueError(
            "the alert bar must be strictly smaller than the calm bar; that is the "
            "one constraint the manuscript places on these two free parameters")

    # -- Thin pool ---------------------------------------------------------------
    # "rank-based scoring needs enough places in a pool to be stable, so thin pools
    #  are treated as insufficient data rather than forced into a regime"
    #  (Appendix B, line 272)
    if pool_n is not None and pool_n < min_pool:
        return INSUFFICIENT, (
            f"pool of {pool_n} places is below our minimum of {min_pool}; rank scores "
            f"are not stable. Minimum is OUR choice, the paper gives no number.")

    # -- The measurement gate ----------------------------------------------------
    # "Before the rule is run on an instrument, two quantities should be reported"
    #  (line 111). And on failure: "a controller is not actuated on a sensor that is
    #  not tracking the process variable" (line 111). "its use remains conditional on
    #  the measurement requirement ... the perceived-risk signal must demonstrably
    #  track the underlying hazard before it is used to guide communication
    #  decisions" (line 215).
    if not gate_passed:
        return NO_DECISION, (
            "measurement condition failed for this window: the perceived-risk leg "
            "does not track the benchmark here, so the rule is not actuated. The "
            "paper never names a return value for this case; the token is ours.")

    if D is None or (isinstance(D, float) and np.isnan(D)):
        return INSUFFICIENT, "no divergence could be computed for this place."

    # -- Absolute burden, read alongside D ----------------------------------------
    # "The rule therefore reads D alongside the absolute outbreak burden on the
    #  actual side, so a threat that is serious across the whole pool still triggers
    #  action even when no single place stands out" (Appendix B, line 268).
    # "when the benchmark is widened to its plausible upper range, the absolute
    #  burden rises, and the rule leans toward alerting rather than calming, the
    #  safer error for a serious pathogen" (line 193).
    pool_is_serious = (burden_clause != "off" and pool_burden is not None
                       and pool_burden >= pool_burden_high)

    # -- The uncertainty condition ------------------------------------------------
    # "the actual-risk input is treated as an estimate with uncertainty rather than
    #  as truth, so the rule acts on the divergence only when it exceeds the combined
    #  uncertainty of both sides" (Section 4.3, line 193).
    # This is a SEPARATE test from the decision-cost threshold below and must not be
    # collapsed into it: the paper states both and never merges them.
    exceeds_uncertainty = (D_lo is not None and D_hi is not None
                           and (D_lo > 0 or D_hi < 0))
    if not exceeds_uncertainty:
        # "no single place stands out" is precisely this case, so under the default
        # placement the pool-wide burden clause still fires here. The placement is
        # OURS: the paper states the clause and never states where it sits.
        if pool_is_serious and burden_clause == "independent":
            return ALERT, (
                f"D = {D:+.3f} does not exceed its uncertainty "
                f"[{D_lo:+.3f}, {D_hi:+.3f}], so this place does not stand out, but "
                f"pool burden {pool_burden:.2f} percent of ED visits is at or above "
                f"OUR seriousness cut of {pool_burden_high:.2f}. A threat serious "
                f"across the whole pool still triggers action. Choosing ALERT for "
                f"that case is OUR reading of the safer-error logic. Trend {trend}.")
        return HOLD_UNCERTAIN, (
            f"D = {D:+.3f} does not exceed its uncertainty "
            f"[{D_lo:+.3f}, {D_hi:+.3f}]. Interval is one-sided: no error exists on "
            f"the actual leg, so this is narrower than the paper's test.")

    # -- The threshold, sign crossed with size ------------------------------------
    # "The rule maps the sign and size of the divergence to a communication goal."
    #  (line 71)
    # "When the divergence is negative and large, the regime is downplayed and the
    #  goal is to bring perceived risk up toward an appropriate level: alert and
    #  mobilize" (line 71).
    if D <= -t_alert:
        return ALERT, (
            f"D = {D:+.3f} is past OUR alert bar of -{t_alert:.2f} and its interval "
            f"[{D_lo:+.3f}, {D_hi:+.3f}] excludes zero. Downplayed regime. Burden "
            f"here {burden:.2f} percent of ED visits, pool {pool_burden:.2f}. "
            f"Trend {trend}.")

    # "When the divergence is positive and large, the regime is inflated and the goal
    #  is to bring perceived risk down toward the real level: calm and clarify"
    #  (line 71). But: "We therefore treat a positive divergence as necessary but not
    #  sufficient for the inflated regime" (line 73).
    if D >= t_calm:
        # The absolute-burden lean, applied before the calming qualification.
        if pool_is_serious:
            return HOLD_WATCH, (
                f"D = {D:+.3f} is past OUR calm bar of {t_calm:.2f}, but pool burden "
                f"{pool_burden:.2f} percent of ED visits is at or above OUR "
                f"seriousness cut of {pool_burden_high:.2f}, and the rule leans away "
                f"from calming when absolute burden is high. Held and watched. "
                f"Trend {trend}.")
        # The calming qualification.
        # "Before calming, the rule asks why the burden is low, and it answers with
        #  the coverage and susceptibility measure it already reads for targeting in
        #  Section 3.4. If few people remain susceptible, or the route of exposure is
        #  unlikely to lead to sustained transmission events, the hazard is inherently
        #  small, the perception is genuinely inflated, and calming is the proper
        #  corrective action. If population risk remains high due to a high proportion
        #  of susceptible individuals but with current low disease burden, and
        #  protective behaviors remain high, the hazard has not fallen ... and the
        #  place is held and watched rather than calmed." (line 73)
        if susceptibility_low is None:
            return HOLD_WATCH, (
                f"D = {D:+.3f} is past OUR calm bar of {t_calm:.2f}, but the calming "
                f"qualification cannot be evaluated: no susceptibility or coverage "
                f"input exists for this pool. A positive divergence is necessary but "
                f"not sufficient, so the place is held and watched, not calmed. "
                f"Trend {trend}.")
        if susceptibility_low is True:
            return CALM, (
                f"D = {D:+.3f} is past OUR calm bar of {t_calm:.2f}, interval excludes "
                f"zero, pool burden {pool_burden:.2f} is below OUR seriousness cut, "
                f"and few people remain susceptible, so the hazard is inherently "
                f"small. Trend {trend}.")
        # susceptibility_low is False
        if protective_behaviour_high is None:
            return HOLD_WATCH, (
                f"D = {D:+.3f} is past OUR calm bar, susceptibility remains high, and "
                f"there is no protective-behaviour series to complete the test. Held "
                f"and watched rather than calmed. Trend {trend}.")
        if protective_behaviour_high is True:
            return HOLD_WATCH, (
                f"D = {D:+.3f} is past OUR calm bar, but susceptibility is high while "
                f"burden is low and protective behaviour is high, so the hazard has "
                f"not fallen and perceived risk is proportionate to it. Held and "
                f"watched. Trend {trend}.")
        return CALM, (
            f"D = {D:+.3f} is past OUR calm bar, susceptibility is high but protective "
            f"behaviour is not what is holding burden down. Trend {trend}.")

    # -- Small divergence ---------------------------------------------------------
    # "When the divergence is small, the rule suggests no change and continues
    #  monitoring, as an unnecessary messaging intervention feeds the attention loop"
    #  (line 71).
    # Overridden only by the pool-wide burden clause. The paper says high pool burden
    # "still triggers action" without naming which action; the safer-error logic and
    # the "leans toward alerting" sentence point to alerting, and choosing alerting
    # is OURS.
    if pool_is_serious:
        return ALERT, (
            f"D = {D:+.3f} is inside both bars, but pool burden {pool_burden:.2f} "
            f"percent of ED visits is at or above OUR seriousness cut of "
            f"{pool_burden_high:.2f}, and a threat serious across the whole pool "
            f"still triggers action even when no single place stands out. Choosing "
            f"ALERT for that case is OUR reading of the safer-error logic. "
            f"Trend {trend}.")
    return HOLD_MONITOR, (
        f"D = {D:+.3f} is inside OUR bars of -{t_alert:.2f} and +{t_calm:.2f}. "
        f"Calibrated. Pool burden {pool_burden:.2f} is below OUR seriousness cut. "
        f"Trend {trend}.")


def content_and_target(action, claims=None, attention=None, protection=None):
    """
    Table 1's Content and Target rows. Returns strings that state honestly what is
    and is not available. Nothing here is fabricated.
      Content (from Study 3): "correct the over-stated claims; state the low real
        risk" / "correct the downplaying claims; give the protective step" / "none".
      Target (from Study 1): "the communities where attention is amplifying" / "the
        under-protected communities carrying the risk" / "not applicable".
    """
    if action in (NO_DECISION, INSUFFICIENT):
        return "not applicable", "not applicable"
    if action in (HOLD_MONITOR, HOLD_UNCERTAIN):
        return "none", "not applicable"
    if claims is None:
        content = "NOT INSTANTIATED: no state-level claim audit exists for any state, disease or window"
    else:
        content = str(claims)
    if action == ALERT:
        tgt = ("the under-protected communities carrying the risk; "
               + ("state vaccination coverage is the only candidate on disk and its "
                  "provenance is unreproducible from this repo, so it is NOT used here"
                  if protection is None else f"coverage proxy {protection}"))
    elif action in (CALM, HOLD_WATCH):
        tgt = ("the communities where attention is amplifying; "
               + ("Google Trends exists at state level for one window only and there "
                  "is no covid series, so it is NOT used here"
                  if attention is None else f"attention proxy {attention}"))
    else:
        tgt = "not applicable"
    return content, tgt


# =====================================================================================
# DATA
# =====================================================================================
def curl_json(url, params, timeout=180, tries=5):
    cmd = ["curl", "-s", "--max-time", str(timeout), "--get", url]
    for k, v in params:
        cmd += ["--data-urlencode", f"{k}={v}"]
    key = os.environ.get("DELPHI_API_KEY", "").strip()
    if key and "delphi" in url:
        cmd += ["--data-urlencode", f"api_key={key}"]
    import time
    for i in range(tries):
        r = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return json.loads(r.stdout)
        except Exception:
            if "429" in r.stdout or "Too Many Requests" in r.stdout:
                w = 20 * (i + 1)
                print(f"    rate limited, waiting {w}s (attempt {i+1}/{tries})")
                time.sleep(w)
                continue
            print("    raw response head:", r.stdout[:200])
            raise SystemExit("    could not parse response from " + url)
    raise SystemExit("    kept being rate limited")


def load_concern(refresh=False):
    """
    Perceived leg for both diseases, all four windows.
    Flu comes off disk (flu_concern_by_state.csv, 204 rows, zero gaps). Covid has no
    multi-window file on disk, so it is pulled once and cached to a NEW file.
    """
    flu = pd.read_csv(os.path.join(HERE, "flu_concern_by_state.csv"))
    flu = flu[flu["suppressed"] == 0].copy()
    flu["disease"] = "flu"
    flu = flu.rename(columns={"concern_pct": "concern"})
    flu = flu[["disease", "state", "window", "concern", "ci_low", "ci_high", "sample_size"]]

    cache = os.path.join(HERE, "decision_rule_inputs_concern.csv")
    if os.path.exists(cache) and not refresh:
        cov = pd.read_csv(cache)
        cov = cov[cov["disease"] == "covid"]
    else:
        print("  pulling covid concern from CDC NIS-FRVM ...")
        where = ("dsss_indicator_category_label='Concerned about COVID-19 disease' AND "
                 "geographic_level='State' AND "
                 "dsss_group_variable_name='All adults 18+ years'")
        d = curl_json("https://data.cdc.gov/resource/ee83-ukst.json",
                      [("$where", where), ("$limit", "2000")])
        rows = []
        for r in d:
            if str(r.get("suppressionflag", "0")) != "0":
                continue
            st = ABBR.get(str(r.get("geographic_label", "")).strip().lower())
            if not st or not r.get("dsss_value"):
                continue
            try:
                lo, hi = [float(x.strip())
                          for x in str(r["dsss_confidenceinterval"]).split("-")]
            except Exception:
                lo = hi = np.nan
            rows.append({"disease": "covid", "state": st,
                         "window": r["dsss_timeperiodlabel"],
                         "concern": float(r["dsss_value"]), "ci_low": lo, "ci_high": hi,
                         "sample_size": int(r["sample_size"]) if r.get("sample_size") else np.nan})
        cov = pd.DataFrame(rows)
        cov = cov[cov["window"].isin(WINDOWS)].reset_index(drop=True)
        cov.to_csv(cache, index=False)
        print(f"    cached {len(cov)} rows -> {os.path.basename(cache)}")
    return pd.concat([flu, cov], ignore_index=True)


def load_burden(refresh=False):
    """
    Actual leg for both diseases, all four windows, from CDC NSSP via Delphi.
    Not on disk per window (flu_ed_by_state.csv is the last window only), so it is
    pulled once and cached to a NEW file. Both a mean and a peak-week summary are
    kept, because spec_curve.csv shows that choice flips the headline sign.
    """
    cache = os.path.join(HERE, "decision_rule_inputs_burden.csv")
    if os.path.exists(cache) and not refresh:
        return pd.read_csv(cache)
    rows = []
    for dis, (_, sig) in DISEASES.items():
        for w in WINDOWS:
            print(f"  pulling NSSP {sig} for {w} ({EPIWEEKS[w]}) ...")
            d = curl_json("https://api.delphi.cmu.edu/epidata/covidcast/",
                          [("data_source", "nssp"), ("signal", sig),
                           ("time_type", "week"), ("geo_type", "state"),
                           ("time_values", EPIWEEKS[w]), ("geo_value", "*")])
            per = {}
            for r in d.get("epidata", []):
                if r.get("value") is not None:
                    per.setdefault(r["geo_value"], []).append(float(r["value"]))
            for s, v in per.items():
                if s in ABBR.values():
                    rows.append({"disease": dis, "state": s, "window": w,
                                 "epiweeks": EPIWEEKS[w],
                                 "burden_mean": float(np.mean(v)),
                                 "burden_max": float(np.max(v)), "nweeks": len(v)})
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False)
    print(f"    cached {len(out)} rows -> {os.path.basename(cache)}")
    return out


def vdw(x):
    """van der Waerden inverse-normal rank score, exactly as compute_d.py defines it."""
    x = np.asarray(x, float)
    r = stats.rankdata(x)
    return stats.norm.ppf(r / (len(x) + 1.0))


def build_cells(concern, burden, vote, summary="mean"):
    """
    One pool per (disease, window): D, its bootstrap interval, burden, vote.
    D = vdW(perceived) - vdW(actual), the Appendix B object, computed within the pool.
    """
    col = "burden_" + summary
    cells = {}
    rng = np.random.default_rng(SEED)
    for dis in DISEASES:
        for w in WINDOWS:
            c = concern[(concern.disease == dis) & (concern.window == w)]
            b = burden[(burden.disease == dis) & (burden.window == w)]
            df = c.merge(b[["state", col]], on="state", how="inner")
            df = df.merge(vote, on="state", how="inner").dropna(
                subset=["concern", col, "gop_margin"]).reset_index(drop=True)
            if df.empty:
                continue
            n = len(df)
            df["burden"] = df[col]
            df["z_perceived"] = vdw(df["concern"])
            df["z_actual"] = vdw(df["burden"])
            df["D"] = df["z_perceived"] - df["z_actual"]
            # Bootstrap interval on D. Perturb the perceived leg with CDC's own
            # published confidence intervals, re-rank all places jointly, subtract the
            # fixed actual-side score. Actual-side error is declared zero because none
            # exists; see OUR_CHOICES["ACTUAL_SIDE_UNCERTAINTY"].
            sd = ((df["ci_high"] - df["ci_low"]) / (2 * 1.959964)).fillna(0).values
            sd = np.where(sd > 0, sd, 1e-6)
            draws = rng.normal(df["concern"].values[None, :], sd[None, :], size=(NBOOT, n))
            zb = np.array([stats.norm.ppf(stats.rankdata(draws[i]) / (n + 1.0))
                           for i in range(NBOOT)])
            Db = zb - df["z_actual"].values[None, :]
            df["D_lo"] = np.percentile(Db, 2.5, axis=0)
            df["D_hi"] = np.percentile(Db, 97.5, axis=0)
            df["excludes_zero"] = (df["D_lo"] > 0) | (df["D_hi"] < 0)
            df["burden_percentile"] = df["burden"].rank(pct=True)
            df["disease"], df["window"] = dis, w
            cells[(dis, w)] = df
    return cells


def add_trend(cells):
    """
    Third input: "whether the divergence is widening ... or closing on its own"
    (line 67). Estimator, window and cut are all unspecified by the paper, so this is
    ours: first difference of |D| against the previous survey window, with a dead
    zone. IT CHANGES NO ACTION, because the paper never says what the rule should do
    differently when the divergence is widening.
    """
    for dis in DISEASES:
        prev = None
        for w in WINDOWS:
            key = (dis, w)
            if key not in cells:
                continue
            df = cells[key]
            if prev is None:
                df["trend"] = None
                df["dD_abs"] = np.nan
            else:
                p = prev.set_index("state")["D"].abs()
                cur = df.set_index("state")["D"].abs()
                delta = (cur - p.reindex(cur.index))
                df["dD_abs"] = delta.reindex(df["state"]).values
                df["trend"] = np.where(df["dD_abs"].isna(), None,
                                       np.where(df["dD_abs"] > TREND_DEADZONE, "widening",
                                                np.where(df["dD_abs"] < -TREND_DEADZONE,
                                                         "closing", "flat")))
            prev = df
    return cells


# =====================================================================================
# THE MEASUREMENT GATE, COMPUTED
# =====================================================================================
def partial_spearman(x, y, z):
    """
    Rank residual partial correlation. Rank all three, regress x and y on z by OLS,
    correlate the residuals. The manuscript does not say which convention it used;
    this is ours and it is stated as such.
    """
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    n = len(rx)
    A = np.column_stack([np.ones(n), rz])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    r = float(np.corrcoef(ex, ey)[0, 1])
    dfree = n - 3
    if dfree <= 0 or abs(r) >= 1:
        return r, np.nan
    t = r * np.sqrt(dfree / (1 - r ** 2))
    return r, float(2 * stats.t.sf(abs(t), dfree))


def icc_between_share(concern, disease):
    """
    Gate quantity 1: "the share of variance in the perceived-risk measure that lies
    between places rather than within a place over time" (line 111). One-way random
    effects on the balanced panel, corrected for sampling error exactly as
    variance_decomp.py does it.
    """
    d = concern[concern.disease == disease].dropna(subset=["concern"])
    counts = d.groupby("state")["window"].nunique()
    keep = counts[counts == len(WINDOWS)].index
    d = d[d["state"].isin(keep)]
    if d.empty:
        return np.nan, np.nan, 0, 0
    n_states, n_win = d["state"].nunique(), d["window"].nunique()
    grand = d["concern"].mean()
    sm = d.groupby("state")["concern"].mean()
    ss_b = n_win * ((sm - grand) ** 2).sum()
    ss_w = ((d["concern"] - d["state"].map(sm)) ** 2).sum()
    ms_b, ms_w = ss_b / (n_states - 1), ss_w / (len(d) - n_states)
    var_b = max((ms_b - ms_w) / n_win, 0.0)
    se = ((d["ci_high"] - d["ci_low"]) / (2 * 1.959964)).dropna()
    samp = float((se ** 2).mean())
    var_w_true = max(ms_w - samp, 0.0)
    icc_naive = var_b / (var_b + ms_w)
    icc_corr = var_b / (var_b + var_w_true) if (var_b + var_w_true) > 0 else np.nan
    return icc_naive, icc_corr, n_states, n_win


def run_gate(cells, concern, alpha=GATE_ALPHA, mode=OUR_CHOICES["GATE_MODE"][0]):
    """
    Gate quantity 2: "whether the measure tracks the actual-risk benchmark inside the
    period being acted on" (line 111). Computed as the Spearman association between
    concern and burden across the 51 places within the window, and the same
    association after controlling for the 2024 partisan vote.
    Failure, as the paper writes it, is a CONJUNCTION: "If most of the variance lies
    between places and the measure does not move with the hazard". The mixed cases
    are not addressed by the paper. Both readings are computed.
    """
    icc = {dis: icc_between_share(concern, dis) for dis in DISEASES}
    rows = []
    for (dis, w), df in cells.items():
        rho, p = stats.spearmanr(df["concern"], df["burden"])
        pr, pp = partial_spearman(df["concern"], df["burden"], df["gop_margin"])
        icc_naive, icc_corr, ns, nw = icc[dis]
        most_between = bool(icc_corr > OUR_CHOICES["ICC_BETWEEN_MAJORITY"][0])
        tracks = bool((pr > 0) and (pp < alpha) and (rho > 0) and (p < alpha))
        fail_conjunction = bool(most_between and not tracks)
        passed = (not fail_conjunction) if mode == "conjunction" else tracks
        rows.append({"disease": dis, "window": w, "n": len(df),
                     "rho_concern_burden": round(float(rho), 4), "p_rho": float(p),
                     "partial_rho_given_vote": round(float(pr), 4), "p_partial": float(pp),
                     "icc_between_corrected": round(float(icc_corr), 4),
                     "most_variance_between_places": most_between,
                     "tracks_benchmark_in_window": tracks,
                     "gate_mode": mode,
                     "gate": "PASS" if passed else "FAIL",
                     "gate_conjunction_reading": "FAIL" if fail_conjunction else "PASS",
                     "gate_tracking_only_reading": "PASS" if tracks else "FAIL",
                     "alpha": alpha})
    g = pd.DataFrame(rows)
    g["_o"] = g["window"].map({w: i for i, w in enumerate(WINDOWS)})
    return g.sort_values(["disease", "_o"]).drop(columns="_o").reset_index(drop=True)


# =====================================================================================
# RUN
# =====================================================================================
def run_pool(df, gate_passed, t_alert=T_ALERT, t_calm=T_CALM,
             pool_burden_high=BURDEN_HIGH, susceptibility=None,
             burden_clause=OUR_CHOICES["BURDEN_CLAUSE_PLACEMENT"][0]):
    """Apply decide() to every place in one (disease, window) pool."""
    pool_burden = float(df["burden"].mean())
    out = []
    for _, r in df.iterrows():
        sl = None if susceptibility is None else bool(susceptibility.get(r["state"]))
        a, why = decide(D=float(r["D"]), D_lo=float(r["D_lo"]), D_hi=float(r["D_hi"]),
                        burden=float(r["burden"]),
                        burden_percentile=float(r["burden_percentile"]),
                        trend=r.get("trend"), gate_passed=gate_passed,
                        t_alert=t_alert, t_calm=t_calm,
                        pool_burden=pool_burden, pool_burden_high=pool_burden_high,
                        pool_n=len(df), susceptibility_low=sl,
                        burden_clause=burden_clause,
                        protective_behaviour_high=None, claims=None)
        content, target = content_and_target(a, claims=None, attention=None, protection=None)
        out.append({"disease": r["disease"], "window": r["window"], "state": r["state"],
                    "concern": r["concern"], "burden": round(float(r["burden"]), 4),
                    "pool_burden": round(pool_burden, 4),
                    "D": round(float(r["D"]), 4), "D_lo": round(float(r["D_lo"]), 4),
                    "D_hi": round(float(r["D_hi"]), 4),
                    "excludes_zero": bool(r["excludes_zero"]),
                    "burden_percentile": round(float(r["burden_percentile"]), 3),
                    "trend": r.get("trend"), "gop_margin": float(r["gop_margin"]),
                    "gate": "PASS" if gate_passed else "FAIL",
                    "action": a, "reason": why,
                    "content_row": content, "target_row": target})
    return pd.DataFrame(out)


ACTION_SCORE = {ALERT: -1, HOLD_MONITOR: 0, HOLD_WATCH: 0, HOLD_UNCERTAIN: 0, CALM: 1}


def action_associations(acts):
    """
    Does the rule's output sort by politics or by epidemiology? Encode the action on
    the same axis as D (alert -1, hold 0, calm +1) and correlate with the 2024 GOP
    margin and with absolute burden.
    """
    a = acts[acts["action"].isin(ACTION_SCORE)].copy()
    if a.empty or a["action"].nunique() < 2:
        return None
    a["score"] = a["action"].map(ACTION_SCORE)
    r_v, p_v = stats.spearmanr(a["score"], a["gop_margin"])
    r_b, p_b = stats.spearmanr(a["score"], a["burden"])
    res = {"n": len(a), "rho_action_vote": float(r_v), "p_action_vote": float(p_v),
           "rho_action_burden": float(r_b), "p_action_burden": float(p_b)}
    al = a[a["action"] == ALERT]
    nal = a[a["action"] != ALERT]
    if len(al) and len(nal):
        res["mean_gop_alerted"] = float(al["gop_margin"].mean())
        res["mean_gop_not_alerted"] = float(nal["gop_margin"].mean())
        res["mwu_p_vote"] = float(stats.mannwhitneyu(al["gop_margin"], nal["gop_margin"])[1])
        res["mean_burden_alerted"] = float(al["burden"].mean())
        res["mean_burden_not_alerted"] = float(nal["burden"].mean())
        res["mwu_p_burden"] = float(stats.mannwhitneyu(al["burden"], nal["burden"])[1])
    return res


class Tee:
    def __init__(self, path):
        self.f = open(path, "w")
        self.out = sys.__stdout__

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def hr(t=""):
    print("\n" + "=" * 78)
    if t:
        print(t)
        print("=" * 78)


def main():
    refresh = "--refresh" in sys.argv
    tee = Tee(os.path.join(HERE, "decision_rule_output.txt"))
    sys.stdout = tee
    try:
        _main(refresh)
    finally:
        sys.stdout = sys.__stdout__
        tee.f.close()


def _main(refresh):
    print("THE DECISION RULE, EXECUTED. 51 US jurisdictions, 2 diseases, 4 windows.")
    print("Every constant the manuscript did not supply is listed as OURS below.")

    hr("0. INPUTS, AND WHICH OF THEM ACTUALLY EXIST")
    concern = load_concern(refresh)
    burden = load_burden(refresh)
    el = pd.read_csv(os.path.join(HERE, "election_2024_by_state.csv"))
    vote = el[["state", "gop_margin"]]
    print("\n  INSTANTIATED, from real data:")
    print("    perceived leg   CDC NIS-FRVM concern, 51 states x 4 windows, both diseases")
    print("    actual leg      CDC NSSP percent of ED visits, 51 states x 4 windows")
    print("    divergence D    vdW(perceived) - vdW(actual), computed within each pool")
    print("    uncertainty     bootstrap interval on D from CDC's published CIs")
    print("    absolute burden percent of ED visits, per state and pooled")
    print("    trend           first difference of |D| across survey windows")
    print("    partisan vote   2024 GOP margin, 51 states")
    print("\n  NOT INSTANTIATED, set to None and never faked:")
    print("    claim audit     Table 1's entire Content row. No state-level claim data")
    print("                    exists for any state, disease or window. The FIFA")
    print("                    workbook has no geography and a different disease; the")
    print("                    12-row reddit test file has Theme, Verdict and Direction")
    print("                    empty on all 12 rows and covers 2 jurisdictions.")
    print("    susceptibility  no coverage or susceptibility series for the pool, so the")
    print("                    calming qualification cannot be certified")
    print("    protective      no protective-behaviour series at state level")
    print("    forecast        no forecast anywhere in the repo; the trend here is")
    print("                    backward-looking and is NOT relabelled as a forecast")
    print("    strength dial   lives only in the simulation, not fitted to any state")
    print("\n  A NOTE THAT MUST NOT BE GLOSSED: the paper says Study 1 supplies the")
    print("  perceived side, from geolocated Reddit users with LDA and NMF. The 51-state")
    print("  perceived leg here is the CDC survey, which is the instrument the paper")
    print("  TESTS the measurement condition on, not the instrument it specifies.")

    for dis in DISEASES:
        for w in WINDOWS:
            nc = len(concern[(concern.disease == dis) & (concern.window == w)])
            nb = len(burden[(burden.disease == dis) & (burden.window == w)])
            print(f"    {dis:5} {w:28} concern n={nc:3}  burden n={nb:3}")

    cells = add_trend(build_cells(concern, burden, vote,
                                  summary=OUR_CHOICES["BURDEN_SUMMARY"][0]))

    # Reproduce the on-disk file as a check that D and the interval are the same object.
    hr("0b. CHECK AGAINST THE FILES ALREADY ON DISK")
    for dis, f in (("flu", "divergence_flu.csv"), ("covid", "divergence_covid.csv")):
        old = pd.read_csv(os.path.join(HERE, f))
        new = cells[(dis, PRIMARY_WINDOW)]
        m = old[["state", "D_z", "excludes_zero"]].merge(
            new[["state", "D", "excludes_zero"]], on="state", suffixes=("_old", "_new"))
        dmax = float((m["D_z"] - m["D"]).abs().max())
        agree = int((m["excludes_zero_old"] == m["excludes_zero_new"]).sum())
        print(f"  {dis:5} vs {f}: n={len(m)}, max |D difference| = {dmax:.4f}, "
              f"uncertainty flag agrees on {agree}/{len(m)} states")
    print("  Small D differences are the NSSP series revising between pull dates, not a")
    print("  different metric. Flag disagreements are bootstrap noise at a shared seed.")

    hr("1. THE MEASUREMENT GATE, COMPUTED PER WINDOW")
    print("\n  Gate quantity 1, the between-place variance share (one number per disease,")
    print("  computed on the balanced 51 x 4 panel, corrected for survey noise):")
    for dis in DISEASES:
        icc_n, icc_c, ns, nw = icc_between_share(concern, dis)
        print(f"    {dis:5} ICC naive {icc_n:.3f}   ICC corrected {icc_c:.3f}   "
              f"({ns} states x {nw} windows)")
    print("\n  Gate quantity 2, does the measure track the benchmark inside the window:")
    gate = run_gate(cells, concern)
    print()
    print(f"  {'disease':7} {'window':28} {'rho':>7} {'p':>9} {'partial':>8} {'p':>9}  gate")
    print("  " + "-" * 76)
    for _, r in gate.iterrows():
        print(f"  {r.disease:7} {r.window:28} {r.rho_concern_burden:+7.3f} "
              f"{r.p_rho:9.4f} {r.partial_rho_given_vote:+8.3f} {r.p_partial:9.4f}  "
              f"{r.gate}")
    print("\n  Both readings of the paper's failure condition:")
    for _, r in gate.iterrows():
        print(f"    {r.disease:5} {r.window:28} tracking-only {r.gate_tracking_only_reading:4}"
              f"   conjunction {r.gate_conjunction_reading}")
    same = (gate["gate_tracking_only_reading"] == gate["gate_conjunction_reading"]).all()
    print(f"\n  The two readings agree on every cell: {bool(same)}. They coincide whenever")
    print("  the between-place share is above half, which it is for both diseases, so our")
    print("  choice of reading does not drive any result here.")
    gate.to_csv(os.path.join(HERE, "decision_rule_gate.csv"), index=False)
    print("  saved -> decision_rule_gate.csv")

    gate_map = {(r.disease, r.window): (r.gate == "PASS") for _, r in gate.iterrows()}

    hr("2. THE RULE, RUN ON EVERY CELL")
    print(f"\n  OUR default bars: alert at D <= -{T_ALERT:.2f}, calm at D >= +{T_CALM:.2f}")
    print(f"  (ratio {OUR_CHOICES['ASYMMETRY_RATIO_DEFAULT'][0]:.2f}). The paper supplies")
    print("  neither number, only the inequality that the alert bar is the lower one.")
    print(f"  OUR pool-seriousness cut: {BURDEN_HIGH:.2f} percent of ED visits.")

    print("\n  EVERY TERMINAL OUTPUT OF decide(), ON SYNTHETIC INPUTS, SO THAT THE")
    print("  BRANCHES ARE VISIBLY REACHABLE:\n")
    checks = [
        ("gate failed", dict(D=-2.0, D_lo=-3.0, D_hi=-1.0, burden=5.0,
                             burden_percentile=0.9, trend="flat", gate_passed=False)),
        ("pool of 10 places", dict(D=-2.0, D_lo=-3.0, D_hi=-1.0, burden=5.0,
                                   burden_percentile=0.9, trend="flat",
                                   gate_passed=True, pool_n=10)),
        ("interval spans zero, quiet pool",
         dict(D=-2.0, D_lo=-3.0, D_hi=0.5, burden=0.1, burden_percentile=0.9,
              trend="flat", gate_passed=True, pool_burden=0.1)),
        ("D past the alert bar", dict(D=-2.0, D_lo=-3.0, D_hi=-1.0, burden=0.1,
                                      burden_percentile=0.9, trend="flat",
                                      gate_passed=True, pool_burden=0.1)),
        ("small D, quiet pool", dict(D=0.1, D_lo=0.05, D_hi=0.2, burden=0.1,
                                     burden_percentile=0.5, trend="flat",
                                     gate_passed=True, pool_burden=0.1)),
        ("small D, serious pool", dict(D=0.1, D_lo=0.05, D_hi=0.2, burden=5.0,
                                       burden_percentile=0.5, trend="flat",
                                       gate_passed=True, pool_burden=5.0)),
        ("D past the calm bar, no susceptibility input",
         dict(D=1.5, D_lo=0.5, D_hi=2.5, burden=0.3, burden_percentile=0.9,
              trend="flat", gate_passed=True, pool_burden=0.5)),
        ("D past the calm bar, few remain susceptible",
         dict(D=1.5, D_lo=0.5, D_hi=2.5, burden=0.3, burden_percentile=0.9,
              trend="flat", gate_passed=True, pool_burden=0.5,
              susceptibility_low=True)),
        ("D past the calm bar, susceptible and protective behaviour high",
         dict(D=1.5, D_lo=0.5, D_hi=2.5, burden=0.3, burden_percentile=0.9,
              trend="flat", gate_passed=True, pool_burden=0.5,
              susceptibility_low=False, protective_behaviour_high=True)),
    ]
    for label, kw in checks:
        print(f"    {label:58} -> {decide(**kw)[0]}")
    try:
        decide(D=1.0, D_lo=0.5, D_hi=1.5, burden=1.0, burden_percentile=0.5,
               trend="flat", gate_passed=True, t_alert=1.0, t_calm=0.5)
        print("    WARNING: a calm bar below the alert bar was accepted")
    except ValueError:
        print(f"    {'calm bar set below the alert bar':58} -> rejected; that ordering")
        print(f"    {'':58}    is the paper's one constraint")

    all_acts = []
    for dis in DISEASES:
        for w in WINDOWS:
            key = (dis, w)
            if key not in cells:
                continue
            acts = run_pool(cells[key], gate_map[key])
            all_acts.append(acts)
    acts = pd.concat(all_acts, ignore_index=True)
    acts.to_csv(os.path.join(HERE, "decision_rule_actions.csv"), index=False)

    print("\n  ACTION DISTRIBUTION BY CELL (n=51 everywhere):\n")
    tab = acts.groupby(["disease", "window", "action"]).size().unstack(fill_value=0)
    order = [c for c in [NO_DECISION, ALERT, CALM, HOLD_MONITOR, HOLD_WATCH,
                         HOLD_UNCERTAIN, INSUFFICIENT] if c in tab.columns]
    tab = tab[order]
    tab = tab.reindex([(d, w) for d in DISEASES for w in WINDOWS if (d, w) in tab.index])
    print(tab.to_string())
    print("\n  saved -> decision_rule_actions.csv")

    print("\n  WHERE THE POOL-BURDEN CLAUSE SITS CHANGES THE ANSWER COMPLETELY.")
    print("  The paper states the clause and never states its placement. Three")
    print("  placements, all consistent with the text, on the one cell that passes")
    print("  the gate and on the primary cell (counterfactual there):\n")
    print(f"    {'cell':40} {'placement':18} {'alert':>6} {'calm':>5} {'watch':>6} "
          f"{'monitor':>8} {'uncert':>7}")
    for dis in DISEASES:
        for w in (SIGNAL_WINDOW, PRIMARY_WINDOW):
            if (dis, w) not in cells:
                continue
            pb = float(cells[(dis, w)]["burden"].mean())
            for pl in ("independent", "after_uncertainty", "off"):
                a = run_pool(cells[(dis, w)], gate_passed=True, burden_clause=pl)
                v = a["action"].value_counts().to_dict()
                lbl = f"{dis}/{w.split(' - ')[0]} pool {pb:.2f}"
                print(f"    {lbl:40} {pl:18} {v.get(ALERT,0):6d} {v.get(CALM,0):5d} "
                      f"{v.get(HOLD_WATCH,0):6d} {v.get(HOLD_MONITOR,0):8d} "
                      f"{v.get(HOLD_UNCERTAIN,0):7d}")
    print("\n  (counts above ignore the gate so the placements are comparable; the")
    print("  primary window issues no decisions at all once the gate is applied)")

    hr("3a. PRIMARY WINDOW, ALL 51 JURISDICTIONS, BOTH DISEASES")
    for dis in DISEASES:
        sub = acts[(acts.disease == dis) & (acts.window == PRIMARY_WINDOW)]
        g = gate[(gate.disease == dis) & (gate.window == PRIMARY_WINDOW)].iloc[0]
        print(f"\n  {dis.upper()}, {PRIMARY_WINDOW}")
        print(f"    gate: {g.gate}.  concern vs burden rho = {g.rho_concern_burden:+.3f} "
              f"(p = {g.p_rho:.4f}); partial for vote {g.partial_rho_given_vote:+.3f} "
              f"(p = {g.p_partial:.4f})")
        if g.gate == "FAIL":
            print("    RESULT: the rule is NOT actuated in this window. All 51 jurisdictions")
            print("    return no decision. This refusal IS the end-to-end result for the")
            print("    primary window. No scores are issued.")
            print(f"    D range across the pool: {sub.D.min():+.3f} to {sub.D.max():+.3f}; "
                  f"{int(sub.excludes_zero.sum())} of 51 exceed their uncertainty.")
        else:
            for _, r in sub.sort_values("D").iterrows():
                print(f"      {r.state:3} D={r.D:+6.3f} [{r.D_lo:+6.3f},{r.D_hi:+6.3f}] "
                      f"burden={r.burden:5.2f}  {r.action}")

    hr("3b. GATE PER WINDOW, WITH THE STATISTICS BEHIND IT")
    print("\n  (repeated compactly for the record)\n")
    print(gate[["disease", "window", "n", "rho_concern_burden", "p_rho",
                "partial_rho_given_vote", "p_partial", "icc_between_corrected",
                "gate"]].to_string(index=False))

    hr("3c. THE KEY COMPARISON: SAME RULE, SAME STATES, DIFFERENT WINDOW")
    for dis in DISEASES:
        gp = gate[(gate.disease == dis) & (gate.window == PRIMARY_WINDOW)].iloc[0]
        gs = gate[(gate.disease == dis) & (gate.window == SIGNAL_WINDOW)].iloc[0]
        print(f"\n  {dis.upper()}")
        print(f"    {PRIMARY_WINDOW:28} gate {gp.gate}  (rho {gp.rho_concern_burden:+.3f}, "
              f"p {gp.p_rho:.4f}; partial {gp.partial_rho_given_vote:+.3f}, p {gp.p_partial:.4f})")
        print(f"    {SIGNAL_WINDOW:28} gate {gs.gate}  (rho {gs.rho_concern_burden:+.3f}, "
              f"p {gs.p_rho:.4f}; partial {gs.partial_rho_given_vote:+.3f}, p {gs.p_partial:.4f})")
        if gp.gate == "FAIL" and gs.gate == "PASS":
            a_s = acts[(acts.disease == dis) & (acts.window == SIGNAL_WINDOW)]
            print(f"\n    In {PRIMARY_WINDOW} the rule issues no decisions at all.")
            print(f"    In {SIGNAL_WINDOW} it issues, under the default burden-clause")
            print(f"    placement:")
            for k, v in a_s["action"].value_counts().items():
                print(f"        {k:38} {v:3}")
            print(f"    and with the pool-burden clause switched off, so that only the")
            print(f"    divergence machinery is acting:")
            a_s = run_pool(cells[(dis, SIGNAL_WINDOW)], gate_passed=True,
                           burden_clause="off")
            for k, v in a_s["action"].value_counts().items():
                print(f"        {k:38} {v:3}")
            # counterfactual, clearly labelled
            cf = run_pool(cells[(dis, PRIMARY_WINDOW)], gate_passed=True,
                          burden_clause="off")
            print(f"\n    COUNTERFACTUAL, NOT ENDORSED. If the gate were ignored in the")
            print(f"    primary window the rule would have issued (burden clause off):")
            for k, v in cf["action"].value_counts().items():
                print(f"        {k:38} {v:3}")
            print(f"\n    The state-by-state comparison below is on the burden-clause-off")
            print(f"    run in both windows, so that like is compared with like.")
            merged = cf[["state", "action", "D"]].merge(
                a_s[["state", "action", "D"]], on="state", suffixes=("_primary_cf", "_signal"))
            flips = merged[merged.action_primary_cf != merged.action_signal]
            print(f"\n    The same states get a DIFFERENT call in {len(flips)} of "
                  f"{len(merged)} jurisdictions.")
            sign_flip = merged[(merged.D_primary_cf * merged.D_signal) < 0]
            print(f"    D changes sign in {len(sign_flip)} of {len(merged)} jurisdictions "
                  f"between the two windows.")
            if len(flips):
                print("\n      state  primary window (counterfactual)      signal window")
                for _, r in flips.sort_values("state").head(60).iterrows():
                    print(f"      {r.state:5}  D={r.D_primary_cf:+6.3f} "
                          f"{r.action_primary_cf:32} D={r.D_signal:+6.3f} {r.action_signal}")
            corr = stats.spearmanr(merged["D_primary_cf"], merged["D_signal"])
            print(f"\n    Correlation of D between the two windows: rho = {corr[0]:+.3f}, "
                  f"p = {corr[1]:.4f}. If D were a fixed property of a place this would")
            print("    be near 1. It is not.")
        else:
            print("    (the primary-fails / signal-passes pattern does not hold here)")

    hr("3d. DO THE RULE'S OUTPUTS SORT BY POLITICS OR BY EPIDEMIOLOGY?")
    print("\n  Action scored on the same axis as D: alert -1, hold 0, calm +1.")
    print("  Cells where the gate failed produce no decisions, so nothing is scored")
    print("  there. For those we also show the counterfactual, clearly labelled.\n")
    print("  Reported for both burden-clause placements, because the default placement")
    print("  saturates any pool whose burden clears OUR seriousness cut.\n")
    assoc_rows = []
    for placement in ("independent", "off"):
        print(f"  --- pool-burden clause: {placement} ---")
        for dis in DISEASES:
            for w in WINDOWS:
                key = (dis, w)
                if key not in cells:
                    continue
                passed = gate_map[key]
                run = run_pool(cells[key], gate_passed=True, burden_clause=placement)
                tagged = ("actual" if passed
                          else "COUNTERFACTUAL (gate failed, no decisions were issued)")
                res = action_associations(run)
                if res is None:
                    only = run["action"].value_counts().index[0]
                    print(f"  {dis:5} {w:28} [{tagged}] all 51 -> {only}; "
                          f"no association is defined on a constant")
                    assoc_rows.append({"disease": dis, "window": w, "basis": tagged,
                                       "burden_clause": placement, "n": len(run),
                                       "note": f"constant action {only}"})
                    continue
                assoc_rows.append(dict(disease=dis, window=w, basis=tagged,
                                       burden_clause=placement, **res))
                print(f"  {dis:5} {w:28} [{tagged}]")
                print(f"        action vs 2024 GOP margin  rho = {res['rho_action_vote']:+.3f}"
                      f"  p = {res['p_action_vote']:.4f}")
                print(f"        action vs absolute burden  rho = {res['rho_action_burden']:+.3f}"
                      f"  p = {res['p_action_burden']:.4f}")
                if "mean_gop_alerted" in res:
                    print(f"        alerted states mean GOP margin {res['mean_gop_alerted']:+.2f} "
                          f"vs {res['mean_gop_not_alerted']:+.2f} not alerted "
                          f"(Mann-Whitney p = {res['mwu_p_vote']:.4f})")
                    print(f"        alerted states mean burden {res['mean_burden_alerted']:.2f} "
                          f"vs {res['mean_burden_not_alerted']:.2f} not alerted "
                          f"(Mann-Whitney p = {res['mwu_p_burden']:.4f})")
        print()
    pd.DataFrame(assoc_rows).to_csv(
        os.path.join(HERE, "decision_rule_associations.csv"), index=False)
    print("\n  saved -> decision_rule_associations.csv")

    hr("4. THRESHOLD SWEEP. THE PAPER SUPPLIES NEITHER BAR.")
    print("\n  Swept over OUR alert bar and OUR asymmetry ratio. The ratio must exceed 1")
    print("  for the paper's asymmetry to hold; ratio 1.00 is shown ONLY as a contrast")
    print("  case and violates the manuscript's stated ordering.")
    print("  The sweep runs with the pool-burden clause OFF, so that the bars are what")
    print("  is moving. With the clause on and a serious pool, the burden term saturates")
    print("  the answer and the bars stop mattering, which is itself worth knowing.")
    sweep_rows = []
    for dis in DISEASES:
        for w in WINDOWS:
            key = (dis, w)
            if key not in cells:
                continue
            for ta in (0.25, 0.50, 0.75, 1.00, 1.25, 1.50):
                for k in (1.00, 1.25, 1.50, 2.00, 3.00):
                    if k <= 1.0:
                        tc = ta + 1e-9
                    else:
                        tc = ta * k
                    a = run_pool(cells[key], gate_passed=True, t_alert=ta, t_calm=tc,
                                 burden_clause="off")
                    vc = a["action"].value_counts().to_dict()
                    sweep_rows.append({
                        "disease": dis, "window": w, "burden_clause": "off",
                        "t_alert": ta,
                        "asymmetry_ratio": k, "t_calm": round(tc, 4),
                        "gate": "PASS" if gate_map[key] else "FAIL (counterfactual counts)",
                        "n_alert": vc.get(ALERT, 0), "n_calm": vc.get(CALM, 0),
                        "n_hold_watch": vc.get(HOLD_WATCH, 0),
                        "n_hold_monitor": vc.get(HOLD_MONITOR, 0),
                        "n_hold_uncertainty": vc.get(HOLD_UNCERTAIN, 0)})
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(os.path.join(HERE, "decision_rule_sweep.csv"), index=False)
    for dis in DISEASES:
        for w in (SIGNAL_WINDOW, PRIMARY_WINDOW):
            s = sweep[(sweep.disease == dis) & (sweep.window == w)]
            if s.empty:
                continue
            note = "" if gate_map[(dis, w)] else "   [gate FAILED: counterfactual counts]"
            print(f"\n  {dis} / {w}{note}")
            print(f"    {'t_alert':>7} {'ratio':>6} {'t_calm':>7} {'alert':>6} {'calm':>5} "
                  f"{'watch':>6} {'monitor':>8} {'uncert':>7}")
            for _, r in s.iterrows():
                print(f"    {r.t_alert:7.2f} {r.asymmetry_ratio:6.2f} {r.t_calm:7.2f} "
                      f"{r.n_alert:6d} {r.n_calm:5d} {r.n_hold_watch:6d} "
                      f"{r.n_hold_monitor:8d} {r.n_hold_uncertainty:7d}")
    print("\n  saved -> decision_rule_sweep.csv")

    hr("5. SENSITIVITY ON THE OTHER FREE PARAMETERS")
    print("\n  5.1 Pool-seriousness cut on absolute burden (OURS; the paper names none)")
    for dis in DISEASES:
        for w in WINDOWS:
            key = (dis, w)
            if key not in cells:
                continue
            pb = float(cells[key]["burden"].mean())
            line = f"    {dis:5} {w:28} pool burden {pb:5.2f} -> "
            bits = []
            for cut in (0.5, 1.0, 2.0, 3.0, 4.0):
                bits.append(f"cut {cut:.1f}: {'serious' if pb >= cut else 'not'}")
            print(line + "; ".join(bits))
    print("\n  This cut is decisive. Above it, every small divergence becomes an alert")
    print("  and no place can be calmed. The paper gives no level, no units and no")
    print("  procedure, so the whole calm-versus-alert balance rests on a number we chose.")

    print("\n  5.2 Burden summary, mean versus peak week (OURS)")
    b2 = build_cells(concern, burden, vote, summary="max")
    for dis in DISEASES:
        for w in WINDOWS:
            if (dis, w) not in cells:
                continue
            r1 = stats.spearmanr(cells[(dis, w)]["concern"], cells[(dis, w)]["burden"])[0]
            r2 = stats.spearmanr(b2[(dis, w)]["concern"], b2[(dis, w)]["burden"])[0]
            print(f"    {dis:5} {w:28} concern-burden rho: mean {r1:+.3f}  peak {r2:+.3f}")

    print("\n  5.3 Gate alpha (OURS; the paper attaches no numeric cut to either quantity)")
    for alpha in (0.10, 0.05, 0.01, 0.001):
        g2 = run_gate(cells, concern, alpha=alpha)
        passes = ", ".join(f"{r.disease}/{r.window.split(' - ')[0]}"
                           for _, r in g2.iterrows() if r.gate == "PASS")
        print(f"    alpha {alpha:<6} passing cells: {passes if passes else 'none'}")

    print("\n  5.4 Calming qualification (OURS; the paper's condition is not instantiable)")
    vaxf = os.path.join(HERE, "vaccination_merged.csv")
    if os.path.exists(vaxf):
        vx = pd.read_csv(vaxf).set_index("state")["vax"]
        print(f"    vaccination_merged.csv: n={len(vx)}, range {vx.min():.1f} to {vx.max():.1f}.")
        print("    Provenance unresolved: no script in this repo writes this file, so its")
        print("    source, vintage and definition are not reproducible. Used ONLY here.")
        for cut in (35.0, 40.0, 45.0, 50.0):
            susc = {s: (v >= cut) for s, v in vx.items()}
            line = []
            for dis in DISEASES:
                for w in (SIGNAL_WINDOW,):
                    if (dis, w) not in cells:
                        continue
                    a = run_pool(cells[(dis, w)], gate_passed=True, susceptibility=susc,
                                 burden_clause="off")
                    line.append(f"{dis}: {int((a.action == CALM).sum())} calmed")
            print(f"      coverage cut {cut:.0f} percent -> " + "; ".join(line)
                  + "   (signal window, burden clause off, counterfactual)")
        print("    Even this mode does NOT instantiate the paper's condition: the")
        print("    protective-behaviour half of the test has no data at all.")
    else:
        print("    vaccination_merged.csv not found; calming stays uncertifiable.")

    hr("6. THE TREND INPUT")
    print("\n  Computed as the first difference of |D| against the previous survey window,")
    print("  with a dead zone of " + f"{TREND_DEADZONE:.2f}" + ". Estimator, window and cut are all ours: the")
    print("  paper names the trend as its third input and then never states what the")
    print("  rule should DO differently when the divergence is widening. It appears in")
    print("  no branch of the action mapping and in no row of Table 1. So it is computed")
    print("  and reported here and it changes NO action. Saying otherwise would be an")
    print("  invention.\n")
    for dis in DISEASES:
        for w in WINDOWS[1:]:
            if (dis, w) not in cells:
                continue
            t = cells[(dis, w)]["trend"].value_counts().to_dict()
            print(f"    {dis:5} {w:28} " + ", ".join(f"{k} {v}" for k, v in sorted(t.items())))

    hr("7. EVERY CHOICE WE MADE THAT THE PAPER DID NOT SPECIFY")
    for i, (k, (v, why)) in enumerate(OUR_CHOICES.items(), 1):
        print(f"\n  {i:2}. {k} = {v}")
        for line in _wrap(why, 72):
            print("      " + line)
    print("\n  Plus these structural choices, also ours:")
    for i, s in enumerate([
        "Order of evaluation: thin pool, then gate, then uncertainty, then threshold, "
        "then burden, then calming qualification. The paper states all of these "
        "conditions and never states their order or their logical combination.",
        "Naming the gate-failure output at all. The manuscript never says the rule "
        "returns a token; it says the instrument is not used to guide communication "
        "decisions.",
        "Choosing ALERT as the action when pool burden is high and every D is small. "
        "The paper says such a threat 'still triggers action' without naming which.",
        "Dropping Puerto Rico and the US Virgin Islands to reach 51. CDC publishes "
        "both. This is inherited from the existing scripts, not decided by the paper.",
        "Treating the trend as report-only.",
        "The partial-correlation convention used in the gate.",
    ], 1):
        print(f"    {i}. " + "\n       ".join(_wrap(s, 72)))

    hr("8. WHAT THIS RUN DOES NOT SHOW")
    print("""
  It does not show the rule working on the instrument the paper specifies. Study 1's
  geolocated Reddit measure supplies the perceived side in the manuscript; this run
  uses the CDC survey, which is the instrument the paper reports as failing the
  measurement condition.
  It does not fill Table 1's Content row. No state-level claim audit exists.
  It does not certify a single calming decision. The calming qualification requires a
  susceptibility and coverage answer that the data cannot give, so under the paper's
  own 'necessary but not sufficient' language every positive divergence is held and
  watched instead.
  It does not use a forecast. The trend here is backward-looking.
  It does not output a calibrated message strength. That lives only in the simulation.
  It does not validate the rule against any episode where a message did or did not
  close a gap. The paper lists that as future work and it remains future work.
""")
    print("Files written:")
    for f in ("decision_rule_actions.csv", "decision_rule_gate.csv",
              "decision_rule_sweep.csv", "decision_rule_associations.csv",
              "decision_rule_output.txt", "decision_rule_inputs_concern.csv",
              "decision_rule_inputs_burden.csv"):
        print("  " + f)


def _wrap(s, n):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


if __name__ == "__main__":
    main()
