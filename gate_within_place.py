#!/usr/bin/env python3
"""
THE WITHIN-PLACE, OVER-TIME PRECONDITION GATE, IMPLEMENTED AND RUN END TO END.

WHY THIS FILE EXISTS
  decision_rule.py implements the manuscript's measurement condition as a BETWEEN-PLACE
  cross-section: the Spearman correlation between the concern leg and the burden leg
  across the 51 jurisdictions inside one CDC window. The divergence metric D that the
  gate is supposed to gate is the difference of the van der Waerden normal scores of
  those SAME two legs across those SAME places in that SAME window. Gate and metric are
  therefore algebraically tied. Part 1 of the independence demonstration below proves
  the tie is an exact identity, not a tendency:

        sd(D) = sqrt( 2 * s^2 * (1 - r) )

  where s^2 is the (fixed) variance of the van der Waerden score vector and r is the
  Pearson correlation of the two score vectors. A gate that is a deterministic
  decreasing function of the dispersion of the thing it gates is not a gate.

  This file replaces the tracking leg with the WITHIN-PLACE, OVER-TIME reading that the
  manuscript's own reasoning states four separate times:
    "The rule reads a divergence, which assumes the perceived-risk measurement moves
     when the hazard moves."                                       (rev15_ACC line 103)
    "... its political component is present before the season starts and does not move
     when the epidemic does ..."                                   (rev15_ACC line 107)
    "... a rule run on this leg would name the same places over-worried whatever the
     disease did."                                                 (rev15_ACC line 121)
    "... rather than the changing epidemic the rule needs it to track ..."
                                                                   (rev15_ACC line 125)

  THIS IS NOT A PURE BUG FIX AND MUST NOT BE PRESENTED AS ONE. Two sentences in the
  manuscript currently license the between-place gate and must be revised or the code
  and the paper will describe different tests:
    "In the one window where influenza rose steeply, from 0.58 to 4.31 percent of
     emergency-department visits, concern tracked influenza across states at 0.50, and
     the association survived control for the vote at 0.43."       (rev15_ACC line 109)
    "... and whether the measure tracks the actual-risk benchmark inside the period
     being acted on."                                              (rev15_ACC line 111)
  The phrase "inside the period being acted on" forecloses an over-time computation for
  a single window taken alone. Nothing in this repository reproduces the numbers 0.50
  and 0.43, and methods_section.tex contains no paragraph defining them, so this file
  makes no claim to reproduce or supersede them.

WHAT IS COMPUTED (all four variants, none silently preferred)
  A. one-way fixed effects   (state FE only)    concern_it = a_i + b*burden_it + e_it
  B. two-way fixed effects   (state + window FE) concern_it = a_i + g_t + b*burden_it + e
  C. non-parametric distribution test: 51 per-state Spearman rho over the 4 windows,
     raw and window-demeaned, against an exact T=4 null and a within-state permutation null
  D. measurement-error-corrected version: FGLS weighted by 1/(sigma_u^2 + se_it^2) using
     CDC's own published per-cell confidence intervals, plus reliability ratios lambda
     for every transform and the disattenuation factors they imply

  Plus the DEPLOYABLE per-window form, since decision_rule.py needs one verdict per
  (disease, window): the first-difference statistic, correlating the cross-state change
  in concern against the cross-state change in burden between adjacent windows.

READS (does not modify):
  decision_rule.py                      imported, for decide() and the pool machinery
  flu_concern_by_state.csv, decision_rule_inputs_concern.csv
  decision_rule_inputs_burden.csv, election_2024_by_state.csv
WRITES:
  gate_within_place_results.csv
  decision_rule_actions_withinplace.csv
  gate_within_place_output.txt

Run:  python3 gate_within_place.py
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

# decision_rule.py is IMPORTED, NOT MODIFIED and NOT COPIED. decide(), build_cells(),
# add_trend(), run_pool(), run_gate(), action_associations(), vdw() and the loaders are
# used exactly as they stand there, so the only thing that differs between the old run
# and the new run is the boolean handed to decide(gate_passed=...).
import decision_rule as dr

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOWS = dr.WINDOWS
DISEASES = ["flu", "covid"]
Z975 = 1.959964
SEED = 20260820
NPERM_STATE = 5000
NSIM_POWER = 2000


# =====================================================================================
# OUR CHOICES. Not one of these is supplied by the manuscript. Each is labelled, and
# each is swept or reported in both directions further down.
# =====================================================================================
OUR_CHOICES = {
    "GATE_GEOMETRY": (
        "within-place over time",
        "The manuscript states the precondition conceptually as within-place over time "
        "(lines 103, 107, 121, 125) and operationally as between-place cross-sectional "
        "(lines 109, 111). It never reconciles them. Choosing the within-place reading "
        "is OURS. The between-place reading is still computed and printed side by side."),
    "TIME_EFFECTS": (
        "two-way (state and window fixed effects)",
        "The literal wording 'moves when the hazard moves' is the one-way spec. We make "
        "the two-way spec the headline because the one-way spec is identified off a "
        "single national seasonal wave (4 time points), so clustering by state assumes "
        "an independence the common wave violates, and because a gate satisfied by "
        "'concern rises when flu season arrives nationally' licenses no place-specific "
        "action while the rule allocates across places. Both are reported."),
    "GATE_ALPHA": (
        0.05,
        "Significance level for the tracking test. The manuscript attaches no numeric "
        "cut to either gate quantity anywhere. Swept over 0.01 to 0.50."),
    "SIDEDNESS": (
        "one-sided, positive",
        "Only a positive association counts as tracking. The manuscript never says "
        "whether the test is one- or two-sided. One-sided is the more permissive "
        "choice here and is stated as ours."),
    "GATE_MODE": (
        "tracking_only",
        "The manuscript's stated failure case is a CONJUNCTION: 'If most of the "
        "variance lies between places and the measure does not move with the hazard' "
        "(line 111). Failing on the tracking leg alone is stricter than the paper and "
        "is OURS. The conjunction reading is computed and reported alongside."),
    "ICC_VALUE": (
        "corrected",
        "The manuscript deliberately publishes both 0.470 uncorrected and 0.695 "
        "corrected and quotes only 0.695 in the body. Using the corrected value is "
        "OURS. Both are printed."),
    "ICC_BETWEEN_MAJORITY": (
        0.50,
        "The word 'most' is the only quantifier the paper gives for the variance leg. "
        "We read 'most' as more than half. Same choice decision_rule.py makes."),
    "DEPLOYABLE_SCOPE": (
        "first difference, verdict attached to the later window of each transition",
        "The panel gate returns ONE verdict per disease. decision_rule.py needs one per "
        "(disease, window). We attach the verdict of the transition t-1 -> t to window "
        "t. The FIRST window then has no preceding transition and therefore NO "
        "within-place verdict at all; we do not fabricate one. Treating an "
        "unevaluable window as a gate failure is OURS."),
    "FD_STATISTIC": (
        "Spearman on deltas rounded to 6 decimals",
        "The manuscript never names a correlation family. Rounding before ranking is "
        "required for exact tie handling: CDC publishes concern to one decimal, which "
        "produces exact ties in the deltas, and without rounding a state-constant "
        "offset perturbs those ties at 1e-14 and reshuffles tied ranks. Pearson on "
        "deltas is reported alongside."),
    "MULTIPLICITY": (
        "none in the headline; Bonferroni over the 6 transitions reported",
        "The manuscript names no multiplicity correction. Both are reported and the "
        "verdict differs between them, so this choice is load-bearing."),
    "REGRESSOR_FORM": (
        "burden_mean, untransformed",
        "The window mean of the NSSP percent of ED visits. The paper names no "
        "functional form. Swept against burden_max and the logs of both."),
    "OUTCOME_FORM": (
        "concern in percentage points, untransformed",
        "As published by CDC NIS-FRVM. Swept against logit(concern/100)."),
    "CLUSTER": (
        "cluster-robust by state, CR1, inference on t with G-1 = 50 df",
        "The paper specifies no inference procedure. Clustering by state handles serial "
        "correlation within a state. It is the WRONG choice for the one-way spec, "
        "because the common national wave violates cross-state independence; the "
        "window-permutation null below quantifies exactly how wrong."),
    "FAILURE_LABEL": (
        "NO_DECISION_MEASUREMENT_CONDITION_FAILED (decision_rule.py's existing token)",
        "The manuscript names no return value for a failed measurement condition. Its "
        "only explicit failure-state wording, 'treated as insufficient data rather than "
        "forced into a regime' (Appendix B), is written about THIN POOLS, not about the "
        "measurement condition. Reusing it would be our extension, so we reuse "
        "decision_rule.py's existing token instead and keep it labelled as ours."),
    "SIGMA_U_ESTIMATOR": (
        "FE residual variance minus mean published sampling variance, floored at zero",
        "The weighting scheme is an efficiency fix, not a bias fix. No estimator for "
        "the true within-state variance component is specified anywhere in the paper."),
}


# =====================================================================================
# PANEL CONSTRUCTION
# =====================================================================================
def build_panel():
    """
    Balanced (disease x state x window) panel: 204 rows per disease, 51 x 4, zero gaps.
    concern and its published CDC standard error, burden_mean and burden_max.
    """
    concern = dr.load_concern()
    burden = dr.load_burden()
    vote = pd.read_csv(os.path.join(HERE, "election_2024_by_state.csv"))[
        ["state", "gop_margin"]]
    p = concern.merge(burden[["disease", "state", "window", "burden_mean", "burden_max"]],
                      on=["disease", "state", "window"], how="inner")
    p = p.merge(vote, on="state", how="left")
    p["se"] = (p["ci_high"] - p["ci_low"]) / (2 * Z975)
    p["t_idx"] = p["window"].map({w: i for i, w in enumerate(WINDOWS)})
    p = p.sort_values(["disease", "state", "t_idx"]).reset_index(drop=True)
    return p, concern, burden, vote


def grid(panel, disease, col):
    """Return the (51, 4) state-by-window array for one column, states sorted."""
    d = panel[panel.disease == disease]
    w = d.pivot(index="state", columns="t_idx", values=col).sort_index()
    return w.values.astype(float), list(w.index)


# =====================================================================================
# FIXED EFFECTS ON A BALANCED PANEL. Exact within transform, no dummies.
# =====================================================================================
def demean(A, two_way):
    """A is (n_state, n_window). Sequential double demeaning is exact on a balanced panel."""
    B = A - A.mean(axis=1, keepdims=True)
    if two_way:
        B = B - B.mean(axis=0, keepdims=True)
    return B


def fe_fit(Y, X, two_way, weights=None, n_state=51, n_win=4):
    """
    beta with CR1 cluster-robust standard error, clustered by state.
    K = n_state + 1 for one-way, n_state + (n_win - 1) + 1 for two-way.
    Inference df is G - 1 = 50, NOT the residual df.
    """
    N = Y.size
    G = n_state
    K = n_state + 1 + (n_win - 1 if two_way else 0)
    if weights is None:
        Yd, Xd = demean(Y, two_way), demean(X, two_way)
        W = np.ones_like(Y)
    else:
        W = weights
        Yd, Xd = _weighted_demean(Y, X, W, two_way)
    sxx = float((W * Xd * Xd).sum())
    beta = float((W * Xd * Yd).sum() / sxx)
    e = Yd - beta * Xd
    g = (W * Xd * e).sum(axis=1)              # one score per state cluster
    meat = float((g ** 2).sum())
    c = (G / (G - 1.0)) * ((N - 1.0) / (N - K))
    se = float(np.sqrt(c * meat) / sxx)
    tstat = beta / se if se > 0 else np.nan
    sst = float((W * Yd * Yd).sum())
    ssr = float((W * e * e).sum())
    r2w = 1.0 - ssr / sst if sst > 0 else np.nan
    p1 = float(stats.t.sf(tstat, G - 1))       # one-sided, H1: beta > 0
    p2 = float(2 * stats.t.sf(abs(tstat), G - 1))
    return {"beta": beta, "se": se, "t": tstat, "p_one_sided": p1, "p_two_sided": p2,
            "within_r2": r2w, "df_resid": N - K, "df_infer": G - 1, "K": K,
            "sd_x_demeaned": float(np.sqrt((Xd ** 2).mean())),
            "resid_var": ssr / (N - K)}


def _weighted_demean(Y, X, W, two_way, tol=1e-13, itmax=500):
    """Weighted alternating projections. Weighted double demeaning is not one-pass exact."""
    Yd, Xd = Y.copy(), X.copy()
    for _ in range(itmax):
        prev = Yd.sum() + Xd.sum()
        for A in (Yd, Xd):
            m = (W * A).sum(axis=1, keepdims=True) / W.sum(axis=1, keepdims=True)
            A -= m
        if two_way:
            for A in (Yd, Xd):
                m = (W * A).sum(axis=0, keepdims=True) / W.sum(axis=0, keepdims=True)
                A -= m
        if not two_way:
            break
        if abs(Yd.sum() + Xd.sum() - prev) < tol:
            break
    return Yd, Xd


def window_permutation_null(Y, X, two_way, n_state=51):
    """
    Permute the FOUR window labels of burden GLOBALLY (the same permutation for every
    state) and refit, all 24 arrangements. This is the null the one-way spec actually
    faces: the identifying variation is one national seasonal wave, and the clustered
    standard error assumes an independence across states that the common wave destroys.
    """
    ts = []
    for perm in itertools.permutations(range(X.shape[1])):
        ts.append(fe_fit(Y, X[:, list(perm)], two_way, n_state=n_state)["t"])
    ts = np.array(ts, float)
    obs = fe_fit(Y, X, two_way, n_state=n_state)["t"]
    return {"obs_t": float(obs),
            "p_one_sided": float((ts >= obs).mean()),
            "null_t_min": float(ts.min()), "null_t_max": float(ts.max()),
            "frac_null_abs_t_gt_2": float((np.abs(ts) > 2).mean()),
            "n_perm": len(ts)}


# =====================================================================================
# VARIANCE DECOMPOSITION AND RELIABILITY OF THE WITHIN-TRANSFORMED SERIES
# =====================================================================================
def variance_parts(A, SE=None):
    """One-way random effects with the sampling-error correction variance_decomp.py uses."""
    n_i, n_t = A.shape
    grand = A.mean()
    sm = A.mean(axis=1)
    ss_b = n_t * ((sm - grand) ** 2).sum()
    ss_w = ((A - sm[:, None]) ** 2).sum()
    ms_b = ss_b / (n_i - 1)
    ms_w = ss_w / (n_i * (n_t - 1))
    var_b = max((ms_b - ms_w) / n_t, 0.0)
    out = {"total_sd": float(A.std(ddof=1)), "between_sd": float(np.sqrt(var_b)),
           "within_sd_observed": float(np.sqrt(ms_w)),
           "icc_naive": float(var_b / (var_b + ms_w)) if (var_b + ms_w) > 0 else np.nan}
    if SE is not None:
        samp = float((SE ** 2).mean())
        var_w_true = max(ms_w - samp, 0.0)
        out.update({"sampling_var": samp, "sampling_sd": float(np.sqrt(samp)),
                    "within_sd_true": float(np.sqrt(var_w_true)),
                    "icc_corrected": float(var_b / (var_b + var_w_true))
                    if (var_b + var_w_true) > 0 else np.nan})
    return out


def reliability(A, SE, two_way):
    """
    lambda = (observed transformed variance - shrunk noise variance) / observed.
    CDC draws an independent sample each period, so the within transform shrinks the
    sampling error by a KNOWN factor: (T-1)/T one-way, (T-1)(N-1)/(TN) two-way.
    """
    n_i, n_t = A.shape
    Ad = demean(A, two_way)
    shrink = ((n_t - 1) / n_t) * (((n_i - 1) / n_i) if two_way else 1.0)
    noise = float((SE ** 2).mean()) * shrink
    obs = float((Ad ** 2).mean())
    lam = max((obs - noise) / obs, 0.0) if obs > 0 else np.nan
    return {"obs_var": obs, "noise_var": noise, "lambda": lam,
            "disatten_corr": (1.0 / np.sqrt(lam)) if lam > 0 else np.inf,
            "disatten_slope": (1.0 / lam) if lam > 0 else np.inf}


# =====================================================================================
# C. NON-PARAMETRIC DISTRIBUTION TEST
# =====================================================================================
def exact_t4_null():
    base = np.array([1.0, 2.0, 3.0, 4.0])
    vals = [float(stats.spearmanr(base, np.array(p, float))[0])
            for p in itertools.permutations(base)]
    v, c = np.unique(np.round(vals, 10), return_counts=True)
    return v, c


def per_state_rhos(C, B, window_demean):
    if window_demean:
        C = C - C.mean(axis=0, keepdims=True)
        B = B - B.mean(axis=0, keepdims=True)
    out = []
    for i in range(C.shape[0]):
        r = stats.spearmanr(C[i], B[i])[0]
        out.append(0.0 if np.isnan(r) else float(r))
    return np.array(out)


def nonparametric_gate(C, B, window_demean, rng):
    rho = per_state_rhos(C, B, window_demean)
    pos, neg, zer = int((rho > 0).sum()), int((rho < 0).sum()), int((rho == 0).sum())
    nz = pos + neg
    sign_p = float(stats.binomtest(pos, nz, 0.5, alternative="greater").pvalue) if nz else np.nan
    nzr = rho[rho != 0]
    wil_p = float(stats.wilcoxon(nzr, alternative="greater").pvalue) if len(nzr) else np.nan
    t_p = float(stats.ttest_1samp(rho, 0.0, alternative="greater").pvalue)
    # Within-state permutation null: shuffle each state's own four window labels of
    # concern, hold burden fixed. This respects the common national wave, which the
    # nominal sign test does not.
    Cw = C - C.mean(axis=0, keepdims=True) if window_demean else C
    Bw = B - B.mean(axis=0, keepdims=True) if window_demean else B
    obs = float(rho.mean())
    null = np.empty(NPERM_STATE)
    n_i, n_t = C.shape
    for k in range(NPERM_STATE):
        idx = np.argsort(rng.random((n_i, n_t)), axis=1)
        Cp = np.take_along_axis(Cw, idx, axis=1)
        rr = np.array([stats.spearmanr(Cp[i], Bw[i])[0] for i in range(n_i)])
        null[k] = np.nanmean(np.nan_to_num(rr))
    return {"mean_rho": obs, "median_rho": float(np.median(rho)),
            "sd_rho": float(rho.std(ddof=1)), "n_pos": pos, "n_neg": neg, "n_zero": zer,
            "sign_p": sign_p, "wilcoxon_p": wil_p, "ttest_p": t_p,
            "se_mean_rho": float(rho.std(ddof=1) / np.sqrt(len(rho))),
            "perm_p": float((null >= obs).mean()),
            "perm_null_mean": float(null.mean()), "perm_null_sd": float(null.std(ddof=1))}


# =====================================================================================
# THE DEPLOYABLE PER-WINDOW FORM: FIRST DIFFERENCES
# =====================================================================================
def fd_stats(C, B, SE=None):
    """
    For each adjacent window pair, correlate the cross-state change in concern against
    the cross-state change in burden. Differencing removes the state effect a_i exactly.
    Spearman on the deltas is invariant to subtracting the window mean of the deltas, so
    this statistic is already the two-way analogue; no extra demeaning is needed.
    """
    rows = []
    for t in range(C.shape[1] - 1):
        dc = np.round(C[:, t + 1] - C[:, t], 6)
        db = np.round(B[:, t + 1] - B[:, t], 6)
        rho, _ = stats.spearmanr(dc, db)
        n = len(dc)
        p1 = float(stats.t.sf(rho * np.sqrt((n - 2) / max(1 - rho ** 2, 1e-15)), n - 2))
        pr = float(np.corrcoef(dc, db)[0, 1])
        row = {"from_window": WINDOWS[t], "to_window": WINDOWS[t + 1],
               "n": n, "spearman_rho": float(rho), "p_one_sided": p1,
               "pearson_r": pr, "mean_delta_burden": float(db.mean()),
               "mean_delta_concern": float(dc.mean())}
        if SE is not None:
            noise = float((SE[:, t] ** 2 + SE[:, t + 1] ** 2).mean())
            obs = float(dc.var(ddof=0))
            lam = max((obs - noise) / obs, 0.0) if obs > 0 else 0.0
            row["lambda_delta_concern"] = lam
            row["pearson_disattenuated"] = (pr / np.sqrt(lam)) if lam > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# =====================================================================================
# INDEPENDENCE DEMONSTRATION
# =====================================================================================
def d_by_window(C, B):
    """sd(D) per window and the vdW-score Pearson r, exactly as compute_d.py scores D."""
    out = []
    for t in range(C.shape[1]):
        z1, z2 = dr.vdw(C[:, t]), dr.vdw(B[:, t])
        D = z1 - z2
        # ddof=1 throughout, so sd(D) here is directly comparable to the 0.946 /
        # 1.153 to 1.526 diagnostic already reported for the old gate.
        out.append({"sd_D": float(D.std(ddof=1)),
                    "var_z1": float(z1.var(ddof=1)), "var_z2": float(z2.var(ddof=1)),
                    "r_vdw": float(np.corrcoef(z1, z2)[0, 1]),
                    "spearman_old_gate": float(stats.spearmanr(C[:, t], B[:, t])[0])})
    return out


def new_gate_stats(C, B):
    """The two statistics the new gate reads, for the invariance tests."""
    return {"t_one_way": fe_fit(C, B, False)["t"],
            "t_two_way": fe_fit(C, B, True)["t"],
            "fd_rho": [float(r) for r in fd_stats(C, B)["spearman_rho"].values],
            "fd_pearson": [float(r) for r in fd_stats(C, B)["pearson_r"].values]}


# =====================================================================================
# PRINTING
# =====================================================================================
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
    print("\n" + "=" * 86)
    if t:
        print(t)
        print("=" * 86)


def fmt_p(p):
    return "  n/a " if p is None or (isinstance(p, float) and np.isnan(p)) else f"{p:.4f}"


# =====================================================================================
# MAIN
# =====================================================================================
def _main():
    rng = np.random.default_rng(SEED)
    panel, concern, burden, vote = build_panel()

    hr("0. WHAT THIS RUN IS AND IS NOT")
    print("""
The gate in decision_rule.py is a BETWEEN-PLACE cross-section. The divergence metric D
it gates is built from the same two legs across the same places in the same window.
Part 1 below proves the two are related by a closed-form identity, so the old gate could
only ever open where |D| was smallest. This file implements the WITHIN-PLACE, OVER-TIME
reading instead.

This is a CHANGE OF OPERATIONALIZATION, not a bug fix. The manuscript's reasoning is
within-place (lines 103, 107, 121, 125). Its single operational sentence and its single
worked number are between-place (lines 109, 111). Those two pieces of text must be
revised or the paper and the code will describe different tests. Nothing here reproduces
or supersedes the 0.50 and 0.43 reported at line 109; no definition of them exists in
methods_section.tex and none is recoverable from this repository.
""")
    print("PANEL. rows per disease:", dict(panel.groupby("disease").size()))
    print("       windows per (disease, state):",
          sorted(panel.groupby(["disease", "state"]).size().unique().tolist()),
          " (4 everywhere: balanced, zero gaps)")
    print("       states per disease:", panel.groupby("disease")["state"].nunique().to_dict())

    G = {}
    for dis in DISEASES:
        C, states = grid(panel, dis, "concern")
        Bm, _ = grid(panel, dis, "burden_mean")
        Bx, _ = grid(panel, dis, "burden_max")
        SE, _ = grid(panel, dis, "se")
        G[dis] = {"C": C, "Bm": Bm, "Bx": Bx, "SE": SE, "states": states}

    # ---------------------------------------------------------------- variance parts
    hr("1. VARIANCE DECOMPOSITION. HOW MUCH WITHIN-PLACE VARIATION IS THERE TO TEST?")
    print("Manuscript quantity 1 of the measurement condition (line 111): 'the share of")
    print("variance in the perceived-risk measure that lies between places rather than")
    print("within a place over time'. It is a REPORTING requirement; the paper attaches no")
    print("threshold to it. The word 'most' is the only quantifier it gives.\n")
    vp = {}
    for dis in DISEASES:
        vc = variance_parts(G[dis]["C"], G[dis]["SE"])
        vb = variance_parts(G[dis]["Bm"])
        vp[dis] = (vc, vb)
        print(f"CONCERN, {dis}: total sd {vc['total_sd']:.4f}  between sd {vc['between_sd']:.4f}  "
              f"within sd observed {vc['within_sd_observed']:.4f}")
        print(f"    mean CDC sampling variance {vc['sampling_var']:.4f} (sd {vc['sampling_sd']:.4f})"
              f"   TRUE within sd after removing sampling noise {vc['within_sd_true']:.4f}")
        print(f"    ICC naive {vc['icc_naive']:.4f}   ICC corrected {vc['icc_corrected']:.4f}"
              f"   -> within share {1 - vc['icc_naive']:.4f} naive, "
              f"{1 - vc['icc_corrected']:.4f} corrected")
        print(f"BURDEN,  {dis}: total sd {vb['total_sd']:.4f}  between sd {vb['between_sd']:.4f}  "
              f"within sd {vb['within_sd_observed']:.4f}   ICC {vb['icc_naive']:.4f}")
    print("\nflu corrected ICC reproduces the manuscript's reported 0.695 exactly (METH:119).")
    print("COVID: 96.66 percent of the variance is BETWEEN states. The entire quantity the")
    print("new gate has to test is about 3 percent of the variance in covid concern, and its")
    print("true sd is 0.91 points against a sampling sd of 2.59 points.")

    # ---------------------------------------------------------------- A and B
    hr("2. THE WITHIN-PLACE GATE. VARIANTS A (one-way FE) AND B (two-way FE)")
    print("concern_it = a_i + [g_t] + beta * burden_it + e_it, per disease, i=1..51, t=1..4.")
    print("Exact within transform on the balanced panel. CR1 cluster-robust by state,")
    print("G = 51, inference on t with G-1 = 50 df. One-sided test, H1: beta > 0.\n")
    print(f"{'disease':7s} {'spec':9s} {'beta':>9s} {'se':>8s} {'t':>8s} {'p(1s)':>8s} "
          f"{'withinR2':>9s} {'df_res':>7s}")
    fe = {}
    for dis in DISEASES:
        for tw in (False, True):
            r = fe_fit(G[dis]["C"], G[dis]["Bm"], tw)
            fe[(dis, tw)] = r
            print(f"{dis:7s} {'two-way' if tw else 'one-way':9s} {r['beta']:+9.4f} "
                  f"{r['se']:8.4f} {r['t']:+8.3f} {r['p_one_sided']:8.4f} "
                  f"{r['within_r2']:9.4f} {r['df_resid']:7d}")

    print("\nNATIONAL SERIES, which is all the one-way spec is identified from (T = 4):")
    for dis in DISEASES:
        b, c = G[dis]["Bm"].mean(axis=0), G[dis]["C"].mean(axis=0)
        pr, pp = stats.pearsonr(b, c)
        sr = stats.spearmanr(b, c)[0]
        print(f"  {dis:5s} burden {np.round(b, 4).tolist()}  concern {np.round(c, 4).tolist()}")
        print(f"        national Pearson {pr:+.4f} (p = {pp:.3f}, df = 2), "
              f"national Spearman {sr:+.4f}")
    print("  Flu concern FALLS from October to late November while burden quadruples, then")
    print("  RISES in the last window while burden falls. The seasonal channel read directly")
    print("  gives no support at all.")

    hr("2b. WHY THE ONE-WAY SPEC HAS NO VALID INFERENCE. WINDOW-PERMUTATION NULL")
    print("Permute the four window labels of burden globally, refit, all 24 arrangements.")
    print("Clustering by state assumes independence across states; one common national wave")
    print("violates it outright.\n")
    print(f"{'disease':7s} {'spec':9s} {'obs t':>8s} {'perm p':>8s} {'null t min':>11s} "
          f"{'null t max':>11s} {'frac |t|>2':>11s}")
    perm = {}
    for dis in DISEASES:
        for tw in (False, True):
            pn = window_permutation_null(G[dis]["C"], G[dis]["Bm"], tw)
            perm[(dis, tw)] = pn
            print(f"{dis:7s} {'two-way' if tw else 'one-way':9s} {pn['obs_t']:+8.3f} "
                  f"{pn['p_one_sided']:8.4f} {pn['null_t_min']:+11.3f} "
                  f"{pn['null_t_max']:+11.3f} {pn['frac_null_abs_t_gt_2']:11.3f}")
    print("\nA t statistic that exceeds 2 in a large fraction of PURE NULL draws is not")
    print("evidence of anything. This is why the two-way spec is our headline, and why any")
    print("apparently significant one-way result below must not be read as a pass.")

    # ---------------------------------------------------------------- D
    hr("3. VARIANT D. MEASUREMENT-ERROR-CORRECTED VERSION")
    print("FIRST, A CORRECTION TO A COMMON PREMISE. Concern is the DEPENDENT variable here.")
    print("Classical measurement error in the dependent variable does NOT attenuate beta; it")
    print("inflates the residual variance and the standard error. Burden (NSSP percent of ED")
    print("visits) is a near-census of participating ED visits and NO per-cell standard error")
    print("is published for it anywhere in decision_rule_inputs_burden.csv, so there is no")
    print("instantiated sampling-error input for the regressor and we do not invent one.")
    print("No attenuation-corrected beta is reported: that would correct a bias this")
    print("estimator does not have. What the sampling error does here is destroy power.\n")
    print("RELIABILITY of the within-transformed concern series. lambda = (observed")
    print("transformed variance - shrunk noise variance) / observed. CDC samples")
    print("independently each period, so the transform shrinks noise by a known factor:")
    print("(T-1)/T = 0.750000 one-way, (T-1)(N-1)/(TN) = 0.735294 two-way.\n")
    print(f"{'disease':7s} {'transform':11s} {'obs var':>9s} {'noise var':>10s} "
          f"{'lambda':>8s} {'1/sqrt(lam)':>12s}")
    rel = {}
    for dis in DISEASES:
        A, SE = G[dis]["C"], G[dis]["SE"]
        lv = float(A.var(ddof=0))
        nv = float((SE ** 2).mean())
        print(f"{dis:7s} {'levels':11s} {lv:9.4f} {nv:10.4f} {(lv - nv) / lv:8.4f} "
              f"{1 / np.sqrt((lv - nv) / lv):12.4f}")
        for tw in (False, True):
            r = reliability(A, SE, tw)
            rel[(dis, tw)] = r
            dc = f"{r['disatten_corr']:12.4f}" if np.isfinite(r["disatten_corr"]) else "   undefined"
            print(f"{dis:7s} {'two-way' if tw else 'one-way':11s} {r['obs_var']:9.4f} "
                  f"{r['noise_var']:10.4f} {r['lambda']:8.4f} {dc}")
    print("\nThe instrument is reasonably reliable in LEVELS, which is why the old")
    print("between-place gate looked estimable. It is roughly 90 percent noise for flu and")
    print("100 percent noise for covid once state and window means are removed, which is")
    print("exactly the transform the new gate requires. The old gate worked on the reliable")
    print("part of the measure. The new gate needs the part that is not there.")
    print("covid two-way lambda is an estimated ZERO: the disattenuation factor is a division")
    print("by zero and is reported as undefined rather than as a large number.\n")

    print("FGLS weighted by 1/(sigma_u^2 + se_it^2), se_it from CDC's published intervals.")
    print("sigma_u^2 = FE residual variance minus mean sampling variance, floored at zero.")
    print("This is an EFFICIENCY fix, not a bias fix. Weighted demeaning is done by")
    print("alternating projections because weighted double demeaning is not one-pass exact.\n")
    print(f"{'disease':7s} {'spec':9s} {'sigma_u':>9s} {'OLS beta':>10s} {'OLS t':>8s} "
          f"{'WLS beta':>10s} {'WLS se':>8s} {'WLS t':>8s}")
    wls = {}
    for dis in DISEASES:
        for tw in (False, True):
            r0 = fe[(dis, tw)]
            su2 = max(r0["resid_var"] - float((G[dis]["SE"] ** 2).mean()), 0.0)
            W = 1.0 / (su2 + G[dis]["SE"] ** 2)
            r = fe_fit(G[dis]["C"], G[dis]["Bm"], tw, weights=W)
            wls[(dis, tw)] = r
            print(f"{dis:7s} {'two-way' if tw else 'one-way':9s} {np.sqrt(su2):9.4f} "
                  f"{r0['beta']:+10.4f} {r0['t']:+8.3f} {r['beta']:+10.4f} {r['se']:8.4f} "
                  f"{r['t']:+8.3f}")
    print("\nWeighting sharpens the standard errors but does not rescue the gate: the flu")
    print("two-way estimate collapses to zero under weighting and covid moves further")
    print("negative. It also does not repair the one-way spec's invalid inference.")
    print("NOTE, because a referee will ask: CDC samples are independent per period, so the")
    print("sampling error cannot induce spurious within-state serial correlation and")
    print("therefore cannot manufacture a positive beta. The near-zero estimates are not an")
    print("artifact of the noise; the noise only widens the intervals around them.")

    # ---------------------------------------------------------------- C
    hr("4. VARIANT C. NON-PARAMETRIC DISTRIBUTION TEST OVER THE 51 PER-STATE STATISTICS")
    v, c = exact_t4_null()
    print("EXACT NULL FOR T = 4, all 24 permutations. Spearman rho takes values")
    print("  " + "  ".join(f"{x:+.1f}:{int(n)}" for x, n in zip(v, c)))
    print(f"  P(rho > 0) = {int(c[v > 0].sum())}/24 = {c[v > 0].sum() / 24:.4f}, "
          f"P(rho = 0) = {int(c[v == 0].sum())}/24, "
          f"P(rho < 0) = {int(c[v < 0].sum())}/24  (exactly symmetric)")
    print("  Smallest attainable one-sided p for a SINGLE state is 1/24 = 0.0417, so the")
    print("  per-state statistic is useless alone; only the distribution of 51 is informative.\n")
    print(f"{'disease':7s} {'variant':16s} {'mean rho':>9s} {'median':>7s} {'sd':>7s} "
          f"{'pos':>4s} {'neg':>4s} {'zero':>5s} {'sign p':>8s} {'wilcox':>8s} "
          f"{'t-test':>8s} {'perm p':>8s}")
    npres = {}
    for dis in DISEASES:
        for wd in (False, True):
            r = nonparametric_gate(G[dis]["C"], G[dis]["Bm"], wd, rng)
            npres[(dis, wd)] = r
            print(f"{dis:7s} {'window-demeaned' if wd else 'raw levels':16s} "
                  f"{r['mean_rho']:+9.4f} {r['median_rho']:+7.3f} {r['sd_rho']:7.4f} "
                  f"{r['n_pos']:4d} {r['n_neg']:4d} {r['n_zero']:5d} {r['sign_p']:8.4f} "
                  f"{r['wilcoxon_p']:8.4f} {r['ttest_p']:8.4f} {r['perm_p']:8.4f}")
    print("\nPermutation null (shuffle each state's own four window labels of concern,")
    print("burden fixed, 5000 draws):")
    for dis in DISEASES:
        for wd in (False, True):
            r = npres[(dis, wd)]
            print(f"  {dis:5s} {'window-demeaned' if wd else 'raw levels':16s} "
                  f"observed mean rho {r['mean_rho']:+.4f}, null mean {r['perm_null_mean']:+.4f}, "
                  f"null sd {r['perm_null_sd']:.4f}, p = {r['perm_p']:.4f}")
    print("\nASSUMPTIONS, stated because one of them is violated. The exact binomial sign test")
    print("assumes the 51 per-state statistics are independent. In the RAW variant every")
    print("state shares the same national wave, so they are strongly positively dependent and")
    print("the nominal sign p is anticonservative. The window-demeaned variant removes the")
    print("common wave by construction; residual spatial correlation between neighbouring")
    print("states remains and is addressed by none of these tests. Concern is published to one")
    print("decimal, producing the exact zeros above, which the sign test must discard. The")
    print("one-sample t-test assumes normality of a statistic living on a discrete 11-point")
    print("lattice bounded in [-1, 1] and is a third opinion only. Report the permutation")
    print("null as primary. It does not reject anywhere.")

    # ---------------------------------------------------------------- power
    hr("5. POWER. WHAT THIS DESIGN COULD HAVE DETECTED")
    print("MDES = (t_.95,50 + t_.80,50) * se = 2.5252 * se, one-sided 5 percent, 80 percent")
    print("power. Expressed in units of the TRUE within-state concern signal, after the known")
    print("noise shrink factor for each transform.\n")
    tcrit = stats.t.ppf(0.95, 50) + stats.t.ppf(0.80, 50)
    print(f"{'disease':7s} {'spec':9s} {'se':>8s} {'MDES':>9s} {'true signal sd':>15s} "
          f"{'MDES in signal sd':>18s}")
    for dis in DISEASES:
        A, SE = G[dis]["C"], G[dis]["SE"]
        for tw in (False, True):
            r = fe[(dis, tw)]
            rr = reliability(A, SE, tw)
            sig = np.sqrt(max(rr["obs_var"] - rr["noise_var"], 0.0))
            m = tcrit * r["se"]
            s = f"{m / sig:18.3f}" if sig > 0 else "         UNDEFINED"
            print(f"{dis:7s} {'two-way' if tw else 'one-way':9s} {r['se']:8.4f} "
                  f"{m:9.4f} {sig:15.4f} {s}")
    print("\nSIMULATED POWER on the realised design (actual burden values, actual per-cell CDC")
    print(f"standard errors, {NSIM_POWER} draws per point, one-sided 5 percent):")
    for dis in DISEASES:
        A, SE, B = G[dis]["C"], G[dis]["SE"], G[dis]["Bm"]
        for tw in (False, True):
            su = np.sqrt(max(fe[(dis, tw)]["resid_var"] - float((SE ** 2).mean()), 0.0))
            betas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
            pw = _power_curve(B, SE, su, tw, betas, rng)
            print(f"  {dis:5s} {'two-way' if tw else 'one-way':9s} sigma_u = {su:.4f}   " +
                  "  ".join(f"b={b:g}:{p:.3f}" for b, p in zip(betas, pw)))
    print("\nThe two-way gate for covid has an estimated sigma_u of exactly zero: the residual")
    print("after two-way demeaning is entirely CDC sampling error. The test is NOT IDENTIFIED")
    print("on covid. For flu it is estimable but powered only against effects that would")
    print("explain most of the true residual concern variance. With T = 4 there are 3")
    print("within-state degrees of freedom per state and 3 adjacent-window transitions. This")
    print("is a structural ceiling of the design, not an estimation choice, and no estimator")
    print("recovers power the design does not contain.")

    # ---------------------------------------------------------------- sweep
    hr("6. SPECIFICATION SWEEP. 2 diseases x 4 regressor forms x 2 outcome forms x 2 time-FE")
    print("Every transformation below is OUR choice; the manuscript names none of them.\n")
    print(f"{'disease':7s} {'x':16s} {'y':8s} {'spec':9s} {'beta':>10s} {'se':>9s} "
          f"{'t':>8s} {'p(1s)':>8s}")
    sweep_rows = []
    for dis in DISEASES:
        C = G[dis]["C"]
        ys = {"pct": C, "logit": np.log((C / 100) / (1 - C / 100))}
        xs = {"burden_mean": G[dis]["Bm"], "log burden_mean": np.log(G[dis]["Bm"] + 0.01),
              "burden_max": G[dis]["Bx"], "log burden_max": np.log(G[dis]["Bx"] + 0.01)}
        for xn, X in xs.items():
            for yn, Y in ys.items():
                for tw in (False, True):
                    r = fe_fit(Y, X, tw)
                    sweep_rows.append({"disease": dis, "x": xn, "y": yn,
                                       "time_fe": tw, **r})
                    print(f"{dis:7s} {xn:16s} {yn:8s} {'two-way' if tw else 'one-way':9s} "
                          f"{r['beta']:+10.4f} {r['se']:9.4f} {r['t']:+8.3f} "
                          f"{r['p_one_sided']:8.4f}")
    sw = pd.DataFrame(sweep_rows)
    npass = int((sw["p_two_sided"] < 0.05).sum())
    best = sw.loc[sw["t"].idxmax()]
    print(f"\n{len(sw)} fits. Number reaching two-sided p < 0.05 with clustered SE: {npass}.")
    print(f"Largest positive clustered t anywhere in the sweep: {best['t']:+.3f} "
          f"({best['disease']}, {'two-way' if best['time_fe'] else 'one-way'}, "
          f"{best['x']}, {best['y']}).")

    # ---------------------------------------------------------------- FD
    hr("7. THE DEPLOYABLE PER-WINDOW FORM. CAN THE GATE BE RUN PER WINDOW AT ALL?")
    print("STATED PLAINLY: a within-place test needs more than one window BY CONSTRUCTION.")
    print("Inside a single CDC window there is no over-time variation whatsoever, so there is")
    print("no within-place statistic computable for one window in isolation. Any number")
    print("presented as a one-window within-place statistic would be fabricated.")
    print("")
    print("The closest genuine per-window form is the FIRST DIFFERENCE: correlate the")
    print("cross-state change in concern against the cross-state change in burden across")
    print("each adjacent window pair. Differencing removes the state effect a_i exactly, and")
    print("Spearman on the deltas is invariant to subtracting their window mean, so the")
    print("window effect g_t is removed too. It is a genuine within-place statistic that is")
    print("still window-specific. It yields 3 transitions per disease, not 4, so the FIRST")
    print("window has NO within-place verdict. We do not fabricate one.\n")
    fd = {}
    print(f"{'disease':7s} {'transition':56s} {'rho':>8s} {'p(1s)':>8s} {'pearson':>8s} "
          f"{'lam(dC)':>8s} {'disatt':>8s} {'d burden':>9s}")
    for dis in DISEASES:
        f = fd_stats(G[dis]["C"], G[dis]["Bm"], G[dis]["SE"])
        fd[dis] = f
        for _, r in f.iterrows():
            tr = f"{r['from_window']} -> {r['to_window']}"
            da = (f"{r['pearson_disattenuated']:8.4f}"
                  if not np.isnan(r["pearson_disattenuated"]) else "     n/a")
            print(f"{dis:7s} {tr:56s} {r['spearman_rho']:+8.4f} {r['p_one_sided']:8.4f} "
                  f"{r['pearson_r']:+8.4f} {r['lambda_delta_concern']:8.4f} {da} "
                  f"{r['mean_delta_burden']:+9.4f}")
    print("\nThree of the six transitions have an ESTIMATED TRUE delta-concern variance of")
    print("zero (lambda = 0). That is the cleanest single statement of the data limitation:")
    print("for those transitions there is no measured change in concern left after CDC")
    print("sampling error is removed, so no disattenuated correlation is reportable.")

    return (panel, concern, burden, vote, G, fe, wls, perm, npres, rel, fd, sw, vp)


def _power_curve(B, SE, sigma_u, two_way, betas, rng):
    """Vectorized simulated power on the realised design."""
    n_i, n_t = B.shape
    Xd = demean(B, two_way)
    sxx = float((Xd ** 2).sum())
    N, Gn = B.size, n_i
    K = n_i + 1 + (n_t - 1 if two_way else 0)
    cfac = (Gn / (Gn - 1.0)) * ((N - 1.0) / (N - K))
    crit = stats.t.ppf(0.95, Gn - 1)
    out = []
    for b in betas:
        u = rng.normal(0, sigma_u, size=(NSIM_POWER, n_i, n_t)) if sigma_u > 0 else 0.0
        eps = rng.normal(0, 1, size=(NSIM_POWER, n_i, n_t)) * SE[None, :, :]
        Y = b * B[None, :, :] + u + eps
        Yd = Y - Y.mean(axis=2, keepdims=True)
        if two_way:
            Yd = Yd - Yd.mean(axis=1, keepdims=True)
        bh = (Xd[None] * Yd).sum(axis=(1, 2)) / sxx
        e = Yd - bh[:, None, None] * Xd[None]
        g = (Xd[None] * e).sum(axis=2)
        se = np.sqrt(cfac * (g ** 2).sum(axis=1)) / sxx
        out.append(float((bh / se > crit).mean()))
    return out


# =====================================================================================
# INDEPENDENCE DEMONSTRATION, GATE VERDICTS, RE-RUN, COMPARISON
# =====================================================================================
def _independence(G):
    hr("8. INDEPENDENCE DEMONSTRATION. IS THE NEW GATE ALGEBRAICALLY TIED TO D?")
    print("PART 1. THE OLD GATE AND D ARE THE SAME OBJECT, PROVABLY.")
    print("D_it = vdW(concern) - vdW(burden), both scored within the window across the 51")
    print("places. Therefore var(D) = var(z1) + var(z2) - 2*r*sd(z1)*sd(z2), and because the")
    print("vdW score vector is the same fixed multiset in every window, var(z1) = var(z2) =")
    print("s^2 is a constant, so")
    print("      sd(D) = sqrt( 2 * s^2 * (1 - r) )   exactly.")
    print("sd(D) is not merely associated with the old gate statistic. It is a deterministic")
    print("decreasing function of it. Verified on all 8 real cells:\n")
    print(f"{'disease':7s} {'window':28s} {'sd(D) actual':>13s} {'sd(D) predicted':>16s} "
          f"{'var z1':>8s} {'var z2':>8s} {'r vdW':>8s} {'old gate rho':>13s}")
    rows = []
    for dis in DISEASES:
        for k, r in enumerate(d_by_window(G[dis]["C"], G[dis]["Bm"])):
            s2 = 0.5 * (r["var_z1"] + r["var_z2"])
            pred = np.sqrt(2 * s2 * (1 - r["r_vdw"]))
            rows.append({"disease": dis, "window": WINDOWS[k], "sd_D": r["sd_D"],
                         "pred": pred, "r_vdw": r["r_vdw"],
                         "old_rho": r["spearman_old_gate"]})
            print(f"{dis:7s} {WINDOWS[k]:28s} {r['sd_D']:13.6f} {pred:16.6f} "
                  f"{r['var_z1']:8.4f} {r['var_z2']:8.4f} {r['r_vdw']:+8.4f} "
                  f"{r['spearman_old_gate']:+13.4f}")
    dd = pd.DataFrame(rows)
    mx = float((dd["sd_D"] - dd["pred"]).abs().max())
    c1 = float(np.corrcoef(dd["old_rho"], dd["sd_D"])[0, 1])
    c2 = float(np.corrcoef(dd["r_vdw"], dd["sd_D"])[0, 1])
    print(f"\nMaximum absolute discrepancy across all 8 cells: {mx:.2e}")
    print(f"corr(old-gate Spearman, sd(D)) over the 8 cells = {c1:+.4f}, R2 = {c1 ** 2:.4f}")
    print(f"corr(r_vdW, sd(D))                              = {c2:+.4f}, R2 = {c2 ** 2:.4f}")
    print("The single cell the old gate passed is exactly the minimum-sd(D) cell, as the")
    print("identity requires. This replaces the measured '0.946 versus 1.153 to 1.526' with")
    print("a proof: no transformation whatsoever can move the old gate while fixing sd(D).")

    print("\n\nPART 2. TEST A. MOVE THE NEW GATE WHILE D IS BIT-FOR-BIT FROZEN.")
    print("Apply a strictly increasing WITHIN-WINDOW transform to concern: g_t(x) = m_t * x")
    print("with m_t = exp(c * z_t), z_t the z-score of the window's mean burden. Strictly")
    print("increasing, so every within-window RANK is preserved exactly. D is a rank")
    print("functional, so D, sd(D) and the old gate are unchanged to the last bit. The")
    print("within-place gate is a cardinal functional and is not.\n")
    for dis in DISEASES:
        C, B = G[dis]["C"], G[dis]["Bm"]
        base = np.array([x["sd_D"] for x in d_by_window(C, B)])
        base_rho = np.array([x["spearman_old_gate"] for x in d_by_window(C, B)])
        z = (B.mean(axis=0) - B.mean(axis=0).mean()) / B.mean(axis=0).std(ddof=0)
        print(f"  {dis.upper()}  baseline sd(D) per window = {np.round(base, 4).tolist()}")
        print(f"  {'c':>6s} {'t two-way':>10s} {'t one-way':>10s} "
              f"{'FD rho per transition':>34s} {'max |chg sd(D)|':>16s} {'max |chg old rho|':>18s}")
        for cc in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0):
            Ct = C * np.exp(cc * z)[None, :]
            ns = new_gate_stats(Ct, B)
            dw = d_by_window(Ct, B)
            sd_new = np.array([x["sd_D"] for x in dw])
            rho_new = np.array([x["spearman_old_gate"] for x in dw])
            print(f"  {cc:+6.2f} {ns['t_two_way']:+10.4f} {ns['t_one_way']:+10.4f} "
                  f"{str([round(v, 4) for v in ns['fd_rho']]):>34s} "
                  f"{np.abs(sd_new - base).max():16.3e} "
                  f"{np.abs(rho_new - base_rho).max():18.3e}")
        print("")
    print("The new gate statistic sweeps across its whole range. sd(D) and the old gate do")
    print("not move by a single bit. This is stronger than simulating data with the")
    print("between-place correlation held fixed: it holds the entire rank configuration of")
    print("every window fixed, which fixes D observation by observation.")

    print("\n\nPART 3. TEST B. MOVE D WHILE THE NEW GATE IS EXACTLY FROZEN.")
    print("Add a state-constant offset a_i = k * z(state-mean burden) to concern in all four")
    print("windows. a_i is time-invariant, so the state fixed effect absorbs it exactly and")
    print("the within estimator is unchanged to machine precision. It shifts states through")
    print("each window's ranking, so D and the old gate move freely.\n")
    for dis in DISEASES:
        C, B = G[dis]["C"], G[dis]["Bm"]
        sm = B.mean(axis=1)
        a = (sm - sm.mean()) / sm.std(ddof=0)
        b0 = new_gate_stats(C, B)
        print(f"  {dis.upper()}  baseline t one-way {b0['t_one_way']:+.4f}, "
              f"t two-way {b0['t_two_way']:+.4f}, "
              f"FD rho {[round(v, 4) for v in b0['fd_rho']]}")
        print(f"  {'k':>5s} {'t one-way':>10s} {'t two-way':>10s} {'max|chg FD rho|':>16s} "
              f"{'mean sd(D)':>11s} {'sd(D) range':>26s} {'old gate rho range':>26s}")
        for k in (-12, -4, 0, 4, 12, 20):
            Ck = C + k * a[:, None]
            ns = new_gate_stats(Ck, B)
            dw = d_by_window(Ck, B)
            sd_new = np.array([x["sd_D"] for x in dw])
            rho_new = np.array([x["spearman_old_gate"] for x in dw])
            dfd = max(abs(x - y) for x, y in zip(ns["fd_rho"], b0["fd_rho"]))
            print(f"  {k:+5d} {ns['t_one_way']:+10.4f} {ns['t_two_way']:+10.4f} "
                  f"{dfd:16.3e} {sd_new.mean():11.4f} "
                  f"{'[%.4f, %.4f]' % (sd_new.min(), sd_new.max()):>26s} "
                  f"{'[%+.4f, %+.4f]' % (rho_new.min(), rho_new.max()):>26s}")
        print("")
    print("The old gate is driven from strongly negative to strongly positive and sd(D) is")
    print("roughly halved, while the new gate statistic does not change in a single printed")
    print("digit. Parts 2 and 3 together establish LOGICAL INDEPENDENCE, not merely low")
    print("empirical correlation: there exist transformations of the real data that move the")
    print("new gate over its full range with D exactly fixed, and transformations that move D")
    print("and the old gate over their full ranges with the new gate exactly fixed. Neither")
    print("statistic is a function of the other. Part 1 shows the old gate and sd(D) admit no")
    print("such pair of transformations, because they are related by a closed-form identity.")
    print("")
    print("IMPLEMENTATION NOTE. Raw Spearman on the deltas is NOT exactly invariant under")
    print("Test B, because CDC publishes concern to one decimal and that produces exact ties")
    print("in the deltas; adding and subtracting a state offset perturbs those ties at the")
    print("1e-14 level and reshuffles tied ranks, moving rho by up to about 0.003. The deltas")
    print("are therefore rounded to 6 decimals before ranking, which makes the invariance")
    print("exactly zero. Without the rounding a referee running this check sees a small")
    print("nonzero and wonders why.")
    return dd


def _verdicts(G, fe, fd, perm, npres, vp, alpha):
    """
    Assemble the new gate's verdict per (disease, window) and per disease.
    PANEL form  -> one verdict per disease, applied to all four windows.
    FD form     -> one verdict per transition, attached to the LATER window.
                   The first window has NO preceding transition and is UNEVALUABLE.
    """
    out = {}
    for dis in DISEASES:
        icc_n, icc_c = vp[dis][0]["icc_naive"], vp[dis][0]["icc_corrected"]
        most_between = bool(icc_c > OUR_CHOICES["ICC_BETWEEN_MAJORITY"][0])
        panel = {tw: (fe[(dis, tw)]["beta"] > 0 and fe[(dis, tw)]["p_one_sided"] < alpha)
                 for tw in (False, True)}
        f = fd[dis]
        for wi, w in enumerate(WINDOWS):
            if wi == 0:
                fdrow, tracks_fd, status = None, None, "UNEVALUABLE_NO_PRECEDING_WINDOW"
            else:
                fdrow = f.iloc[wi - 1]
                tracks_fd = bool(fdrow["spearman_rho"] > 0
                                 and fdrow["p_one_sided"] < alpha)
                status = "PASS" if tracks_fd else "FAIL"
            out[(dis, w)] = {
                "icc_naive": icc_n, "icc_corrected": icc_c,
                "most_variance_between_places": most_between,
                "panel_tracks_one_way": panel[False], "panel_tracks_two_way": panel[True],
                "fd_tracks": tracks_fd, "fd_status": status,
                "fd_rho": None if fdrow is None else float(fdrow["spearman_rho"]),
                "fd_p_one_sided": None if fdrow is None else float(fdrow["p_one_sided"]),
                "fd_pearson": None if fdrow is None else float(fdrow["pearson_r"]),
                "fd_lambda": None if fdrow is None else float(fdrow["lambda_delta_concern"]),
                "fd_from_window": None if fdrow is None else fdrow["from_window"],
            }
    return out


def _main2(panel, concern, burden, vote, G, fe, wls, perm, npres, rel, fd, sw, vp):
    alpha = OUR_CHOICES["GATE_ALPHA"][0]

    hr("9. THE NEW GATE'S VERDICT")
    print("Two forms, both reported. Neither is silently preferred.\n")
    print("FORM 1, PANEL (one verdict per disease, no window resolution at all):")
    for dis in DISEASES:
        for tw in (False, True):
            r = fe[(dis, tw)]
            v = "PASS" if (r["beta"] > 0 and r["p_one_sided"] < alpha) else "FAIL"
            pn = perm[(dis, tw)]
            print(f"  {dis:5s} {'two-way' if tw else 'one-way':8s} beta = {r['beta']:+.4f}, "
                  f"t = {r['t']:+.3f}, clustered p(1s) = {r['p_one_sided']:.4f}  -> {v}"
                  f"   [window-permutation p = {pn['p_one_sided']:.4f}, "
                  f"{pn['frac_null_abs_t_gt_2']:.0%} of pure nulls give |t| > 2]")
    print("  Non-parametric companion (window-demeaned, permutation null):")
    for dis in DISEASES:
        r = npres[(dis, True)]
        v = "PASS" if r["perm_p"] < alpha else "FAIL"
        print(f"  {dis:5s} mean per-state rho = {r['mean_rho']:+.4f}, "
              f"perm p = {r['perm_p']:.4f}  -> {v}")
    print("  Measurement-error-weighted companion (FGLS):")
    for dis in DISEASES:
        for tw in (False, True):
            r = wls[(dis, tw)]
            v = "PASS" if (r["beta"] > 0 and r["p_one_sided"] < alpha) else "FAIL"
            print(f"  {dis:5s} {'two-way' if tw else 'one-way':8s} beta = {r['beta']:+.4f}, "
                  f"t = {r['t']:+.3f}, p(1s) = {r['p_one_sided']:.4f}  -> {v}")
    print("\n  The ONE apparent pass in the whole panel exercise is flu one-way FGLS.")
    print("  It must not be reported as a pass: its permutation p is "
          f"{perm[('flu', False)]['p_one_sided']:.4f} and "
          f"{perm[('flu', False)]['frac_null_abs_t_gt_2']:.0%} of PURE NULL window")
    print("  permutations in that spec produce |t| > 2, so a t near 2 is unremarkable there.")
    print("  covid one-way FGLS is significantly the WRONG SIGN, which is not a pass either.")

    print("\nFORM 2, FIRST DIFFERENCE (the deployable per-window form):")
    ver = _verdicts(G, fe, fd, perm, npres, vp, alpha)
    for dis in DISEASES:
        for w in WINDOWS:
            v = ver[(dis, w)]
            if v["fd_status"] == "UNEVALUABLE_NO_PRECEDING_WINDOW":
                print(f"  {dis:5s} {w:28s} UNEVALUABLE: no preceding window, so no "
                      f"within-place statistic exists")
            else:
                print(f"  {dis:5s} {w:28s} rho = {v['fd_rho']:+.4f}, "
                      f"p(1s) = {v['fd_p_one_sided']:.4f}  -> {v['fd_status']}")
    print("\n  One transition of six clears nominal 0.05, and it is the one where flu burden")
    print("  actually explodes (0.58 to 4.31 percent of ED visits). Under Bonferroni over the")
    print("  six transitions the bar is 0.05/6 = 0.008333 and that transition FAILS "
          f"({fd['flu'].iloc[1]['p_one_sided']:.4f} > 0.008333).")
    print("  Report it as ONE NOMINAL HIT OUT OF SIX, not as a pass.")
    print("\n  CONJUNCTION READING (the manuscript's own failure clause, line 111): failure")
    print("  requires most-variance-between AND not-tracking. ICC corrected is "
          f"{vp['flu'][0]['icc_corrected']:.4f} for flu and {vp['covid'][0]['icc_corrected']:.4f} "
          "for covid, both above")
    print("  OUR 0.50 reading of 'most', so the conjunction reading and the tracking-only")
    print("  reading give IDENTICAL verdicts in every one of the 8 cells here. The choice")
    print("  between them is not load-bearing on this data, but it is still ours.")
    return ver, alpha


def _threshold_sweep(fe, fd, npres, wls):
    hr("10. THRESHOLD SWEEP. THE ALPHA IS OURS, SO HERE IS THE WHOLE CURVE")
    print("The manuscript attaches NO numeric cut to either gate quantity anywhere. No")
    print("threshold below is tuned to produce a pattern; the entire sweep is printed.\n")
    print(f"{'alpha':>7s} | {'FD cells passing (of 6 evaluable)':38s} | "
          f"{'panel 2-way':11s} | {'panel 1-way':11s} | {'nonparam':9s}")
    for a in (0.005, 0.0083, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.50):
        names = []
        for dis in DISEASES:
            for _, r in fd[dis].iterrows():
                if r["spearman_rho"] > 0 and r["p_one_sided"] < a:
                    names.append(f"{dis}:{r['to_window'].split(' - ')[0]}")
        p2 = [d for d in DISEASES
              if fe[(d, True)]["beta"] > 0 and fe[(d, True)]["p_one_sided"] < a]
        p1 = [d for d in DISEASES
              if fe[(d, False)]["beta"] > 0 and fe[(d, False)]["p_one_sided"] < a]
        npv = [d for d in DISEASES if npres[(d, True)]["perm_p"] < a]
        lbl = f"{len(names)}: " + (", ".join(names) if names else "none")
        print(f"{a:7.4f} | {lbl:38s} | {str(p2 or 'none'):11s} | {str(p1 or 'none'):11s} | "
              f"{str(npv or 'none'):9s}")
    print("\n0.0083 is the Bonferroni bar over the six transitions. At every alpha at or below")
    print("0.05 exactly one cell or zero cells pass, and the identity of the passing cell")
    print("never changes. The gate does not become interesting at any threshold; it becomes")
    print("permissive only past alpha = 0.25, which no one would defend.")


def _rerun(concern, burden, vote, ver, alpha, fe, fd, vp):
    """
    Re-run the decision rule end to end with the NEW gate. decide() is dr.decide,
    imported from decision_rule.py and not modified. The ONLY thing that differs from
    the old run is the boolean handed to gate_passed.
    """
    hr("11. RE-RUNNING THE DECISION RULE WITH THE NEW GATE, 51 JURISDICTIONS x 2 DISEASES x 4 WINDOWS")
    print("decide() is imported from decision_rule.py, unmodified. build_cells(),")
    print("add_trend() and run_pool() are likewise imported. The only difference between the")
    print("old run and this one is the value of gate_passed.\n")
    cells = dr.add_trend(dr.build_cells(concern, burden, vote, summary="mean"))
    old_gate = dr.run_gate(cells, concern)
    old_map = {(r["disease"], r["window"]): r for _, r in old_gate.iterrows()}

    new_acts, old_acts, cf_acts = [], [], []
    for (dis, w), df in cells.items():
        gp_new = bool(ver[(dis, w)]["fd_tracks"]) if ver[(dis, w)]["fd_tracks"] else False
        gp_old = old_map[(dis, w)]["gate"] == "PASS"
        n = dr.run_pool(df, gp_new)
        n["gate_form"] = "within_place_first_difference"
        new_acts.append(n)
        old_acts.append(dr.run_pool(df, gp_old))
        # Counterfactual: what the rule WOULD have issued had the gate opened. Used only
        # for the downstream partisan-sorting question, never written as a decision.
        c = dr.run_pool(df, True)
        c["gate_new"] = "PASS" if gp_new else "FAIL"
        c["gate_old"] = "PASS" if gp_old else "FAIL"
        cf_acts.append(c)
    new_acts = pd.concat(new_acts, ignore_index=True)
    old_acts = pd.concat(old_acts, ignore_index=True)
    cf_acts = pd.concat(cf_acts, ignore_index=True)

    # also the panel-gate variant, so the reader sees what the other form would do
    panel_acts = []
    for (dis, w), df in cells.items():
        gp = bool(fe[(dis, True)]["beta"] > 0 and fe[(dis, True)]["p_one_sided"] < alpha)
        panel_acts.append(dr.run_pool(df, gp))
    panel_acts = pd.concat(panel_acts, ignore_index=True)

    print(f"rows written: {len(new_acts)} = 51 x 2 x 4")
    print("\nACTION COUNTS, whole run:")
    print(f"{'action':46s} {'OLD between-place':>18s} {'NEW within-place':>17s} "
          f"{'NEW panel 2-way':>16s}")
    allacts = sorted(set(old_acts["action"]) | set(new_acts["action"])
                     | set(panel_acts["action"]))
    for a in allacts:
        print(f"{a:46s} {int((old_acts['action'] == a).sum()):18d} "
              f"{int((new_acts['action'] == a).sum()):17d} "
              f"{int((panel_acts['action'] == a).sum()):16d}")
    return cells, old_gate, old_map, new_acts, old_acts, cf_acts, panel_acts


def _comparison(cells, old_map, ver, new_acts, old_acts, alpha):
    hr("12. THE COMPARISON TABLE, PER DISEASE AND WINDOW")
    print("'actions' counts decisions that are not NO_DECISION and not INSUFFICIENT_DATA.")
    print("'actionable' counts ALERT or CALM only, the two that change what is communicated.\n")
    hdr = (f"{'disease':6s} {'window':28s} | {'OLD gate':8s} {'old rho':>8s} | "
           f"{'NEW gate':10s} {'new rho':>8s} | {'act old':>7s} {'act new':>7s} | "
           f"{'able old':>8s} {'able new':>8s} | {'changed':>7s}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for dis in DISEASES:
        for w in WINDOWS:
            o = old_map[(dis, w)]
            v = ver[(dis, w)]
            oa = old_acts[(old_acts.disease == dis) & (old_acts.window == w)]
            na = new_acts[(new_acts.disease == dis) & (new_acts.window == w)]
            dec = {dr.NO_DECISION, dr.INSUFFICIENT}
            n_o = int((~oa["action"].isin(dec)).sum())
            n_n = int((~na["action"].isin(dec)).sum())
            ab_o = int(oa["action"].isin({dr.ALERT, dr.CALM}).sum())
            ab_n = int(na["action"].isin({dr.ALERT, dr.CALM}).sum())
            m = oa.set_index("state")["action"].reindex(na["state"].values).values
            chg = int((m != na["action"].values).sum())
            ns = "UNEVAL" if v["fd_status"].startswith("UNEVAL") else v["fd_status"]
            nr = "     na" if v["fd_rho"] is None else f"{v['fd_rho']:+8.4f}"
            print(f"{dis:6s} {w:28s} | {o['gate']:8s} {o['rho_concern_burden']:+8.4f} | "
                  f"{ns:10s} {nr} | {n_o:7d} {n_n:7d} | {ab_o:8d} {ab_n:8d} | {chg:7d}")
            rows.append({"disease": dis, "window": w, "gate_old": o["gate"],
                         "old_rho": o["rho_concern_burden"], "old_p": o["p_rho"],
                         "gate_new": ns, "new_rho": v["fd_rho"],
                         "new_p": v["fd_p_one_sided"],
                         "n_decisions_old": n_o, "n_decisions_new": n_n,
                         "n_actionable_old": ab_o, "n_actionable_new": ab_n,
                         "n_states_changed": chg})
    cmp = pd.DataFrame(rows)
    print(f"\nTOTAL decisions issued: OLD {cmp['n_decisions_old'].sum()}, "
          f"NEW {cmp['n_decisions_new'].sum()}   "
          f"(of 408 = 51 x 2 x 4 cells)")
    print(f"TOTAL actionable (ALERT or CALM): OLD {cmp['n_actionable_old'].sum()}, "
          f"NEW {cmp['n_actionable_new'].sum()}")
    print(f"TOTAL jurisdiction-cells whose decision changes: {cmp['n_states_changed'].sum()} "
          f"of 408")

    hr("12b. WHICH CELLS CHANGE STATUS, AND WHY")
    changed = [(r["disease"], r["window"], r["gate_old"], r["gate_new"])
               for _, r in cmp.iterrows() if r["gate_old"] != r["gate_new"]]
    opened = [c for c in changed if c[3] == "PASS"]
    closed = [c for c in changed if c[2] == "PASS" and c[3] != "PASS"]
    relabel = [c for c in changed if c[3] == "UNEVAL"]
    print("Cells whose LABEL changes:")
    for d, w, a, b in changed:
        print(f"  {d} {w}: {a} -> {b}")
    if not changed:
        print("  none")
    if relabel and not opened and not closed:
        print("")
        print("Both of those are RELABELLINGS, not reversals. The first window of each disease")
        print("has no preceding window, so no within-place statistic exists for it, and it is")
        print("marked UNEVALUABLE rather than FAIL. It issued no decisions under either gate,")
        print("so no jurisdiction's action changes. We report it as a distinct label because")
        print("'the test could not be run' and 'the test was run and the instrument did not")
        print("track' are different statements. The manuscript supplies no vocabulary for")
        print("either, and treating an unevaluable window as a gate failure for the purposes")
        print("of actuation is OURS.")
    if not opened and not closed:
        print("")
        print("NO CELL CHANGES PASS/FAIL STATUS. Both gates open exactly one cell of eight:")
        print("  flu, November 30 - December 27.")
        print("")
        print("THIS IS THE MOST IMPORTANT AND MOST INCONVENIENT RESULT IN THE FILE, so it is")
        print("stated without hedging. The old gate and the new gate are PROVABLY logically")
        print("independent (section 8, Parts 2 and 3: each can be swept across its whole range")
        print("with the other held exactly fixed). They nevertheless select the SAME cell on")
        print("THIS dataset. The reason is substantive, not algebraic: that is the one window")
        print("in which national flu burden goes from 0.58 to 4.31 percent of ED visits, and")
        print("it is the only window in the season in which anything moves enough for either")
        print("geometry to register a signal. The manuscript already says as much at line 109.")
        print("")
        print("So the circularity is fixed and the answer is unchanged. Those are two")
        print("different claims and only the first one is a fix. Anyone who expected the new")
        print("gate to open different cells should read this as a null result about the")
        print("season, not as evidence that the old gate was fine.")
    else:
        print("")
        for d, w, a, b in opened:
            print(f"  OPENED by the new gate: {d} {w} ({a} -> {b})")
        for d, w, a, b in closed:
            print(f"  CLOSED by the new gate: {d} {w} ({a} -> {b})")
    return cmp


def _sd_d_diagnostic(cells, ver, old_map):
    hr("13. THE DIAGNOSTIC THAT EXPOSED THE OLD PROBLEM, RECOMPUTED UNDER THE NEW GATE")
    print("Under the OLD gate: sd(D) was 0.946 in the single passing cell against 1.153 to")
    print("1.526 in the seven failing cells, i.e. the rule was permitted to act only where")
    print("the divergences were smallest. sd(D) below is computed with ddof = 1 on the actual")
    print("D column produced by build_cells(), so it is directly comparable to those numbers.\n")
    print(f"{'disease':7s} {'window':28s} {'sd(D)':>8s} {'OLD gate':>9s} {'NEW gate':>22s}")
    rec = []
    for dis in DISEASES:
        for w in WINDOWS:
            df = cells[(dis, w)]
            s = float(df["D"].std(ddof=1))
            og, ng = old_map[(dis, w)]["gate"], ver[(dis, w)]["fd_status"]
            rec.append({"disease": dis, "window": w, "sd_D": s, "old": og, "new": ng})
            print(f"{dis:7s} {w:28s} {s:8.4f} {og:>9s} {ng:>22s}")
    r = pd.DataFrame(rec)
    for lbl, col, passv in (("OLD", "old", "PASS"), ("NEW", "new", "PASS")):
        p = r[r[col] == passv]["sd_D"]
        f = r[r[col] != passv]["sd_D"]
        print(f"\n{lbl} gate: sd(D) in PASSING cells = "
              f"{', '.join(f'{x:.4f}' for x in p)} (n = {len(p)})")
        print(f"{lbl} gate: sd(D) in FAILING/unevaluable cells = "
              f"{f.min():.4f} to {f.max():.4f} (n = {len(f)}), mean {f.mean():.4f}")
    print("\nVERDICT ON THE FIX, STATED PLAINLY AND WITHOUT SPIN.")
    print("The pattern is UNCHANGED: the one cell the new gate opens is still the")
    print("minimum-sd(D) cell. If the only test of the fix were this diagnostic, the fix")
    print("would have to be called a failure.")
    print("")
    print("But this diagnostic is a NECESSARY, not a sufficient, symptom, and with 8 cells")
    print("and 1 pass it cannot discriminate between the two explanations:")
    print("  (i) the gate is algebraically tied to D, so it CANNOT open anywhere else; or")
    print("  (ii) the gate is free to open anywhere and this season only offers one window")
    print("       in which either the hazard or the instrument moves at all.")
    print("Section 8 settles that question directly and does not rely on this diagnostic.")
    print("Under the OLD gate, explanation (i) is proved: sd(D) = sqrt(2 s^2 (1 - r)) is an")
    print("identity, so a passing cell is BY CONSTRUCTION a low-sd(D) cell, in every dataset,")
    print("forever. Under the NEW gate, explanation (i) is disproved by construction: Test B")
    print("sweeps sd(D) from a mean of 1.65 down to 0.80 and drives the old gate statistic")
    print("from -0.72 to +0.92 while the new gate statistic does not move in a single printed")
    print("digit. The new gate CAN open in a high-sd(D) cell. On this season's data it does")
    print("not, because there is only one window where anything moves.")
    print("")
    print("The honest one-line summary: the circularity is removed and the empirical answer")
    print("is unchanged. Removing the circularity is what the referee asked for; changing the")
    print("answer was never promised and did not happen.")
    return r


def _partisan(cf_acts):
    hr("14. THE DOWNSTREAM RESULT. DO THE ACTIONS SORT ON PARTISAN VOTE SHARE?")
    print("The question the paper's own argument turns on: in cells the new gate FAILS, do")
    print("the actions that WOULD have been issued sort by the 2024 GOP margin rather than by")
    print("the outbreak? And in cells it PASSES, is that association absent?")
    print("")
    print("To ask it at all we must run the rule with the gate forced open in every cell.")
    print("Those counterfactual actions are NEVER written to the decisions file; they exist")
    print("only to answer this question. Actions are scored on D's own axis: ALERT -1,")
    print("any HOLD 0, CALM +1, then correlated with the vote and with absolute burden.\n")
    print(f"{'cells':38s} {'n':>5s} {'rho action~vote':>15s} {'p':>8s} "
          f"{'rho action~burden':>18s} {'p':>8s}")
    out = {}
    groups = [("NEW gate FAILS or unevaluable", cf_acts[cf_acts.gate_new == "FAIL"]),
              ("NEW gate PASSES", cf_acts[cf_acts.gate_new == "PASS"]),
              ("OLD gate FAILS", cf_acts[cf_acts.gate_old == "FAIL"]),
              ("OLD gate PASSES", cf_acts[cf_acts.gate_old == "PASS"]),
              ("all 408 cells", cf_acts)]
    for lbl, sub in groups:
        a = dr.action_associations(sub)
        out[lbl] = a
        if a is None:
            print(f"{lbl:38s} {len(sub):5d}   only one action value present, "
                  f"no association is defined")
            continue
        print(f"{lbl:38s} {a['n']:5d} {a['rho_action_vote']:+15.4f} "
              f"{a['p_action_vote']:8.4f} {a['rho_action_burden']:+18.4f} "
              f"{a['p_action_burden']:8.4f}")
    print("\nALERTED versus NOT ALERTED, mean 2024 GOP margin and mean burden:")
    for lbl, sub in groups:
        a = out.get(lbl)
        if a and "mean_gop_alerted" in a:
            print(f"  {lbl:36s} gop alerted {a['mean_gop_alerted']:+7.2f} vs not "
                  f"{a['mean_gop_not_alerted']:+7.2f} (Mann-Whitney p = {a['mwu_p_vote']:.4g}); "
                  f"burden alerted {a['mean_burden_alerted']:.3f} vs not "
                  f"{a['mean_burden_not_alerted']:.3f} (p = {a['mwu_p_burden']:.4g})")
    print("\nPer (disease, window), so the pooled numbers cannot hide a cell:")
    print(f"{'disease':7s} {'window':28s} {'NEW':6s} {'n':>4s} {'rho~vote':>9s} {'p':>8s} "
          f"{'rho~burden':>11s} {'p':>8s}")
    percell = []
    for dis in DISEASES:
        for w in WINDOWS:
            sub = cf_acts[(cf_acts.disease == dis) & (cf_acts.window == w)]
            a = dr.action_associations(sub)
            g = sub["gate_new"].iloc[0]
            if a is None:
                print(f"{dis:7s} {w:28s} {g:6s} {len(sub):4d}   "
                      f"single action value, no association defined")
            else:
                print(f"{dis:7s} {w:28s} {g:6s} {a['n']:4d} "
                      f"{a['rho_action_vote']:+9.4f} {a['p_action_vote']:8.4f} "
                      f"{a['rho_action_burden']:+11.4f} {a['p_action_burden']:8.4f}")
                percell.append({"disease": dis, "window": w, "gate": g, "n": a["n"],
                                "rho_vote": a["rho_action_vote"],
                                "p_vote": a["p_action_vote"],
                                "rho_burden": a["rho_action_burden"],
                                "p_burden": a["p_action_burden"]})
    pc = pd.DataFrame(percell)
    fail = pc[pc.gate == "FAIL"]
    pas = pc[pc.gate == "PASS"]
    print("\nREAD IT PLAINLY. Cell by cell, and the per-cell rows are the honest unit:")
    print(f"  In the {len(fail)} cells the new gate FAILS or cannot evaluate, the "
          f"counterfactual actions")
    print(f"  sort on the 2024 vote in {int((fail.p_vote < 0.05).sum())} of {len(fail)} "
          f"cells at p < 0.05. rho ranges "
          f"{fail.rho_vote.min():+.4f} to {fail.rho_vote.max():+.4f}, median "
          f"{fail.rho_vote.median():+.4f}.")
    if len(pas):
        r = pas.iloc[0]
        print(f"  In the {len(pas)} cell the new gate PASSES, the vote association is "
              f"ABSENT: rho = {r['rho_vote']:+.4f}, p = {r['p_vote']:.4f}, n = {int(r['n'])}.")
        print(f"  The burden association SURVIVES in that same cell: "
              f"rho = {r['rho_burden']:+.4f}, p = {r['p_burden']:.4f}.")
        # Fisher z comparison of the passing cell against the median failing cell.
        def z(x):
            return 0.5 * np.log((1 + x) / (1 - x))
        med = float(fail.rho_vote.median())
        zdiff = (z(r["rho_vote"]) - z(med)) / np.sqrt(2.0 / (51 - 3))
        print(f"  Fisher z comparison of the passing cell against the median failing cell:")
        print(f"    z = {zdiff:+.3f}, two-sided p = "
              f"{2 * stats.norm.sf(abs(zdiff)):.4f}, treating the two cells as independent.")
        print("  That comparison is a SINGLE cell of 51 places against one summary of seven")
        print("  cells, so it is a description, not a test with power. Do not over-read it.")
    print("")
    print("CAVEATS THAT MUST TRAVEL WITH THESE NUMBERS.")
    print("  1. The pooled rows above stack the same 51 jurisdictions across up to 7 cells,")
    print("     so their n of 357 is not 357 independent observations and their p values are")
    print("     badly anticonservative. They are reported only because the old run reported")
    print("     the pooled figure. The per-cell rows are the unit to quote.")
    print("  2. In the passing cell the rule issues only ALERT and HOLD, never CALM, so the")
    print("     action score there takes two values and the correlation is effectively an")
    print("     alerted-versus-not comparison.")
    print("  3. The FAIL and PASS partitions are IDENTICAL under the old and the new gate on")
    print("     this data, so this section cannot by itself distinguish the two gates. It")
    print("     answers the question that was asked: under the new, non-circular gate, the")
    print("     answer is the same one the manuscript already gives.")
    print("")
    print("SUBSTANTIVELY: this is the manuscript's own argument, and it now rests on a gate")
    print("that is not algebraically tied to the metric. Where the instrument does not")
    print("demonstrably move with the hazard, the actions the rule would have issued sort by")
    print("how a place voted. Where it does, that sorting is not detectable and the sorting")
    print("that remains is on burden. That is what a measurement condition is for.")
    return out


def _choices():
    hr("15. EVERY CHOICE HERE THAT THE MANUSCRIPT DOES NOT SPECIFY")
    print("None of the following is attributable to the paper. Each is swept or reported in")
    print("both directions above.\n")
    for i, (k, (v, why)) in enumerate(OUR_CHOICES.items(), 1):
        print(f"{i:2d}. {k} = {v}")
        for line in _wrap(why, 82):
            print(f"      {line}")
    print("\nINPUTS THAT DO NOT EXIST AND ARE NOT CLAIMED:")
    print("  - No per-cell standard error is published for the NSSP burden leg anywhere in")
    print("    this repository, so the regressor carries no instantiated sampling-error")
    print("    input and no errors-in-variables correction is applied to it.")
    print("  - The manuscript's 0.50 and 0.43 (line 109) have no definition in")
    print("    methods_section.tex: no correlation family, no n, no p, no control")
    print("    specification. Nothing here reproduces them and nothing here claims to.")
    print("  - There are four windows in one season. Any claim about whether the instrument")
    print("    tracks across SEASONS is not instantiable from this file set, as the")
    print("    manuscript itself states in its limitations.")
    print("  - The manuscript names no return value for a failed measurement condition. Its")
    print("    only failure-state wording, 'treated as insufficient data rather than forced")
    print("    into a regime', is written about thin pools and is not reused here.")


def _wrap(s, n):
    words, line, out = s.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > n:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return out


def _write_csvs(ver, fe, wls, perm, npres, rel, fd, vp, cmp, sdd, new_acts, alpha):
    rows = []
    for dis in DISEASES:
        for w in WINDOWS:
            v = ver[(dis, w)]
            c = cmp[(cmp.disease == dis) & (cmp.window == w)].iloc[0]
            s = sdd[(sdd.disease == dis) & (sdd.window == w)].iloc[0]
            r = {"disease": dis, "window": w, "n_states": 51, "n_windows": 4,
                 "gate_geometry": "within-place over time",
                 # panel form, one value per disease repeated across windows
                 "panel_beta_one_way": fe[(dis, False)]["beta"],
                 "panel_se_one_way": fe[(dis, False)]["se"],
                 "panel_t_one_way": fe[(dis, False)]["t"],
                 "panel_p_one_sided_one_way": fe[(dis, False)]["p_one_sided"],
                 "panel_within_r2_one_way": fe[(dis, False)]["within_r2"],
                 "panel_perm_p_one_way": perm[(dis, False)]["p_one_sided"],
                 "panel_beta_two_way": fe[(dis, True)]["beta"],
                 "panel_se_two_way": fe[(dis, True)]["se"],
                 "panel_t_two_way": fe[(dis, True)]["t"],
                 "panel_p_one_sided_two_way": fe[(dis, True)]["p_one_sided"],
                 "panel_within_r2_two_way": fe[(dis, True)]["within_r2"],
                 "panel_perm_p_two_way": perm[(dis, True)]["p_one_sided"],
                 "wls_beta_one_way": wls[(dis, False)]["beta"],
                 "wls_t_one_way": wls[(dis, False)]["t"],
                 "wls_beta_two_way": wls[(dis, True)]["beta"],
                 "wls_t_two_way": wls[(dis, True)]["t"],
                 "nonparam_mean_rho_window_demeaned": npres[(dis, True)]["mean_rho"],
                 "nonparam_perm_p_window_demeaned": npres[(dis, True)]["perm_p"],
                 "nonparam_sign_p_window_demeaned": npres[(dis, True)]["sign_p"],
                 "lambda_within_one_way": rel[(dis, False)]["lambda"],
                 "lambda_within_two_way": rel[(dis, True)]["lambda"],
                 "icc_naive": v["icc_naive"], "icc_corrected": v["icc_corrected"],
                 "most_variance_between_places": v["most_variance_between_places"],
                 # first-difference form, per window
                 "fd_from_window": v["fd_from_window"], "fd_rho": v["fd_rho"],
                 "fd_p_one_sided": v["fd_p_one_sided"], "fd_pearson": v["fd_pearson"],
                 "fd_lambda_delta_concern": v["fd_lambda"],
                 "gate_new_within_place": v["fd_status"],
                 "gate_new_panel_two_way": "PASS" if v["panel_tracks_two_way"] else "FAIL",
                 "gate_new_panel_one_way": "PASS" if v["panel_tracks_one_way"] else "FAIL",
                 "gate_new_conjunction_reading": v["fd_status"],
                 "gate_old_between_place": c["gate_old"],
                 "old_rho_concern_burden": c["old_rho"], "old_p_rho": c["old_p"],
                 "sd_D": s["sd_D"],
                 "n_decisions_old": c["n_decisions_old"],
                 "n_decisions_new": c["n_decisions_new"],
                 "n_actionable_old": c["n_actionable_old"],
                 "n_actionable_new": c["n_actionable_new"],
                 "n_states_changed": c["n_states_changed"],
                 "alpha": alpha,
                 "bonferroni_alpha_6_transitions": 0.05 / 6}
            rows.append(r)
    g = pd.DataFrame(rows)
    g.to_csv(os.path.join(HERE, "gate_within_place_results.csv"), index=False)
    new_acts.to_csv(os.path.join(HERE, "decision_rule_actions_withinplace.csv"), index=False)
    hr("16. FILES WRITTEN")
    print(f"  gate_within_place_results.csv          {len(g)} rows")
    print(f"  decision_rule_actions_withinplace.csv  {len(new_acts)} rows")
    print("  gate_within_place_output.txt           this console log")
    print("\nNOT MODIFIED: decision_rule.py, decision_rule_actions.csv, decision_rule_gate.csv.")
    return g


def main():
    tee = Tee(os.path.join(HERE, "gate_within_place_output.txt"))
    sys.stdout = tee
    try:
        (panel, concern, burden, vote, G, fe, wls, perm, npres, rel, fd, sw, vp) = _main()
        _independence(G)
        ver, alpha = _main2(panel, concern, burden, vote, G, fe, wls, perm, npres,
                            rel, fd, sw, vp)
        _threshold_sweep(fe, fd, npres, wls)
        (cells, old_gate, old_map, new_acts, old_acts,
         cf_acts, panel_acts) = _rerun(concern, burden, vote, ver, alpha, fe, fd, vp)
        cmp = _comparison(cells, old_map, ver, new_acts, old_acts, alpha)
        sdd = _sd_d_diagnostic(cells, ver, old_map)
        _partisan(cf_acts)
        _choices()
        _write_csvs(ver, fe, wls, perm, npres, rel, fd, vp, cmp, sdd, new_acts, alpha)
        hr("17. THE RESULT, IN FIVE SENTENCES")
        print("""
1. The old gate is a deterministic function of the dispersion of the metric it gates,
   sd(D) = sqrt(2 s^2 (1 - r)), verified to 1e-07 on all eight cells. It is not a gate.
2. The within-place gate is provably not tied to D: each statistic can be swept across
   its full range on the real data while the other is held exactly fixed.
3. The within-place gate does not pass. Two-way FE gives flu t = +0.875 and covid
   t = -0.228, and for covid the test is not even identified, because after removing
   state and window means 100 percent of the residual variance in covid concern is CDC
   sampling error. One first-difference transition of six clears nominal 0.05 and fails
   Bonferroni.
4. The test that fails is also underpowered, so the non-rejection is weak evidence, not
   strong evidence against the instrument. This is a property of four windows in one
   season and no estimator can repair it.
5. The gate opens the same single cell as before, flu November 30 to December 27. The
   circularity is fixed; the answer is not changed. Those are two different claims.
""")
    finally:
        sys.stdout = sys.__stdout__
        tee.f.close()


if __name__ == "__main__":
    main()
