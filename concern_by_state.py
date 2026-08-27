#!/usr/bin/env python3
"""
The PERCEIVED-RISK leg for D: CDC's own state-level survey of flu concern.

THIS REPLACES THE REDDIT ROUTE. Why: the Reddit API cannot deliver this. Over an
entire flu season r/nyc returned 9 flu posts, Montana returned zero, and search
caps at ~45-72 results. That was measured, not assumed (see collect_reddit_geo.py).
The failure is real and it is worth reporting, but it is not a reason to stall the
paper when CDC has been asking people the question directly the whole time.

WHAT THIS IS
  CDC's NIS-FRVM asks adults whether they are concerned about flu disease, and
  publishes the result BY STATE with a confidence interval and a sample size.
  That is a survey proportion: worried respondents over respondents. There is no
  count anywhere in it.

WHY THAT MATTERS (the separability argument, which is the whole paper)
  If 'perceived risk' were measured as HOW MUCH people post or search, it would be
  a volume measure, and D would collapse into attention-minus-actual. That is
  Gallotti's infodemic volume, already published, and the thing this paper argues
  against. A survey percentage cannot collapse that way even in principle: the
  denominator is respondents, not posts.
  And it holds empirically, not just by construction:
      Spearman(concern, Google Trends flu) = +0.164, p = 0.249, n = 51
  Not significant. Perceived risk is NOT attention in disguise. Measured, 2026-07-15.

THE THING YOU MUST NOT SKIP
  Flu concern correlates with 2024 partisan lean at roughly rho -0.69 (blue states
  worry more), which is far stronger than its relationship to actual flu burden
  (-0.27). So this variable is substantially political identity. That is not a
  footnote to bury; see compute_d.py, which tests it head-on.

Run:  python3 concern_by_state.py
      python3 concern_by_state.py --all-windows    (all 4 months, not just the peak)

Source: https://data.cdc.gov/resource/ee83-ukst.json  (free, public, no key)
Note: python urllib hits SSL: CERTIFICATE_VERIFY_FAILED against data.cdc.gov on
this Mac, so everything here goes through curl. Same trick as fluview_by_state.py.
"""
import csv, json, subprocess, sys
import numpy as np

RESOURCE = "https://data.cdc.gov/resource/ee83-ukst.json"
INDICATOR = "Concerned about flu disease"
GROUP = "All adults 18+ years"

# The 4 windows CDC publishes. The last one covers the flu peak, and it is the one
# that lines up with the attention and actual legs (Dec 2025 - Feb 2026).
PEAK_WINDOW = "December 28 - January 31"
ALL_WINDOWS = ["October 1 - October 25", "October 26 - November 29",
               "November 30 - December 27", "December 28 - January 31"]

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


def fetch(window=None):
    """One curl. The $where filter does the work server-side."""
    where = (f"dsss_indicator_category_label='{INDICATOR}' "
             f"AND geographic_level='State' "
             f"AND dsss_group_variable_name='{GROUP}'")
    if window:
        where += f" AND dsss_timeperiodlabel='{window}'"
    cmd = ["curl", "-s", "--get", RESOURCE,
           "--data-urlencode", f"$where={where}",
           "--data-urlencode", "$limit=2000"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    try:
        return json.loads(r.stdout)
    except Exception:
        print("  raw response:", r.stdout[:300])
        sys.exit("  could not parse CDC response")


def parse_ci(s):
    """CDC publishes '25.1 - 32.2'. Return (lo, hi) so compute_d.py can propagate it."""
    try:
        lo, hi = [float(x.strip()) for x in str(s).split("-")]
        return lo, hi
    except Exception:
        return None, None


def verdict(vals, label):
    v = np.asarray(vals, dtype=float)
    n = len(v)
    cv = v.std() / v.mean() if v.mean() else 0
    iqr = np.percentile(v, 75) - np.percentile(v, 25)
    print(f"\n  {label}")
    print(f"    states     : {n}")
    print(f"    range      : {v.min():.1f}% to {v.max():.1f}%")
    print(f"    median     : {np.median(v):.1f}%")
    print(f"    IQR        : {iqr:.1f}")
    print(f"    coeff var  : {cv:.3f}")
    if n < 15:
        print("    NO-GO: too few states."); return False
    if cv < 0.05:
        print("    NO-GO: states nearly identical; ranks would be noise."); return False
    print("    GO: real spread across states.")
    return True


def main():
    every = "--all-windows" in sys.argv
    windows = ALL_WINDOWS if every else [PEAK_WINDOW]
    print(f"CDC NIS-FRVM '{INDICATOR}', {GROUP}")
    print(f"windows: {', '.join(windows)}\n")

    rows_out = []
    for w in windows:
        d = fetch(w)
        print(f"  {w:32} -> {len(d):3} rows")
        for r in d:
            st = ABBR.get(str(r.get("geographic_label", "")).strip().lower())
            if not st:
                continue
            lo, hi = parse_ci(r.get("dsss_confidenceinterval"))
            rows_out.append({
                "state": st,
                "window": w,
                "concern_pct": float(r["dsss_value"]) if r.get("dsss_value") else None,
                "ci_low": lo, "ci_high": hi,
                "sample_size": int(r["sample_size"]) if r.get("sample_size") else None,
                "suppressed": r.get("suppressionflag", "0"),
            })

    kept = [r for r in rows_out if r["concern_pct"] is not None
            and str(r["suppressed"]) == "0"]
    supp = len(rows_out) - len(kept)
    if supp:
        print(f"\n  dropped {supp} suppressed/blank rows")

    out = "flu_concern_by_state.csv"
    with open(out, "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=["state", "window", "concern_pct",
                                           "ci_low", "ci_high", "sample_size", "suppressed"])
        w_.writeheader(); w_.writerows(kept)

    peak = [r for r in kept if r["window"] == PEAK_WINDOW]
    if peak:
        ranked = sorted(peak, key=lambda r: -r["concern_pct"])
        print("\n  most concerned about flu:")
        for r in ranked[:8]:
            print(f"    {r['state'].upper():3} {r['concern_pct']:5.1f}%  "
                  f"(CI {r['ci_low']}-{r['ci_high']}, n={r['sample_size']})")
        print("  least concerned:")
        for r in ranked[-5:]:
            print(f"    {r['state'].upper():3} {r['concern_pct']:5.1f}%  "
                  f"(CI {r['ci_low']}-{r['ci_high']}, n={r['sample_size']})")
        verdict([r["concern_pct"] for r in peak], "PERCEIVED-RISK side (% concerned about flu)")

        ns = [r["sample_size"] for r in peak if r["sample_size"]]
        print(f"\n  sample sizes: {min(ns)} to {max(ns)} (median {int(np.median(ns))})")
        print(f"  NOTE: sample sizes are very unequal, so the state with n={min(ns)} is far")
        print(f"  less precise than the one with n={max(ns)}. compute_d.py bootstraps CDC's")
        print("  published CIs for exactly this reason. Do not read the middle of the ranking.")

    print(f"\n  saved -> {out}  ({len(kept)} rows)")
    print("\n  ALL THREE LEGS NOW EXIST:")
    print("    attention  = flu_trends_by_state.csv    Google Trends      51 states")
    print("    actual     = flu_ed_by_state.csv        CDC NSSP           51 states")
    print("    perceived  = flu_concern_by_state.csv   CDC NIS-FRVM       51 states")
    print("\n  NEXT: python3 compute_d.py")


if __name__ == "__main__":
    main()
