#!/usr/bin/env python3
"""
The specification curve, computed live. This is the honesty check on the paper's
core empirical claim.

THE PROBLEM IT EXISTS TO EXPOSE
  "Worry does not track flu" (rho -0.268) reads like a finding. It is not robust.
  The actual-risk leg involves two arbitrary choices - which weeks, and mean vs peak
  - and across defensible combinations of those choices the worry-burden correlation
  runs from about -0.37 to +0.40, with BOTH ENDS significant at p < 0.01. The same
  data supports opposite conclusions depending on forks nobody would think to
  pre-register.

  That is the garden of forking paths (Gelman & Loken). It is not misconduct and it
  does not require anyone to have cheated. It requires only that each individual
  choice be defensible, which they all are. The defence is to pre-specify one primary
  spec, justify it in the open, and publish the whole curve.

PRIMARY SPEC (pre-specified, justified, and NOT chosen because it gave the best result)
  window  = matched to the perceived leg. CDC fixes concern at Dec 28 - Jan 31, so
            the actual leg must come to it. Comparing January worry against February
            flu measures a lag no one theorised.
  summary = mean. Peak is a single week; one outlier week swings a state's whole
            rank. Mean is the conservative choice.

WHAT TO WATCH FOR
  mean is consistently NEGATIVE, peak is consistently POSITIVE, in every window.
  That is a coherent, testable hypothesis - worry may track how bad flu GETS rather
  than how bad it typically IS - and it is a better paper than a fragile -0.268.
  It is also not what the manuscript currently argues. Do not harvest it silently.

Run:  python3 spec_curve_live.py
Note: Delphi rate-limits anonymous queries hard. Free key at
      api.delphi.cmu.edu/epidata/admin/registration_form -> export DELPHI_API_KEY
"""
import json
import os
import subprocess
import sys
import time

import pandas as pd
from scipy import stats

SPECS = [
    ("202601-202605", "MATCHED (Dec 28 - Jan 31)", True),
    ("202548-202609", "Dec 1 - Feb 28", False),
    ("202552-202605", "Dec 21 - Jan 31", False),
    ("202601-202609", "Jan 1 - Feb 28 (lagged)", False),
]


def get(weeks, tries=5):
    key = os.environ.get("DELPHI_API_KEY", "").strip()
    for i in range(tries):
        cmd = ["curl", "-s", "--get", "https://api.delphi.cmu.edu/epidata/covidcast/",
               "--data-urlencode", "data_source=nssp",
               "--data-urlencode", "signal=pct_ed_visits_influenza",
               "--data-urlencode", "time_type=week",
               "--data-urlencode", "geo_type=state",
               "--data-urlencode", f"time_values={weeks}",
               "--data-urlencode", "geo_value=*"]
        if key:
            cmd += ["--data-urlencode", f"api_key={key}"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        try:
            d = json.loads(r.stdout)
            if d.get("result") == 1:
                x = pd.DataFrame(d["epidata"])
                return x.groupby("geo_value")["value"].agg(["mean", "max"])
        except Exception:
            if "429" in r.stdout or "Too Many" in r.stdout:
                w = 20 * (i + 1)
                print(f"    rate-limited; waiting {w}s"); time.sleep(w); continue
            break
    return None


def main():
    if not os.path.exists("flu_concern_by_state.csv"):
        sys.exit("run concern_by_state.py first")
    c = pd.read_csv("flu_concern_by_state.csv")
    c = c[c["window"] == "December 28 - January 31"][["state", "concern_pct"]].set_index("state")

    print("SPECIFICATION CURVE: perceived risk vs actual risk")
    print("The perceived leg is FIXED by CDC (Dec 28 - Jan 31) and cannot move.")
    print("Every free choice below lives in the actual leg.\n")
    print(f"  {'actual-leg window':30} {'summary':8} {'rho':>8} {'p':>8}  {'n':>3}")
    print("  " + "-" * 64)

    out, blocked = [], False
    for weeks, label, primary in SPECS:
        g = get(weeks)
        if g is None:
            print(f"  {label:30} {'--':8} {'BLOCKED':>8}")
            blocked = True
            continue
        for summ in ("mean", "max"):
            j = c.join(g[summ].rename("actual"), how="inner").dropna()
            r, p = stats.spearmanr(j["concern_pct"], j["actual"])
            star = " *" if p < 0.05 else "  "
            tag = "   <- PRIMARY" if (primary and summ == "mean") else ""
            print(f"  {label:30} {summ:8} {r:+8.3f} {p:8.3f}{star}{len(j):3}{tag}")
            out.append({"window": weeks, "label": label, "summary": summ,
                        "rho": round(r, 4), "p": round(p, 5), "n": len(j),
                        "primary": bool(primary and summ == "mean")})
        time.sleep(2)

    if not out:
        sys.exit("\n  everything blocked; get a free DELPHI_API_KEY and retry")

    df = pd.DataFrame(out)
    df.to_csv("spec_curve.csv", index=False)
    lo, hi = df["rho"].min(), df["rho"].max()
    nsig = int((df["p"] < 0.05).sum())

    print("\n  * p < 0.05")
    print(f"\n  RANGE: rho from {lo:+.3f} to {hi:+.3f} across {len(df)} specifications")
    print(f"  {nsig} of {len(df)} reach p < 0.05")
    if lo < 0 and hi > 0:
        print("\n  THE SIGN IS NOT STABLE.")
        print("  Do NOT write 'worry does not track flu' as an established finding.")
        print("  Write: across these specifications the correlation ranges from")
        print(f"  {lo:+.2f} to {hi:+.2f}, so the relationship is not robustly signed.")
        print("  Report the primary spec WITH this curve beside it.")
    else:
        print("\n  The sign is stable across specifications. That is worth stating.")

    mean_neg = (df[df["summary"] == "mean"]["rho"] < 0).all()
    max_pos = (df[df["summary"] == "max"]["rho"] > 0).all()
    if mean_neg and max_pos:
        print("\n  THE PATTERN HOLDS: mean negative everywhere, peak positive everywhere.")
        print("  Hypothesis: worry tracks how bad flu GETS, not how bad it usually is.")
        print("  Testable, interesting, and NOT what the paper currently claims.")

    if blocked:
        print("\n  NOTE: some specs were rate-limited and are missing from the curve above.")
    print("\n  saved -> spec_curve.csv")


if __name__ == "__main__":
    main()
