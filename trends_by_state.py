#!/usr/bin/env python3
"""
GO/NO-GO TEST for the divergence metric D.

Question: does flu search interest vary enough ACROSS US STATES to support a
rank-based pool? D = rank(perceived) - rank(actual) needs real variation on
both sides. If every state looks the same, the ranks are noise and D is dead
before we collect a single Reddit post.

This pulls the ATTENTION side only (Google Trends). It is NOT perceived risk.
Perceived risk needs text (Reddit appraisal per post). Keep them separate.

Run:  python3 trends_by_state.py
Note: Google throttles pytrends (429). If it fails, use the manual export
described at the bottom.
"""
import sys, time
import numpy as np

try:
    from pytrends.request import TrendReq
except ImportError:
    sys.exit("pip install pytrends")

TERM = "flu"
# WINDOW MUST MATCH THE PERCEIVED LEG. This is not a detail; it changed the answer.
# CDC publishes flu concern for 2025-12-28 to 2026-01-31, and that window is fixed
# by CDC, so every other leg has to come to it.
# An earlier version of this file used "2025-12-01 2026-02-28" (13 weeks) against a
# 5-week concern window. That mismatch made perceived risk and attention look
# UNRELATED (rho +0.164, p=0.25). On the matched window they are related
# (rho +0.372, p=0.007). The separability claim depends on getting this right.
TIMEFRAME = "2025-12-28 2026-01-31"
GEO = "US"

def verdict(s):
    """Is there enough spread across states for rank scoring to mean anything?"""
    n = len(s)
    lo, hi = s.min(), s.max()
    cv = s.std() / s.mean() if s.mean() else 0
    iqr = np.percentile(s, 75) - np.percentile(s, 25)
    ties = n - len(set(s.round(0)))
    print(f"\n  states with data : {n}")
    print(f"  range            : {lo:.0f} to {hi:.0f}")
    print(f"  median           : {np.median(s):.0f}")
    print(f"  IQR              : {iqr:.0f}")
    print(f"  coeff. variation : {cv:.3f}")
    print(f"  approx ties      : {ties}")
    print("\n  VERDICT:")
    if n < 15:
        print("   NO-GO: too few states with data. The pool cannot fill.")
    elif cv < 0.05 or iqr < 3:
        print("   NO-GO: states are nearly identical. Ranks would be noise, not signal.")
    elif ties > n * 0.5:
        print("   WEAK: over half the states tie. Rank scoring will be coarse.")
    else:
        print("   GO: real spread across states. A rank pool is viable on the attention side.")
        print("   Next: get the ACTUAL side (CDC FluView, state ILI) and the")
        print("   PERCEIVED side (Reddit appraisal per post). Only then is it D.")

def main():
    print(f"Google Trends: '{TERM}', {TIMEFRAME}, by US state")
    for attempt in range(4):
        try:
            pt = TrendReq(hl="en-US", tz=360)  # no retries=/backoff_factor=: pytrends passes
            # method_whitelist to urllib3 Retry, which urllib3 v2 renamed to allowed_methods
            pt.build_payload([TERM], timeframe=TIMEFRAME, geo=GEO)
            df = pt.interest_by_region(resolution="REGION", inc_low_vol=True)
            if df is None or df.empty:
                print("  empty result, retrying..."); time.sleep(15); continue
            df = df[df[TERM] > 0].sort_values(TERM, ascending=False)
            print("\n  top 8 states:")
            print(df.head(8).to_string())
            print("\n  bottom 5 states:")
            print(df.tail(5).to_string())
            verdict(df[TERM].values.astype(float))
            df.to_csv("flu_trends_by_state.csv")
            print("\n  saved -> flu_trends_by_state.csv")
            return
        except Exception as e:
            print(f"  attempt {attempt+1} failed: {repr(e)[:150]}")
            time.sleep(20 * (attempt + 1))
    print("\nCould not fetch automatically.")
    print("MANUAL PATH: trends.google.com -> search 'flu' -> United States ->")
    print(f"custom range {TIMEFRAME} -> 'Interest by subregion' -> download CSV ->")
    print("save here as flu_trends_by_state.csv, then rerun this script with --from-csv")

if __name__ == "__main__":
    if "--from-csv" in sys.argv:
        import pandas as pd
        d = pd.read_csv("flu_trends_by_state.csv")
        col = d.columns[1]
        s = pd.to_numeric(d[col].astype(str).str.replace("%", ""), errors="coerce").dropna()
        verdict(s.values.astype(float))
    else:
        main()
