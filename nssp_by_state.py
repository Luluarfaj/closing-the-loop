#!/usr/bin/env python3
"""
The ACTUAL-RISK leg for D: CDC NSSP emergency-department visits for influenza.

THIS REPLACES fluview_by_state.py AS THE PRIMARY ACTUAL LEG, BECAUSE FLUVIEW IS
MISSING EIGHT STATES THIS SEASON.

THE SYMPTOM, IN ONE COMMAND
    curl "https://api.delphi.cmu.edu/epidata/fluview/?regions=ny,ct,ak,hi,ca,tx&epiweeks=202601-202605"
    -> result: 1, message: "success", and ONLY ca and tx come back.

FluView reports success and returns nothing. fluview_by_state.py quietly produced 43
states instead of 51, dropping ak, ct, hi, ny, ok, or, sd, ut. Nothing errored. The
CSV just had 43 rows and looked fine.

THE ACTUAL CAUSE - AND IT IS NOT A CODE BUG
  An earlier version of this file said 'ny is not a valid region code'. That was
  WRONG. The real cause is data governance, verified against CDC's own endpoint:
      curl "https://gis.cdc.gov/grasp/flu2/GetPhase02InitApp?appVersion=Public"
  Its state_data_approval table shows which states authorized CDC to PUBLISH their
  weekly ILI data, per season:
      season 64 (2024-25): all 50 states + DC approved
      season 65 (2025-26): NOT approved for Alaska, Connecticut, Hawaii,
                           New York City, Oklahoma, Oregon, South Dakota, Utah
  That is an exact 8/8 match to the states missing from FluView. These states did not
  authorize publication of their 2025-26 ILI data. For NY specifically it is NEW YORK
  CITY that withheld, and Delphi derives 'ny' from 'ny_minus_jfk' + NYC, so 'ny'
  becomes unconstructible and returns nothing.

  This matters for the paper in a way a code bug would not: the actual-risk leg has
  MISSING DATA BY STATE CHOICE, which is not missing-at-random by default. Checked:
  the 8 withholding states are NOT politically distinctive (mean 2024 margin +4.4 vs
  +5.1 for the other 43, Mann-Whitney p = 0.75), so the missingness is not entangled
  with the partisanship confound. That is one sentence in the limitations, and it is
  a sentence worth having.

WHY THIS IS NOT A COSMETIC FIX
  1. NY sits in the over-worried tail of D. The old pipeline was deleting one of the
     strongest divergence signals in the paper.
  2. It moved a headline number. Attention vs actual was reported as rho +0.320,
     p = 0.036 (significant!) on the 43-state pool. On the complete 51-state pool it
     is rho +0.116, p = 0.417. Not significant.
  3. It was the COVERAGE, not the measure. On the SAME 43 states, FluView gives
     +0.320 and NSSP gives +0.294 - the two surveillance measures agree. Adding the
     8 missing states is what collapses the correlation. That diagnostic is in
     compute_d.py and it is worth reporting: it is the honest way to explain why a
     number changed.
  This all HELPS the argument. Weaker attention-actual coupling means MORE room for
  divergence, which is the thesis. But the old number was wrong and it does not get
  quoted again.

WHAT NSSP MEASURES
  Percent of emergency-department visits that were for influenza. All 51 states.
  CAVEAT WORTH SAYING OUT LOUD: ED-visit share is partly care-seeking behavior, so it
  is not purely 'actual'. A worried population may go to the ED more, which puts a
  little bit of perceived risk inside the actual leg. compute_d.py therefore reports
  D under BOTH NSSP and FluView, so you can show the tails are stable either way.

Run:  python3 nssp_by_state.py
Source: https://api.delphi.cmu.edu/epidata/covidcast/  (free, public, no key)
Note: python urllib hits SSL: CERTIFICATE_VERIFY_FAILED against this host on this
Mac, so this goes through curl.
"""
import csv, json, os, subprocess, sys, time
import numpy as np

EPIWEEKS = "202601-202605"   # the flu peak; matches trends and concern windows
SIGNAL = "pct_ed_visits_influenza"


def fetch(source="nssp", signal=SIGNAL, epiweeks=EPIWEEKS, tries=5):
    """
    Delphi rate-limits anonymous queries and answers with an HTML 429, not JSON.
    If you hit it a lot, register a FREE key at api.delphi.cmu.edu/epidata/admin/registration_form
    and set DELPHI_API_KEY in your environment; this picks it up automatically.
    """
    key = os.environ.get("DELPHI_API_KEY", "").strip()
    for i in range(tries):
        cmd = ["curl", "-s", "--get", "https://api.delphi.cmu.edu/epidata/covidcast/",
               "--data-urlencode", f"data_source={source}",
               "--data-urlencode", f"signal={signal}",
               "--data-urlencode", "time_type=week",
               "--data-urlencode", "geo_type=state",
               "--data-urlencode", f"time_values={epiweeks}",
               "--data-urlencode", "geo_value=*"]
        if key:
            cmd += ["--data-urlencode", f"api_key={key}"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        try:
            return json.loads(r.stdout)
        except Exception:
            if "429" in r.stdout or "Too Many Requests" in r.stdout:
                wait = 20 * (i + 1)
                print(f"  rate-limited by Delphi; waiting {wait}s (attempt {i+1}/{tries})")
                time.sleep(wait); continue
            print("  raw response:", r.stdout[:300])
            sys.exit("  could not parse Delphi response")
    sys.exit("  Delphi kept rate-limiting. Get a free key (see fetch() docstring) "
             "and set DELPHI_API_KEY, or just wait a few minutes.")


def show_the_bug():
    """Reproduce the FluView coverage bug so it is documented, not folklore."""
    print("\n  --- the FluView bug, reproduced ---")
    cmd = ["curl", "-s",
           "https://api.delphi.cmu.edu/epidata/fluview/"
           "?regions=ny,ct,ak,hi,ca,tx&epiweeks=202548-202609"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    try:
        d = json.loads(r.stdout)
    except Exception:
        print("    (could not reach FluView; skipping the demo)"); return
    got = sorted(set(x["region"] for x in (d.get("epidata") or [])))
    print(f"    asked FluView for : ny, ct, ak, hi, ca, tx")
    print(f"    FluView said      : result={d.get('result')} ({d.get('message')})")
    print(f"    FluView returned  : {got}")
    print(f"    -> it reports SUCCESS while silently returning nothing for 4 of 6 states.")


def verdict(vals, label):
    v = np.asarray(vals, dtype=float)
    n = len(v)
    cv = v.std() / v.mean() if v.mean() else 0
    iqr = np.percentile(v, 75) - np.percentile(v, 25)
    print(f"\n  {label}")
    print(f"    states     : {n}")
    print(f"    range      : {v.min():.2f}% to {v.max():.2f}%")
    print(f"    median     : {np.median(v):.2f}%")
    print(f"    IQR        : {iqr:.2f}")
    print(f"    coeff var  : {cv:.3f}")
    if n < 15:
        print("    NO-GO: too few states."); return False
    if cv < 0.05:
        print("    NO-GO: states nearly identical; ranks would be noise."); return False
    print("    GO: real spread across states.")
    return True


def main():
    print(f"CDC NSSP '{SIGNAL}', epiweeks {EPIWEEKS}, by state (via Delphi Epidata)")
    d = fetch()
    if not d or d.get("result") != 1:
        sys.exit(f"  API said: {d.get('message') if d else 'no response'}")
    rows = d["epidata"]

    per = {}
    for r in rows:
        if r.get("value") is None:
            continue
        per.setdefault(r["geo_value"], []).append(float(r["value"]))
    means = {s: float(np.mean(v)) for s, v in per.items() if v}
    ranked = sorted(means.items(), key=lambda kv: -kv[1])

    print(f"  rows returned : {len(rows)}")
    print(f"  states        : {len(means)}   <- FluView gave 43 for the same season")
    missing_from_fluview = ["ak", "ct", "hi", "ny", "ok", "or", "sd", "ut"]
    recovered = [s for s in missing_from_fluview if s in means]
    print(f"  recovered by this fix: {', '.join(x.upper() for x in recovered)}")

    print("\n  highest flu burden (mean % of ED visits):")
    for s, v in ranked[:8]:
        star = "  <- was missing from FluView" if s in missing_from_fluview else ""
        print(f"    {s.upper():3} {v:6.2f}{star}")
    print("  lowest:")
    for s, v in ranked[-5:]:
        star = "  <- was missing from FluView" if s in missing_from_fluview else ""
        print(f"    {s.upper():3} {v:6.2f}{star}")

    ok = verdict(list(means.values()), "ACTUAL-RISK side (mean % ED visits for flu)")
    show_the_bug()

    # Stamp the window into the file. compute_d.py refuses to mix windows, because
    # mixing them is exactly how the separability headline got faked once already.
    with open("flu_ed_by_state.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["state", "mean_pct_ed_visits_influenza", "epiweeks"])
        for s, v in ranked:
            w.writerow([s, round(v, 4), EPIWEEKS])
    print("\n  saved -> flu_ed_by_state.csv")

    if ok:
        print("\n  NEXT: python3 compute_d.py")


if __name__ == "__main__":
    main()
