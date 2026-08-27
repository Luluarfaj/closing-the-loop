#!/usr/bin/env python3
"""
SUPERSEDED AS THE PRIMARY ACTUAL LEG. Use nssp_by_state.py instead. Kept because
compute_d.py uses this as a ROBUSTNESS CHECK, and because the bug below is worth
keeping on the record.

THE SYMPTOM: this script silently returns 43 states, not 51. FluView answers
"result": 1, "success" and returns NOTHING for ak, ct, hi, ny, ok, or, sd, ut.
Reproduce it:
    curl "https://api.delphi.cmu.edu/epidata/fluview/?regions=ny,ct,ak,hi,ca,tx&epiweeks=202601-202605"
    -> only ca and tx come back.

THE CAUSE is data governance, not a code bug. CDC's GRASP endpoint
    curl "https://gis.cdc.gov/grasp/flu2/GetPhase02InitApp?appVersion=Public"
publishes a state_data_approval table. For season 65 (2025-26), weekly ILI data was
NOT approved for release by Alaska, Connecticut, Hawaii, New York City, Oklahoma,
Oregon, South Dakota and Utah - an exact match to the 8 missing states. All were
approved in season 64. NY vanishes because NEW YORK CITY withheld and Delphi builds
'ny' from 'ny_minus_jfk' + NYC.

That deletion was load-bearing: 4 of the 8 sit in the extreme tails of D (NY has the
LOWEST flu burden in the country and above-median concern, making it one of the most
over-worried states in the paper), and dropping them moved the headline
attention-vs-actual correlation to +0.320 (p=0.036) from +0.116 (n.s.) on the
complete pool.

---

The ACTUAL-RISK side for D: CDC FluView state-level ILI, via the Delphi Epidata API.

ILI = the percent of outpatient visits for influenza-like illness. It is the
standard state-level flu surveillance measure, and it is the natural
'actual risk' benchmark for the divergence metric.

Window matches trends_by_state.py (the 2025-26 flu season peak).

Run:  python3 fluview_by_state.py
Source: https://api.delphi.cmu.edu/epidata/fluview/  (free, public, no key for fluview)
"""
import json, os, subprocess, sys, time
import numpy as np

STATES = ["al","ak","az","ar","ca","co","ct","de","dc","fl","ga","hi","id","il","in","ia",
          "ks","ky","la","me","md","ma","mi","mn","ms","mo","mt","ne","nv","nh","nj","nm",
          "ny","nc","nd","oh","ok","or","pa","ri","sc","sd","tn","tx","ut","vt","va","wa",
          "wv","wi","wy"]
# WINDOW MUST MATCH THE PERCEIVED LEG (CDC publishes concern for 2025-12-28 to
# 2026-01-31 and that is fixed). This was "202548-202609" (Dec 1 - Feb 28), which is
# 13 weeks against the concern leg's 5. Window choice is not innocent: for the SAME
# NSSP measure, Dec-Feb and Jan rank states at only rho +0.54.
EPIWEEKS = "202601-202605"   # 2025-12-28 to 2026-01-31

def fetch(regions, tries=5):
    """
    Delphi rate-limits anonymous queries with an HTML 429, not JSON. Free key:
    api.delphi.cmu.edu/epidata/admin/registration_form -> set DELPHI_API_KEY.
    """
    key = os.environ.get("DELPHI_API_KEY", "").strip()
    url = ("https://api.delphi.cmu.edu/epidata/fluview/"
           f"?regions={','.join(regions)}&epiweeks={EPIWEEKS}")
    if key:
        url += f"&api_key={key}"
    for i in range(tries):
        r = subprocess.run(["curl", "-s", url], capture_output=True, text=True, timeout=90)
        try:
            return json.loads(r.stdout)
        except Exception:
            if "429" in r.stdout or "Too Many Requests" in r.stdout:
                wait = 20 * (i + 1)
                print(f"  rate-limited by Delphi; waiting {wait}s (attempt {i+1}/{tries})")
                time.sleep(wait); continue
            print("  raw response:", r.stdout[:200]); return None
    print("  Delphi kept rate-limiting. Wait a few minutes or set DELPHI_API_KEY.")
    return None

def verdict(vals, label):
    v = np.asarray(vals, dtype=float)
    n = len(v); cv = v.std()/v.mean() if v.mean() else 0
    iqr = np.percentile(v,75)-np.percentile(v,25)
    print(f"\n  {label}")
    print(f"    states     : {n}")
    print(f"    range      : {v.min():.2f} to {v.max():.2f}")
    print(f"    median     : {np.median(v):.2f}")
    print(f"    IQR        : {iqr:.2f}")
    print(f"    coeff var  : {cv:.3f}")
    if n < 15:   print("    NO-GO: too few states."); return False
    if cv < 0.05: print("    NO-GO: states nearly identical; ranks would be noise."); return False
    print("    GO: real spread across states.")
    return True

def main():
    print(f"Delphi Epidata FluView, epiweeks {EPIWEEKS}, {len(STATES)} states")
    d = fetch(STATES)
    if not d or d.get("result") != 1:
        sys.exit(f"  API said: {d.get('message') if d else 'no response'}")
    rows = d["epidata"]
    print(f"  rows returned: {len(rows)}")
    # prefer wili (weighted ILI); fall back to ili
    key = "wili" if any(r.get("wili") is not None for r in rows) else "ili"
    print(f"  using field: {key}")
    per = {}
    for r in rows:
        v = r.get(key)
        if v is None: continue
        per.setdefault(r["region"], []).append(float(v))
    means = {s: float(np.mean(vs)) for s, vs in per.items() if vs}
    ranked = sorted(means.items(), key=lambda kv: -kv[1])
    print("\n  highest flu burden (mean %s):" % key)
    for s, v in ranked[:8]: print(f"    {s.upper():3} {v:6.2f}")
    print("  lowest:")
    for s, v in ranked[-5:]: print(f"    {s.upper():3} {v:6.2f}")
    ok = verdict(list(means.values()), f"ACTUAL-RISK side (mean {key} per state)")
    import csv
    with open("flu_ili_by_state.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["state", f"mean_{key}", "epiweeks"])
        for s,v in ranked: w.writerow([s, round(v,3), EPIWEEKS])
    print("\n  saved -> flu_ili_by_state.csv")
    if ok:
        print("\n  Both legs now exist with real data:")
        print("    attention   = flu_trends_by_state.csv   (Google Trends)")
        print("    actual risk = flu_ili_by_state.csv      (CDC FluView)")
        print("  STILL MISSING: perceived risk (Reddit appraisal per post).")
        print("  Attention is NOT perceived risk. Without the text leg it is not D.")

if __name__ == "__main__":
    main()
