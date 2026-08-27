#!/usr/bin/env python3
"""Every correlation coefficient, interval, P value and sensitivity the paper reports.

Written so that a reader can reproduce each number in the Methods section
"Uncertainty and statistical analysis" without re-deriving the conventions.

Conventions, stated once and applied throughout:
  Pearson interval  Fisher z transform, SE = 1/sqrt(n-3), transformed back.
  Pearson P         t test on n-2 degrees of freedom.
  Spearman interval Fisher z applied to rho with the Bonett-Wright standard
                    error sqrt((1 + rho^2/2)/(n-3)), transformed back.
  Spearman P        the same t approximation on n-2 degrees of freedom.
  Rank partial      both variables and the control are ranked, each is regressed
                    on the ranked control by ordinary least squares, and the two
                    residual series are correlated. Significance on n-3 df.

Nothing here is fitted or chosen after seeing the answer. Seeds are fixed.
"""
import numpy as np
import pandas as pd
from scipy import stats

D_DIR = "/Users/ahmeds/Desktop/Code/PythonProject1/closing_the_loop"
Z = 1.959964
SEED = 20260728
N_BOOT_AGREE = 20_000      # bootstrap used only to check the transform-based intervals
N_PERM = 200_000           # permutation test on the November partial correlation


# ---------------------------------------------------------------- intervals
def pearson(x, y):
    r, p = stats.pearsonr(x, y)
    n = len(x)
    se = 1.0 / np.sqrt(n - 3)
    z = np.arctanh(r)
    lo, hi = np.tanh(z - Z * se), np.tanh(z + Z * se)
    return r, lo, hi, p


def spearman(x, y):
    rho, p = stats.spearmanr(x, y)
    n = len(x)
    se = np.sqrt((1.0 + rho ** 2 / 2.0) / (n - 3))     # Bonett and Wright
    z = np.arctanh(rho)
    lo, hi = np.tanh(z - Z * se), np.tanh(z + Z * se)
    return rho, lo, hi, p


def pearson_partial(x, y, z_ctrl):
    """Pearson partial correlation, residualising both variables on the control.
       This is the convention for the age relationships, which are reported as Pearson."""
    x, y, z_ctrl = (np.asarray(v, float) for v in (x, y, z_ctrl))
    ex = x - np.polyval(np.polyfit(z_ctrl, x, 1), z_ctrl)
    ey = y - np.polyval(np.polyfit(z_ctrl, y, 1), z_ctrl)
    r = np.corrcoef(ex, ey)[0, 1]
    n = len(x); df = n - 3
    t = r * np.sqrt(df / max(1e-12, 1 - r ** 2))
    p = 2 * stats.t.sf(abs(t), df)
    se = 1.0 / np.sqrt(n - 3 - 1)
    zz = np.arctanh(r)
    return r, np.tanh(zz - Z * se), np.tanh(zz + Z * se), p, df


def rank_partial(x, y, z_ctrl):
    """Spearman partial correlation by the rank residual method."""
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z_ctrl)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    r = np.corrcoef(ex, ey)[0, 1]
    n = len(x); df = n - 3
    t = r * np.sqrt(df / max(1e-12, 1 - r ** 2))
    p = 2 * stats.t.sf(abs(t), df)
    se = 1.0 / np.sqrt(n - 3 - 1)
    zz = np.arctanh(r)
    return r, np.tanh(zz - Z * se), np.tanh(zz + Z * se), p, df


def fmt(name, r, lo, hi, p, extra=""):
    print(f"  {name:44s} {r:+.3f}  95% CI {lo:+.3f} to {hi:+.3f}   P = {p:.4g}{extra}")


# ---------------------------------------------------------------- data
def _read_age_tables():
    """Read the AGE and ABBR literals out of build_age_analysis.py without executing it."""
    import ast
    src = open(f"{D_DIR}/build_age_analysis.py", encoding="utf-8").read()
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("AGE", "ABBR"):
                out[node.targets[0].id] = ast.literal_eval(node.value)
    return out["AGE"], out["ABBR"]


def load():
    flu = pd.read_csv(f"{D_DIR}/divergence_flu.csv")
    cov = pd.read_csv(f"{D_DIR}/divergence_covid.csv")
    AGE, ABBR = _read_age_tables()
    age = {ABBR[k]: v for k, v in AGE.items()}
    for d in (flu, cov):
        d["age"] = d["state"].map(age)
    tr = pd.read_csv(f"{D_DIR}/flu_trends_by_state.csv")
    name2abbr = {k.title(): v for k, v in ABBR.items()}
    tr["state"] = tr["geoName"].map(lambda s: name2abbr.get(s, ABBR.get(s.lower())))
    att = flu.merge(tr[["state", "flu"]].rename(columns={"flu": "attention"}), on="state")
    return flu, cov, att


def main():
    flu, cov, att = load()
    # the two disease files are not guaranteed to share a row order: merge on state
    xd = flu[["state", "worry"]].rename(columns={"worry": "worry_flu"}).merge(
         cov[["state", "worry"]].rename(columns={"worry": "worry_cov"}), on="state")
    assert len(xd) == len(flu), "cross-disease merge lost rows"
    rng = np.random.default_rng(SEED)
    print("=" * 84)
    print("THE ELEVEN HEADLINE COEFFICIENTS      n = %d jurisdictions" % len(flu))
    print("=" * 84)

    pairs = [
        ("influenza concern ~ 2024 two-party vote", flu["worry"], flu["vote"]),
        ("COVID-19 concern ~ 2024 two-party vote",  cov["worry"], cov["vote"]),
        ("influenza concern ~ ED burden",           flu["worry"], flu["actual"]),
        ("COVID-19 concern ~ ED burden",            cov["worry"], cov["actual"]),
        ("influenza concern ~ COVID-19 concern",    xd["worry_flu"], xd["worry_cov"]),
        ("search attention ~ 2024 two-party vote",  att["attention"], att["vote"]),
        ("influenza concern ~ median age",          flu["worry"], flu["age"]),
        ("COVID-19 concern ~ median age",           cov["worry"], cov["age"]),
        ("median age ~ 2024 two-party vote",        flu["age"], flu["vote"]),
    ]
    print("\n-- Pearson --")
    pear = {}
    for name, x, y in pairs:
        r, lo, hi, p = pearson(x, y); pear[name] = p; fmt(name, r, lo, hi, p)
    print("\n-- Spearman (Bonett-Wright interval) --")
    spear = {}
    for name, x, y in pairs:
        r, lo, hi, p = spearman(x, y); spear[name] = p; fmt(name, r, lo, hi, p)

    # ------------------------------------------------ does the transform hold up
    print("\n" + "=" * 84)
    print("DOES THE TRANSFORM-BASED INTERVAL AGREE WITH A BOOTSTRAP?  B = %s" % f"{N_BOOT_AGREE:,}")
    print("=" * 84)
    worst = 0.0
    for name, x, y in pairs:
        x = np.asarray(x, float); y = np.asarray(y, float)
        rho, lo, hi, _ = spearman(x, y)
        idx = rng.integers(0, len(x), size=(N_BOOT_AGREE, len(x)))
        b = np.array([stats.spearmanr(x[i], y[i]).statistic for i in idx])
        blo, bhi = np.percentile(b, [2.5, 97.5])
        d = max(abs(blo - lo), abs(bhi - hi)); worst = max(worst, d)
        print(f"  {name:44s} transform [{lo:+.2f},{hi:+.2f}]  bootstrap [{blo:+.2f},{bhi:+.2f}]  max diff {d:.3f}")
    print(f"\n  largest disagreement across all pairs: {worst:.3f}")

    # ------------------------------------------------ age net of the vote
    print("\n" + "=" * 84)
    print("AGE NET OF THE VOTE  (Pearson partial correlation, per the stated convention)")
    print("=" * 84)
    for lbl, d in (("influenza", flu), ("COVID-19", cov)):
        r, lo, hi, p, df = pearson_partial(d["worry"], d["age"], d["vote"])
        fmt(f"{lbl} concern ~ median age | vote", r, lo, hi, p, f"   df = {df}")

    # ------------------------------------------------ drop-one sensitivities
    print("\n" + "=" * 84)
    print("DROP-ONE SENSITIVITIES  (the four flagged coefficients)")
    print("=" * 84)
    def drop(d, xcol, ycol, state):
        s = d[d["state"] != state]
        return pearson(s[xcol], s[ycol])
    for lbl, d, xc, yc, drops in (
            ("influenza concern ~ ED burden", flu, "worry", "actual", ["dc"]),
            ("median age ~ vote",             flu, "age",   "vote",   ["dc"]),
            ("COVID-19 concern ~ median age", cov, "worry", "age",    ["dc", "ut"])):
        r, lo, hi, p = pearson(d[xc], d[yc])
        print(f"  {lbl}")
        print(f"      all 51            r = {r:+.3f}, P = {p:.4f}")
        for st in drops:
            r2, _, _, p2 = drop(d, xc, yc, st)
            print(f"      dropping {st.upper():3s}       r = {r2:+.3f}, P = {p2:.4f}")

    # ------------------------------------------------ permutation test
    print("\n" + "=" * 84)
    print("PERMUTATION TEST ON THE RAW NOVEMBER RHO  shuffles = %s, seed %d" % (f"{N_PERM:,}", SEED))
    print("=" * 84)
    acts = pd.read_csv(f"{D_DIR}/decision_rule_actions.csv")
    nov = acts[(acts.disease == "flu") & (acts.window == "November 30 - December 27")]
    x = nov["concern"].to_numpy(float); y = nov["burden"].to_numpy(float)
    z_ = nov["gop_margin"].to_numpy(float) if "gop_margin" in nov else flu["vote"].to_numpy(float)
    obs = stats.spearmanr(x, y).statistic
    partial_obs = rank_partial(x, y, z_)[0]
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rxc = rx - rx.mean(); nx = np.sqrt(rxc @ rxc)
    rng2 = np.random.default_rng(SEED)
    hits = 0; done = 0; BATCH = 5000
    while done < N_PERM:
        k = min(BATCH, N_PERM - done)
        P = np.array([rng2.permutation(ry) for _ in range(k)])
        Pc = P - P.mean(axis=1, keepdims=True)
        r = (Pc @ rxc) / (np.sqrt((Pc * Pc).sum(axis=1)) * nx)
        hits += int((np.abs(r) >= abs(obs) - 1e-12).sum())
        done += k
    print(f"  observed Spearman rho             : {obs:+.4f}")
    print(f"  two-sided permutation P           : {(hits + 1) / (N_PERM + 1):.5f}")
    print(f"  rank partial, vote controlled     : {partial_obs:+.4f}  (parametric P from the table above)")

    # ------------------------------------------------ multiplicity
    print("\n" + "=" * 84)
    print("MULTIPLICITY OVER THE FAMILY OF ELEVEN")
    print("=" * 84)
    novp = rank_partial(x, y, z_)[3]
    rho_nov, _, _, p_nov = spearman(x, y)
    fam = {
        "influenza concern ~ vote":            pear["influenza concern ~ 2024 two-party vote"],
        "COVID-19 concern ~ vote":             pear["COVID-19 concern ~ 2024 two-party vote"],
        "influenza concern ~ burden":          pear["influenza concern ~ ED burden"],
        "COVID-19 concern ~ burden":           pear["COVID-19 concern ~ ED burden"],
        "influenza ~ COVID-19 concern":        pear["influenza concern ~ COVID-19 concern"],
        "attention ~ vote":                    pear["search attention ~ 2024 two-party vote"],
        "influenza concern ~ age":             pear["influenza concern ~ median age"],
        "COVID-19 concern ~ age":              pear["COVID-19 concern ~ median age"],
        "median age ~ vote (Pearson, stated)": pear["median age ~ 2024 two-party vote"],
        "November rho":                        p_nov,
        "November partial":                    novp,
    }
    m = len(fam); alpha = 0.05
    print(f"  family size {m}; Bonferroni threshold {alpha/m:.4f}")
    bonf = [k for k, v in fam.items() if v < alpha / m]
    order = sorted(fam.items(), key=lambda kv: kv[1])
    k = 0
    for i, (_, v) in enumerate(order, 1):
        if v <= i / m * alpha: k = i
    print(f"  survive Bonferroni        : {len(bonf)}")
    print(f"  survive Benjamini-Hochberg: {k}")
    alt = dict(fam); alt["median age ~ vote (Pearson, stated)"] = spear["median age ~ 2024 two-party vote"]
    o2 = sorted(alt.values()); k2 = 0
    for i, v in enumerate(o2, 1):
        if v <= i / m * alpha: k2 = i
    print(f"  if the age~vote member entered as Spearman instead: Benjamini-Hochberg {k2}, Bonferroni {sum(1 for v in alt.values() if v < alpha/m)}")


if __name__ == "__main__":
    main()
