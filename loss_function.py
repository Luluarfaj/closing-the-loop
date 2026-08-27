#!/usr/bin/env python3
"""Derive the decision rule's thresholds from a stylised loss function instead of asserting them.

The manuscript asserts three things separately: an asymmetric threshold, a rule to act only when the
divergence exceeds its uncertainty, and a qualification on the calming branch. This derives all three
from one expected-loss argument.

THE STRUCTURAL POINT. The two errors differ in kind, not merely in size:
  under-response (perceived below actual) -> people under-protect -> excess infections, and that harm
                                             scales with how much disease is actually there
  over-response  (perceived above actual) -> wasted precaution and anxiety, and in this paper's own
                                             model a message is itself attention and feeds loop R1
The first scales with burden. The second does not. That asymmetry is what produces an asymmetric
threshold. It is not a parameter we chose.

Units: divergence in van der Waerden rank-score units. Burden in percent of ED visits. Loss in
"over-perception-equivalent" units, so k_over = 1 is the numeraire.
"""
import numpy as np
import pandas as pd
from scipy import stats

D_DIR = "/Users/ahmeds/Desktop/Code/PythonProject1/closing_the_loop"

# ---- stylised parameters (ILLUSTRATIVE; swept below) ----
K_OVER = 1.0     # harm per unit of over-perception (numeraire)
K_UNDER = 1.0    # harm per unit of under-perception PER UNIT BURDEN
M_SHIFT = 1.0    # how far a campaign moves perceived risk, in rank-score units
C_FIXED = 0.20   # fixed cost of running any campaign (budget + credibility)


def _E_pos(mu, sd):
    """E[max(X,0)] for X ~ N(mu, sd). Closed form."""
    if sd <= 1e-12:
        return max(mu, 0.0)
    z = mu / sd
    return mu * stats.norm.cdf(z) + sd * stats.norm.pdf(z)


def expected_loss(action, D, sd, B, k_under=K_UNDER, k_over=K_OVER, m=M_SHIFT, c=C_FIXED):
    """Expected loss of an action, integrating over the posterior on the true divergence."""
    shift = {"HOLD": 0.0, "ALERT": +m, "CALM": -m}[action]
    mu = D + shift
    under = k_under * B * _E_pos(-mu, sd)
    over = k_over * _E_pos(mu, sd)
    fixed = 0.0 if action == "HOLD" else c
    return under + over + fixed


def thresholds(B, k_under=K_UNDER, k_over=K_OVER, m=M_SHIFT, c=C_FIXED):
    """Closed-form thresholds in the zero-uncertainty limit.
       ALERT when D < -t_alert ; CALM when D > +t_calm."""
    denom = k_over + k_under * B
    t_alert = (k_over * m + c) / denom
    t_calm = (k_under * B * m + c) / denom
    return t_alert, t_calm


def decide_loss(D, sd, B, **kw):
    """Choose the action with the lowest expected loss."""
    losses = {a: expected_loss(a, D, sd, B, **kw) for a in ("HOLD", "ALERT", "CALM")}
    best = min(losses, key=losses.get)
    return best, losses


def _simplify(s):
    return (s.replace("ALERT_AND_MOBILIZE", "ALERT")
             .replace("CALM_AND_CLARIFY", "CALM")
             .replace("HOLD_AND_WATCH", "HOLD")
             .replace("HOLD_AND_MONITOR", "HOLD"))


if __name__ == "__main__":
    acts = pd.read_csv(f"{D_DIR}/decision_rule_actions.csv")
    acts["sd"] = ((acts["D_hi"] - acts["D_lo"]) / (2 * 1.959964)).clip(lower=1e-9)

    print("=" * 78)
    print("DERIVED THRESHOLDS AS A FUNCTION OF BURDEN  (k_under=%.2f, m=%.2f, c=%.2f)"
          % (K_UNDER, M_SHIFT, C_FIXED))
    print("=" * 78)
    print(f"{'burden %ED':>11} {'t_alert':>9} {'t_calm':>9}   which way the rule leans")
    for B in [0.0, 0.14, 0.58, 2.0, 3.58, 4.31, 7.0]:
        ta, tc = thresholds(B)
        note = "calms more readily" if tc < ta else "alerts more readily"
        print(f"{B:>11.2f} {ta:>9.3f} {tc:>9.3f}   {note}")
    print("\nAt zero burden the rule calms more readily than it alerts. As burden rises the alert")
    print("threshold falls and the calm threshold rises. That is the asymmetry, derived, and it")
    print("delivers the paper's 'safer error' language as a consequence rather than a claim.")

    live = acts[acts["gate"] == "PASS"].copy()
    print("\n" + "=" * 78)
    print("RERUN OF THE GATE-PASSING CELL (%s, %s), n=%d"
          % (live["disease"].iloc[0], live["window"].iloc[0], len(live)))
    print("=" * 78)
    live["action_loss"] = [decide_loss(r["D"], r["sd"], r["pool_burden"])[0]
                           for _, r in live.iterrows()]
    old = live["action"].map(_simplify)
    print("\nOLD (thresholds we chose, plus the 2.00 percent burden cut):")
    print(old.value_counts().to_string())
    print("\nNEW (derived from the loss function):")
    print(live["action_loss"].value_counts().to_string())
    changed = int((old.values != live["action_loss"].values).sum())
    print(f"\njurisdictions changing decision: {changed} of {len(live)}")

    print("\n" + "=" * 78)
    print("SENSITIVITY TO THE COST RATIO k_under")
    print("=" * 78)
    print(f"{'k_under':>8} {'t_alert@4.31':>13} {'t_calm@4.31':>12} {'ALERT':>7} {'CALM':>6} {'HOLD':>6}")
    for ku in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        ta, tc = thresholds(4.3092, k_under=ku)
        cnt = {"ALERT": 0, "CALM": 0, "HOLD": 0}
        for _, r in live.iterrows():
            cnt[decide_loss(r["D"], r["sd"], r["pool_burden"], k_under=ku)[0]] += 1
        print(f"{ku:>8.2f} {ta:>13.3f} {tc:>12.3f} {cnt['ALERT']:>7} {cnt['CALM']:>6} {cnt['HOLD']:>6}")

    print("\n" + "=" * 78)
    print("DOES THE UNCERTAINTY CONDITION FALL OUT?")
    print("=" * 78)
    agree = dis = 0
    for _, r in live.iterrows():
        acts_by_loss = decide_loss(r["D"], r["sd"], r["pool_burden"])[0] != "HOLD"
        if acts_by_loss == bool(r["excludes_zero"]):
            agree += 1
        else:
            dis += 1
    print(f"loss-based action agrees with the bootstrap-interval rule on {agree} of {len(live)} places,")
    print(f"disagrees on {dis}.")
    print("\nThe loss rule holds when the expected harm reduction does not cover the fixed cost, which")
    print("happens when |D| is small relative to its uncertainty. It is a smooth version of the paper's")
    print("existing stipulation rather than an identical test.")

    live.to_csv(f"{D_DIR}/loss_function_actions.csv", index=False)
    print(f"\nwrote {D_DIR}/loss_function_actions.csv")
