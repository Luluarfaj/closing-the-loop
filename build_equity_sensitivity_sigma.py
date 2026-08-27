#!/usr/bin/env python3
"""Equity sensitivity sweep (Figure 6b) against the SIGMA-CORRECTED model.

Same sweep as build_equity_sensitivity.py, same three drivers, same grids, same
budget-equal comparison, same layout, palette, fonts, figure size and dpi. The
only change is the model: run_two_sigma with sigA = 0.12 (protected majority A)
and sigB = 0.04 (under-protected minority B) replaces run_two's single global
sigma of 0.12.

build_equity_sensitivity.py and sim_calibrated.py are NOT modified.

The manuscript's Methods now says:

    "Across all 30, the advantage of targeting over broadcasting runs from 47 to
     66 percentage points of community B's outbreak averted, and targeting wins
     in 30 of 30 settings."

and Section 4.2.2:

    "targeting averted more of the highly susceptible community's outbreak than
     broadcasting, by 47 to 66 percentage points, with the base-case gap of 85
     against 24 sitting inside that range"

Both ranges are recomputed here at runtime and checked against those figures.

Run:  python3 build_equity_sensitivity_sigma.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_calibrated import run_two, win, CAL
from sim_two_sigma import run_two_sigma

OUT = "/Users/ahmeds/Desktop/Closing the Loop/Papers/NHB_figures_sigma"
os.makedirs(OUT, exist_ok=True)

SIG_A = 0.12
SIG_B = 0.04

# Palette and rcParams: byte-identical to build_equity_sensitivity.py.
GREEN = "#06402F"; RED = "#B0402F"; SLATE = "#2B3A33"
GREY = "#7A8580"; INK = "#22302A"; AMBER = "#8A5A00"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": SLATE, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": SLATE, "ytick.color": SLATE})

P = dict(CAL)
DOSE_BC = 0.010   # broadcast intensity applied to everyone; total budget = DOSE_BC

# Manuscript claims, for a runtime cross-check only. Never used as a label.
MS_RANGE = (47, 66)
MS_WINS = 30
MS_BASE = (24, 85)   # broadcast, targeted


def averted_pair(sigA, sigB, fB=0.20, SA0=0.10, SB0=0.60, assort=0.70, model="sigma"):
    """(broadcast, targeted) percent of community B's outbreak averted, equal budget."""
    fA = 1 - fB
    kw = dict(fA=fA, fB=fB, SA0=SA0, SB0=SB0, assort=assort)
    dose_tg = DOSE_BC / fB          # same total budget, concentrated on B
    if model == "sigma":
        run = lambda **m: run_two_sigma(P, sigA, sigB, **m, **kw)
    else:
        run = lambda **m: run_two(P, **m, **kw)
    none = run()
    bc = run(M_A=lambda t: win(t, 6, 26, DOSE_BC), M_B=lambda t: win(t, 6, 26, DOSE_BC))
    tg = run(M_A=lambda t: 0.0, M_B=lambda t: win(t, 6, 26, dose_tg))
    av = lambda x: 100 * (none["attackB"] - x["attackB"]) / none["attackB"] \
        if none["attackB"] > 1e-9 else 0.0
    return av(bc), av(tg)


new = lambda **kw: averted_pair(SIG_A, SIG_B, model="sigma", **kw)
old = lambda **kw: averted_pair(None, None, model="global", **kw)

# ---- sweeps: vary one driver, hold the others at base --------------------
assorts = np.linspace(0.50, 0.90, 9)    # 0.5 = well mixed, 0.9 = B self-contained
SA0s = np.linspace(0.10, 0.55, 10)      # A susceptibility rising = susceptibility diffuse
fBs = np.linspace(0.10, 0.35, 11)       # community B population share

sw_assort = [new(assort=a) for a in assorts]
sw_SA0 = [new(SA0=s) for s in SA0s]
sw_fB = [new(fB=f) for f in fBs]

fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.2))


def panel(a, x, pairs, xlab, title, base_x):
    bc = [p[0] for p in pairs]; tg = [p[1] for p in pairs]
    a.plot(x, tg, "-o", color=GREEN, lw=2.4, ms=5, label="targeted")
    a.plot(x, bc, "-o", color=GREY, lw=2.2, ms=5, label="broadcast")
    a.fill_between(x, bc, tg, color=GREEN, alpha=.08)
    a.axvline(base_x, color=AMBER, ls=":", lw=1.5)
    a.text(base_x, 101, "base", color=AMBER, fontsize=9, ha="center", va="bottom")
    a.set_xlabel(xlab, fontsize=11); a.set_ylim(0, 108)
    a.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.grid(alpha=.12)


panel(ax[0], assorts, sw_assort, "assortative mixing (0.5 mixed → 0.9 separate)",
      "(a) how separate community B is", 0.70)
panel(ax[1], SA0s, sw_SA0, "susceptibility of the protected majority",
      "(b) how concentrated the susceptibility is", 0.10)
panel(ax[2], fBs, sw_fB, "community B share of the population",
      "(c) how large the under-protected group is", 0.20)
ax[0].set_ylabel("percent of community B's outbreak averted", fontsize=11)
ax[0].legend(fontsize=10, frameon=False, loc="lower left")
fig.suptitle("Targeting's advantage over broadcasting holds across the plausible range of assumptions",
             fontsize=13.5, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_equity_sensitivity.png", dpi=190, facecolor="white", bbox_inches="tight")
plt.close(fig)

# ---- report -------------------------------------------------------------
adv = lambda pairs: [p[1] - p[0] for p in pairs]
print("=" * 78)
print(f"EQUITY SENSITIVITY SWEEP (Figure 6b)   sigA = {SIG_A}   sigB = {SIG_B}")
print("=" * 78)

print("\n[1] OLD, single global sigma = %.2f (what the shipped figure shows)" % P["sigma"])
o_assort = [old(assort=a) for a in assorts]
o_SA0 = [old(SA0=s) for s in SA0s]
o_fB = [old(fB=f) for f in fBs]
o_all = o_assort + o_SA0 + o_fB
o_bc, o_tg = old()
print(f"    base case            broadcast {o_bc:.2f}%  targeted {o_tg:.2f}%  advantage {o_tg-o_bc:.2f} pts")
for lab, sw in (("assort", o_assort), ("SA0", o_SA0), ("fB", o_fB)):
    a_ = adv(sw)
    print(f"    {lab:<8} advantage {min(a_):6.2f} to {max(a_):6.2f} pts")
a_ = adv(o_all)
print(f"    OVERALL  advantage {min(a_):.2f} to {max(a_):.2f} pts, i.e. {min(a_):.0f} to {max(a_):.0f}"
      f"   targeting wins by more than 2 pts in {sum(1 for b, t in o_all if t > b + 2)}/{len(o_all)}")

print(f"\n[2] NEW, sigA = {SIG_A} / sigB = {SIG_B} (what this figure shows)")
n_all = sw_assort + sw_SA0 + sw_fB
n_bc, n_tg = new()
print(f"    base case            broadcast {n_bc:.2f}%  targeted {n_tg:.2f}%  advantage {n_tg-n_bc:.2f} pts")
for lab, sw in (("assort", sw_assort), ("SA0", sw_SA0), ("fB", sw_fB)):
    a_ = adv(sw)
    print(f"    {lab:<8} advantage {min(a_):6.2f} to {max(a_):6.2f} pts")
a_ = adv(n_all)
lo, hi = min(a_), max(a_)
wins = sum(1 for b, t in n_all if t > b + 2)
strict_wins = sum(1 for b, t in n_all if t > b)
print(f"    OVERALL  advantage {lo:.2f} to {hi:.2f} pts, i.e. {lo:.0f} to {hi:.0f}"
      f"   targeting wins by more than 2 pts in {wins}/{len(n_all)}"
      f"   and wins outright in {strict_wins}/{len(n_all)}")

print("\n[3] AGREEMENT WITH THE MANUSCRIPT TEXT")
fails = []
checks = [
    ("sweep range low", round(lo), MS_RANGE[0]),
    ("sweep range high", round(hi), MS_RANGE[1]),
    ("settings won", strict_wins, MS_WINS),
    ("base broadcast averts", round(n_bc), MS_BASE[0]),
    ("base targeted averts", round(n_tg), MS_BASE[1]),
]
for lab, got, want in checks:
    ok = got == want
    fails += [] if ok else [lab]
    print(f"    {lab:<24} model {got:>3}   text {want:>3}   {'OK' if ok else 'MISMATCH'}")
print(f"    VERDICT: {'figure 6b agrees with the manuscript text' if not fails else 'CONFLICT in ' + ', '.join(fails)}")

print("\n[4] ANNOTATIONS ON THE FIGURE ITSELF")
print("    The only text annotation is the non-numeric 'base' marker over each")
print("    panel's base-case vertical line. There is no hard-coded number anywhere")
print("    on this figure; the plotted curve values are the model output, and the")
print("    y range 0 to 108 is unchanged. Nothing on it can go stale.")

print("\n[5] OUTPUT")
from PIL import Image
im = Image.open(f"{OUT}/fig_equity_sensitivity.png")
print(f"    {OUT}/fig_equity_sensitivity.png   {im.size[0]}x{im.size[1]}   REGENERATED")
