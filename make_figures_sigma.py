#!/usr/bin/env python3
"""Rebuild the simulation figures against the SIGMA-CORRECTED two-community model.

Why this file exists
--------------------
The manuscript (ClosingTheLoop_15.08_LA_v6.docx) was updated with a
community-specific attenuation fix: the SARF gain sigma is no longer a single
global number in the two-community test, but sigA = 0.12 for the protected
majority A and sigB = 0.04 for the under-protected minority B. Appendix A now
says so explicitly:

    "The SARF attenuation gain is community-specific too: sigma=0.12 in
     community A, the same value the single-population model uses, and
     sigma=0.04 in community B, so that only community B sits in the
     downplayed regime."

The manuscript tables and text carry the corrected numbers. The EMBEDDED
FIGURES do not: fig6.png still shows "broadcast (averts 29%)", the value the
old global-sigma model produced.

What this script does
---------------------
  fig6.png  REGENERATED from sim_two_sigma.run_two_sigma(sigA=0.12, sigB=0.04).
  fig4.png  VERIFIED UNCHANGED. The single-population model is untouched by the
  fig5.png  fix, so these are re-rendered only to prove byte-for-byte identity
            with the shipped originals, into a separate verify/ subfolder.

Nothing in sim_calibrated.py, sim_two_sigma.py or make_figures.py is modified.
Every number printed on a figure is computed at runtime from the model.

Layout note
-----------
The shipped fig4/fig5/fig6 are the "no title" variants: make_figures.py laid
them out WITH a suptitle (tight_layout(rect=[0,0,1,0.95])) and the suptitle band
was then blanked. Setting the suptitle, running tight_layout with the same rect,
and hiding the artist before saving reproduces that geometry exactly. Verified:
zero differing pixels against all three shipped PNGs.

Run:  python3 make_figures_sigma.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_calibrated import run_single, run_two, win, CAL
from sim_two_sigma import run_two_sigma

# ---------------------------------------------------------------- settings --
OUT = "/Users/ahmeds/Desktop/Closing the Loop/Papers/NHB_figures_sigma"
VERIFY = os.path.join(OUT, "verified_unchanged")
ORIG = "/Users/ahmeds/Desktop/Closing the Loop/Papers/ClosingTheLoop_LaTeX"

SIG_A = 0.12   # protected majority; identical to the published global sigma
SIG_B = 0.04   # under-protected minority; the downplayed regime

# Call-site settings, unchanged from make_figures.py lines 66-68.
DOSE_BC = 0.010
DOSE_TG = 0.050
KW = dict(fA=0.80, fB=0.20, SA0=0.10, SB0=0.60, assort=0.70)

# Suptitle strings from make_figures.py. They are laid out but never drawn;
# they exist only so tight_layout reserves the same space it did originally.
SUP4 = "Figure 2. Calibrated measles model (2017 Minnesota outbreak anchor). Illustrative dynamics."
SUP5 = "Figure 3. Timing keeps perceived risk in step; an over-strong message overshoots. Illustrative."
SUP6 = "Figure 4. Targeting the under-protected community averts far more than broadcasting. Illustrative."

# What the manuscript now claims, for a runtime cross-check. These are read
# OFF the accepted text, not fed into any label.
#   Section 4.2.1: "averts approximately 24 percent of infections in community
#   B ... averts approximately 85% of infections in Community B"
#   Table 2: none 5.9 / 59.9, broadcast 2.6 / 45.4, targeted 5.1 / 9.1
MS_AVERT_BC, MS_AVERT_TG = 24.0, 85.0
MS_ATTACK = {"none": (5.9, 59.9), "broadcast": (2.6, 45.4), "targeted": (5.1, 9.1)}

os.makedirs(OUT, exist_ok=True)
os.makedirs(VERIFY, exist_ok=True)

P = dict(CAL)
BLUE, RED, GREEN, GREY, ORANGE = "#1f5fbf", "#c0392b", "#1e8449", "#7f8c8d", "#e08e0b"


def save_notitle(fig, suptitle, path):
    """Reproduce the shipped no-title geometry, then save."""
    st = fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    st.set_visible(False)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def pixels_identical(a_path, b_path):
    from PIL import Image
    a = np.asarray(Image.open(a_path).convert("RGB")).astype(int)
    b = np.asarray(Image.open(b_path).convert("RGB")).astype(int)
    if a.shape != b.shape:
        return False, a.shape, b.shape, None
    d = int((np.abs(a - b).sum(axis=2) > 0).sum())
    return d == 0, a.shape, b.shape, d


# ======================================================================
# PART 1. Single-population figures: is the sigma fix relevant at all?
# ======================================================================
# sim_calibrated.run_single takes sigma from p['sigma'] and there is only one
# population, so there is no A/B split for the fix to apply to. sim_two_sigma
# adds run_two_sigma and touches nothing else. The assertion below states the
# claim as an executable fact rather than as a comment.
assert P["sigma"] == SIG_A, "single-population sigma must equal the community A gain"

TIMELY = lambda t: win(t, 6, 26, 0.12)
LATE = lambda t: win(t, 24, 54, 0.12)
STRONG = lambda t: win(t, 6, 26, 0.36)

base = run_single(P)
timely = run_single(P, M=TIMELY)
late = run_single(P, M=LATE)
strong = run_single(P, M=STRONG)
t = base["t"]
attack = lambda r: r["attack"] * 100
averted = lambda r: 100 * (base["attack"] - r["attack"]) / base["attack"]

# ---- fig4: baseline / timely / cumulative -------------------------------
fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
thr = base["inc"] / base["inc"].max()
ax[0].plot(t, thr, color=RED, lw=2, label="actual risk (incidence)")
ax[0].plot(t, base["P"], color=BLUE, lw=2, label="perceived risk")
ax[0].fill_between(t, base["P"], thr, where=(thr > base["P"]), color=RED, alpha=0.12)
ax[0].set_title("Baseline downplayed regime")
ax[0].set_xlabel("day")
ax[0].set_ylabel("scaled level")
ax[0].legend(fontsize=8, loc="upper right")
ax[1].plot(t, base["inc"] / base["inc"].max(), color=GREY, lw=1.6, ls="--", label="incidence, no alert")
ax[1].plot(t, timely["inc"] / base["inc"].max(), color=RED, lw=2, label="incidence, timely alert")
ax[1].plot(t, timely["P"], color=BLUE, lw=2, label="perceived risk")
ax[1].set_title("Timely targeted alert")
ax[1].set_xlabel("day")
ax[1].set_ylabel("scaled level")
ax[1].legend(fontsize=8, loc="upper right")
ax[2].plot(t, base["C"] * 100, color=GREY, lw=2, label=f"no alert ({attack(base):.0f}%)")
ax[2].plot(t, timely["C"] * 100, color=GREEN, lw=2, label=f"timely (averts {averted(timely):.0f}%)")
ax[2].plot(t, late["C"] * 100, color=ORANGE, lw=2, label=f"late (averts {averted(late):.0f}%)")
ax[2].set_title("Cumulative infections")
ax[2].set_xlabel("day")
ax[2].set_ylabel("cumulative infections (% of pop)")
ax[2].legend(fontsize=8, loc="lower right")
save_notitle(fig, SUP4, f"{VERIFY}/fig4.png")

# ---- fig5: standardized threat + overshoot ------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.9))
ax[0].plot(t, thr, color="black", lw=2, ls=":", label="standardized threat")
ax[0].plot(t, base["P"], color=GREY, lw=2, label="perceived risk, no alert")
ax[0].plot(t, timely["P"], color=GREEN, lw=2, label="perceived risk, timely")
ax[0].plot(t, late["P"], color=ORANGE, lw=2, label="perceived risk, late")
ax[0].set_title("Perceived risk vs standardized threat")
ax[0].set_xlabel("day")
ax[0].set_ylabel("scaled level")
ax[0].legend(fontsize=8, loc="upper right")
ax[1].plot(t, timely["C"] * 100, color=GREEN, lw=2, label=f"timely: {attack(timely):.0f}% infected")
ax[1].plot(t, strong["C"] * 100, color=RED, lw=2, ls="--", label=f"3x stronger: {attack(strong):.0f}% infected")
ax[1].plot(t, timely["P"], color=GREEN, lw=1.4, alpha=0.6)
ax[1].plot(t, strong["P"], color=RED, lw=1.4, ls="--", alpha=0.6)
ax[1].annotate("perceived-risk overshoot", xy=(24, strong["P"].max()), xytext=(55, 0.5),
               fontsize=8, arrowprops=dict(arrowstyle="->", color=RED))
ax[1].set_title("A 3x stronger message: little extra benefit, big overshoot")
ax[1].set_xlabel("day")
ax[1].set_ylabel("cumulative infections (%) / perceived risk")
ax[1].legend(fontsize=8, loc="center right")
save_notitle(fig, SUP5, f"{VERIFY}/fig5.png")

# ======================================================================
# PART 2. fig6, the two-community targeting figure. THIS is what changes.
# ======================================================================
none = run_two_sigma(P, SIG_A, SIG_B, **KW)
bc = run_two_sigma(P, SIG_A, SIG_B,
                   M_A=lambda t: win(t, 6, 26, DOSE_BC),
                   M_B=lambda t: win(t, 6, 26, DOSE_BC), **KW)
tg = run_two_sigma(P, SIG_A, SIG_B,
                   M_A=lambda t: 0.0,
                   M_B=lambda t: win(t, 6, 26, DOSE_TG), **KW)
avB = lambda x: 100 * (none["attackB"] - x["attackB"]) / none["attackB"]

fig, ax = plt.subplots(1, 2, figsize=(11.1, 3.9))
ax[0].plot(none["t"], none["incB"], color=GREY, lw=2, label="no campaign")
ax[0].plot(bc["t"], bc["incB"], color=ORANGE, lw=2, label=f"broadcast (averts {avB(bc):.0f}%)")
ax[0].plot(tg["t"], tg["incB"], color=GREEN, lw=2, label=f"targeted (averts {avB(tg):.0f}%)")
ax[0].set_title("Infections over time, community B (under-protected)")
ax[0].set_xlabel("day")
ax[0].set_ylabel("incidence in community B")
ax[0].legend(fontsize=8, loc="upper right")
labels = ["Community A\n(protected)", "Community B\n(under-protected)"]
x = np.arange(2)
w = 0.26
ax[1].bar(x - w, [none["attackA"] * 100, none["attackB"] * 100], w, color=GREY, label="no campaign")
ax[1].bar(x, [bc["attackA"] * 100, bc["attackB"] * 100], w, color=ORANGE, label="broadcast")
ax[1].bar(x + w, [tg["attackA"] * 100, tg["attackB"] * 100], w, color=GREEN, label="targeted")
ax[1].set_xticks(x)
ax[1].set_xticklabels(labels, fontsize=8)
ax[1].set_title("Attack rate by community, same budget")
ax[1].set_ylabel("attack rate (%)")
ax[1].legend(fontsize=8)
save_notitle(fig, SUP6, f"{OUT}/fig6.png")

# ======================================================================
# PART 3. Report and check
# ======================================================================
print("=" * 78)
print(f"SIGMA-CORRECTED FIGURES   sigA = {SIG_A}   sigB = {SIG_B}")
print("=" * 78)

print("\n[1] SINGLE-POPULATION MODEL (fig4, fig5): is it affected by the sigma fix?")
print("    sim_calibrated.run_single reads a single p['sigma'] and has one population,")
print(f"    so there is no A/B split to specialise. p['sigma'] = {P['sigma']} = sigA = {SIG_A}.")
print("    sim_two_sigma.py adds run_two_sigma only; run_single is untouched.")
print(f"    base attack       {attack(base):7.4f}%")
print(f"    timely averts     {averted(timely):7.4f}%   (label shows {averted(timely):.0f}%)")
print(f"    late averts       {averted(late):7.4f}%   (label shows {averted(late):.0f}%)")
print(f"    stronger averts   {averted(strong):7.4f}%")
print(f"    timely attack     {attack(timely):7.4f}%   (label shows {attack(timely):.0f}%)")
print(f"    stronger attack   {attack(strong):7.4f}%   (label shows {attack(strong):.0f}%)")
for f in ("fig4.png", "fig5.png"):
    ok, sa, sb, d = pixels_identical(f"{VERIFY}/{f}", f"{ORIG}/{f}")
    print(f"    {f}: shipped {sb[1]}x{sb[0]}  rebuilt {sa[1]}x{sa[0]}  "
          f"differing pixels {d}  -> {'IDENTICAL, no regeneration needed' if ok else 'CHANGED'}")

print("\n[2] TWO-COMMUNITY MODEL (fig6): attack rates, percent of each community")
print(f"    {'scenario':<12}{'A %':>9}{'B %':>9}{'whole %':>10}{'A n':>9}{'B n':>9}{'total n':>10}")
for name, r in (("none", none), ("broadcast", bc), ("targeted", tg)):
    a, b = r["attackA"], r["attackB"]
    print(f"    {name:<12}{a*100:9.2f}{b*100:9.2f}{(0.80*a+0.20*b)*100:10.2f}"
          f"{a*80000:9.0f}{b*20000:9.0f}{a*80000+b*20000:10.0f}")
print(f"    percent of community B's outbreak averted:  "
      f"broadcast {avB(bc):.2f}   targeted {avB(tg):.2f}   advantage {avB(tg)-avB(bc):.2f} pts")
print(f"    fig6 legend now reads: 'broadcast (averts {avB(bc):.0f}%)'  and  "
      f"'targeted (averts {avB(tg):.0f}%)'")

print("\n[3] AGREEMENT WITH THE MANUSCRIPT TEXT")
fails = []
for name, r in (("none", none), ("broadcast", bc), ("targeted", tg)):
    pa, pb = MS_ATTACK[name]
    ga, gb = r["attackA"] * 100, r["attackB"] * 100
    ok = abs(ga - pa) <= 0.05 and abs(gb - pb) <= 0.05
    fails += [] if ok else [f"Table 2 {name}"]
    print(f"    Table 2 {name:<10} model {ga:5.2f} / {gb:5.2f}   text {pa:4.1f} / {pb:4.1f}   "
          f"{'OK' if ok else 'MISMATCH'}")
for lab, got, want in (("broadcast averts", avB(bc), MS_AVERT_BC),
                       ("targeted averts", avB(tg), MS_AVERT_TG)):
    ok = round(got) == want
    fails += [] if ok else [lab]
    print(f"    {lab:<20} model {got:6.2f}%  rounds to {got:.0f}%   text {want:.0f}%   "
          f"{'OK' if ok else 'MISMATCH'}")
print(f"    VERDICT: {'figure 6 agrees with the manuscript text' if not fails else 'CONFLICT in ' + ', '.join(fails)}")

print("\n[4] OUTPUTS")
from PIL import Image
print(f"    {OUT}/fig6.png   {Image.open(f'{OUT}/fig6.png').size[0]}x{Image.open(f'{OUT}/fig6.png').size[1]}   REGENERATED")
for f in ("fig4.png", "fig5.png"):
    print(f"    {VERIFY}/{f}   "
          f"{Image.open(f'{VERIFY}/{f}').size[0]}x{Image.open(f'{VERIFY}/{f}').size[1]}   "
          f"verification copy only, identical to the shipped file, do not swap")
