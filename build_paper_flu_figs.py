#!/usr/bin/env python3
"""Loop paper figures with flu and COVID as CO-EQUAL cases. Reads divergence_flu.csv
and divergence_covid.csv (build_respiratory.py). Writes into ClosingTheLoop_LaTeX/:
  fig2.png       measurement condition: worry vs the vote / vs the disease, flu & COVID
  fig3.png       the divergence, flu and COVID side by side, neutral, with error bars
  fig_trait.png  worry is a fixed trait across diseases (flu worry vs COVID worry)"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats

import os
OUT=os.path.expanduser("~/Desktop/Closing the Loop/Papers/ClosingTheLoop_LaTeX")
GREEN="#06402F"; RED="#B0402F"; SLATE="#2B3A33"; GREY="#7A8580"; INK="#22302A"
GREEN_F="#E8F3EE"; RED_F="#F5E6E1"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.edgecolor":SLATE,"text.color":INK,
                     "axes.labelcolor":INK,"xtick.color":SLATE,"ytick.color":SLATE})
flu=pd.read_csv("divergence_flu.csv"); cov=pd.read_csv("divergence_covid.csv")
flu["state_u"]=flu["state"].str.upper(); cov["state_u"]=cov["state"].str.upper()

# ================= FIGURE 2: worry follows the vote, not the disease (flu & COVID) =================
def drv(ax,x,y,strong,xlab):
    col=RED if strong else GREY
    ax.scatter(x,y,s=42,color=SLATE if strong else GREY,alpha=.72,edgecolor="white",linewidth=.5,zorder=3)
    b,a=np.polyfit(x,y,1); xs=np.linspace(x.min(),x.max(),40); ax.plot(xs,b*xs+a,color=col,lw=2.3,zorder=2)
    r=stats.pearsonr(x,y)[0]
    ax.text(0.05,0.07,f"r = {r:+.2f}",transform=ax.transAxes,fontsize=13,fontweight="bold",color=col,va="bottom")
    ax.set_xlabel(xlab,fontsize=10.5)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.grid(alpha=.12)
fig,ax=plt.subplots(2,2,figsize=(9.6,7.8))
drv(ax[0,0],flu.vote,flu.worry,True,"share voting Republican (2024, %)")
drv(ax[0,1],cov.vote,cov.worry,True,"share voting Republican (2024, %)")
drv(ax[1,0],flu.actual,flu.worry,False,"actual Flu  (% of ER visits)")
drv(ax[1,1],cov.actual,cov.worry,False,"actual COVID  (% of ER visits)")
for a in (ax[0,0],ax[0,1],ax[1,0],ax[1,1]): a.set_ylim(15,52)
ax[0,0].set_title("Flu",fontsize=15,fontweight="bold",color=INK,pad=10)
ax[0,1].set_title("COVID",fontsize=15,fontweight="bold",color=INK,pad=10)
ax[0,0].set_ylabel("how worried people are  (%)",fontsize=10.5); ax[1,0].set_ylabel("how worried people are  (%)",fontsize=10.5)
fig.text(0.5,0.965,"Worry follows the vote, not the disease  (Flu and COVID)",ha="center",fontsize=16,fontweight="bold",color=INK)
fig.tight_layout(rect=[0,0,1,0.93]); fig.subplots_adjust(hspace=0.46)
fig.savefig(f"{OUT}/fig2.png",dpi=190,facecolor="white",bbox_inches="tight"); plt.close(fig)

# ================= FIGURE 3: the divergence, flu and COVID side by side, with error bars =================
def diverge(ax,df,title):
    n=len(df)
    ax.fill_between([0,n+1],[0,n+1],[n+1,n+1],color=GREEN_F,zorder=0)
    ax.fill_between([0,n+1],[0,0],[0,n+1],color=RED_F,zorder=0)
    ax.plot([0,n+1],[0,n+1],ls="--",color=SLATE,lw=1.1,zorder=2)
    for _,r in df.iterrows():
        over=r["D_z"]>0; gated=not r["excludes_zero"]; col=GREEN if over else RED
        ax.errorbar(r["rank_actual"],r["rank_worry"],
                    yerr=[[r["rank_worry"]-r["worry_lo"]],[r["worry_hi"]-r["rank_worry"]]],
                    fmt="o",ms=6.5,color=col,ecolor=col,elinewidth=.9,capsize=1.5,
                    alpha=(0.32 if gated else 0.95),mec="white",mew=.5,zorder=3)
    for st in ["dc","ny","ks","ia","ut"]:
        r=df[df.state==st]
        if len(r): ax.annotate(r["state_u"].iloc[0],(r["rank_actual"].iloc[0],r["rank_worry"].iloc[0]),
                    textcoords="offset points",xytext=(5,3),fontsize=8.5,fontweight="bold",color=INK,zorder=5)
    ax.text(3,n-2,"over-worried",fontsize=11,fontweight="bold",color=GREEN,va="top")
    ax.text(n-2,3,"under-worried",fontsize=11,fontweight="bold",color=RED,ha="right",va="bottom")
    ax.set_xlim(0,n+1); ax.set_ylim(0,n+1); ax.set_aspect("equal")
    ax.set_xlabel("actual burden (state rank)",fontsize=10.5)
    ax.set_title(title,fontsize=14,fontweight="bold",color=INK,pad=8)
    for s in ("top","right"): ax.spines[s].set_visible(False)
fig,ax=plt.subplots(1,2,figsize=(12.6,6.8))
diverge(ax[0],flu,"Flu"); diverge(ax[1],cov,"COVID")
ax[0].set_ylabel("public concern (state rank)",fontsize=10.5)
fig.text(0.5,0.975,"The divergence sorts by the vote, not the disease",ha="center",fontsize=15,fontweight="bold",color=INK)
fig.text(0.5,0.925,"state concern rank vs actual-burden rank; faded dots are within the bootstrap noise, so the signal is in the tails",
         ha="center",fontsize=10,color=GREY,style="italic")
fig.tight_layout(rect=[0,0,1,0.90]); fig.savefig(f"{OUT}/fig3.png",dpi=190,facecolor="white",bbox_inches="tight"); plt.close(fig)

# ================= FIGURE 3b: worry is a fixed trait across diseases =================
m=flu.merge(cov,on="state",suffixes=("_flu","_cov")); r=stats.pearsonr(m.worry_flu,m.worry_cov)[0]
fig,ax=plt.subplots(figsize=(6.8,6.4))
ax.scatter(m.worry_flu,m.worry_cov,s=64,color=SLATE,alpha=.75,edgecolor="white",linewidth=.7,zorder=3)
b,a=np.polyfit(m.worry_flu,m.worry_cov,1); xs=np.linspace(m.worry_flu.min(),m.worry_flu.max(),40)
ax.plot(xs,b*xs+a,color=RED,lw=2.6,zorder=2)
for st in ["dc","ut","ny","wy","ma","id"]:
    rr=m[m.state==st]
    if len(rr): ax.annotate(st.upper(),(rr.worry_flu.iloc[0],rr.worry_cov.iloc[0]),
                textcoords="offset points",xytext=(6,4),fontsize=9.5,fontweight="bold",color=INK)
ax.text(0.05,0.93,f"r = {r:+.2f}",transform=ax.transAxes,fontsize=17,fontweight="bold",color=RED,va="top")
ax.set_xlabel("worry about the Flu  (%)",fontsize=12); ax.set_ylabel("worry about COVID  (%)",fontsize=12)
ax.set_title("A state's worry barely changes with the disease",fontsize=14,fontweight="bold",color=INK,pad=10)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.grid(alpha=.13)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_trait.png",dpi=190,facecolor="white",bbox_inches="tight"); plt.close(fig)
print("wrote fig2.png (drivers), fig3.png (divergence flu+COVID), fig_trait.png to",OUT)
print(f"  gated middle: flu {int((~flu.excludes_zero).sum())}/51, covid {int((~cov.excludes_zero).sum())}/51 | trait r={r:+.2f}")
