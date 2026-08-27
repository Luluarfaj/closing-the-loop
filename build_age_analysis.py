#!/usr/bin/env python3
"""Westyn's question: age is the single biggest real risk factor for flu/COVID, so
does state worry track age? If worry tracks the vote but NOT age, that is a second
sign the worry signal is political rather than a reading of real risk. Median age by
state, 2020 Census. Worry/vote/actual from the divergence CSVs."""
import pandas as pd, numpy as np
from scipy import stats

AGE={"maine":45.0,"new hampshire":43.1,"vermont":43.0,"west virginia":43.0,"florida":42.7,
"delaware":41.4,"connecticut":41.2,"pennsylvania":40.9,"rhode island":40.3,"new jersey":40.2,
"montana":40.2,"michigan":40.1,"south carolina":40.1,"wisconsin":40.0,"hawaii":40.0,"oregon":39.9,
"massachusetts":39.7,"ohio":39.6,"alabama":39.5,"new york":39.4,"kentucky":39.2,"north carolina":39.2,
"maryland":39.2,"tennessee":39.1,"missouri":39.1,"illinois":38.8,"virginia":38.7,"wyoming":38.7,
"iowa":38.6,"new mexico":38.6,"arkansas":38.6,"minnesota":38.5,"nevada":38.5,"arizona":38.5,
"mississippi":38.3,"indiana":38.0,"washington":37.9,"louisiana":37.8,"south dakota":37.6,
"colorado":37.3,"georgia":37.3,"kansas":37.3,"california":37.3,"idaho":37.2,"oklahoma":37.1,
"nebraska":36.9,"north dakota":35.4,"alaska":35.3,"texas":35.2,"district of columbia":34.4,"utah":31.5}
ABBR={"alabama":"al","alaska":"ak","arizona":"az","arkansas":"ar","california":"ca","colorado":"co",
"connecticut":"ct","delaware":"de","district of columbia":"dc","florida":"fl","georgia":"ga","hawaii":"hi",
"idaho":"id","illinois":"il","indiana":"in","iowa":"ia","kansas":"ks","kentucky":"ky","louisiana":"la",
"maine":"me","maryland":"md","massachusetts":"ma","michigan":"mi","minnesota":"mn","mississippi":"ms",
"missouri":"mo","montana":"mt","nebraska":"ne","nevada":"nv","new hampshire":"nh","new jersey":"nj",
"new mexico":"nm","new york":"ny","north carolina":"nc","north dakota":"nd","ohio":"oh","oklahoma":"ok",
"oregon":"or","pennsylvania":"pa","rhode island":"ri","south carolina":"sc","south dakota":"sd",
"tennessee":"tn","texas":"tx","utah":"ut","vermont":"vt","virginia":"va","washington":"wa",
"west virginia":"wv","wisconsin":"wi","wyoming":"wy"}
age_by_abbr={ABBR[k]:v for k,v in AGE.items()}

def partial(x,y,z):
    rxy=stats.pearsonr(x,y)[0]; rxz=stats.pearsonr(x,z)[0]; ryz=stats.pearsonr(y,z)[0]
    return (rxy-rxz*ryz)/np.sqrt((1-rxz**2)*(1-ryz**2))

print(f"{'':7} {'worry~AGE':>10} {'worry~VOTE':>11} {'worry~DISEASE':>14} {'actual~AGE':>11} {'AGE~VOTE':>9} {'worry~AGE|vote':>15}")
for name in ["flu","covid"]:
    d=pd.read_csv(f"divergence_{name}.csv"); d["age"]=d["state"].map(age_by_abbr); d=d.dropna(subset=["age"])
    r_wa=stats.pearsonr(d.worry,d.age)[0]; r_wv=stats.pearsonr(d.worry,d.vote)[0]
    r_wd=stats.pearsonr(d.worry,d.actual)[0]; r_aa=stats.pearsonr(d.actual,d.age)[0]
    r_av=stats.pearsonr(d.age,d.vote)[0]; r_wa_v=partial(d.worry,d.age,d.vote)
    print(f"{name:7} {r_wa:+10.2f} {r_wv:+11.2f} {r_wd:+14.2f} {r_aa:+11.2f} {r_av:+9.2f} {r_wa_v:+15.2f}  (n={len(d)})")
