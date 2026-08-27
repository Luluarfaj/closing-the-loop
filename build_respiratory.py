#!/usr/bin/env python3
"""Flu and COVID as co-equal cases (seasonal respiratory diseases). Pulls concern
(with CIs) and ED-visit burden by state for both, computes the signed rank
divergence D = vdW(concern) - vdW(actual) with a bootstrap CI on the concern rank,
and saves divergence_flu.csv and divergence_covid.csv. RSV is national-only in
NIS-FRVM (no state concern), so it cannot join a state divergence."""
import subprocess, json
import numpy as np, pandas as pd
from scipy import stats

WIN="December 28 - January 31"; GRP="All adults 18+ years"; EPI="202601-202605"
RES='https://data.cdc.gov/resource/ee83-ukst.json'; NBOOT=3000; SEED=20260728
ABBR={"alabama":"al","alaska":"ak","arizona":"az","arkansas":"ar","california":"ca",
"colorado":"co","connecticut":"ct","delaware":"de","district of columbia":"dc","florida":"fl",
"georgia":"ga","hawaii":"hi","idaho":"id","illinois":"il","indiana":"in","iowa":"ia",
"kansas":"ks","kentucky":"ky","louisiana":"la","maine":"me","maryland":"md","massachusetts":"ma",
"michigan":"mi","minnesota":"mn","mississippi":"ms","missouri":"mo","montana":"mt","nebraska":"ne",
"nevada":"nv","new hampshire":"nh","new jersey":"nj","new mexico":"nm","new york":"ny",
"north carolina":"nc","north dakota":"nd","ohio":"oh","oklahoma":"ok","oregon":"or",
"pennsylvania":"pa","rhode island":"ri","south carolina":"sc","south dakota":"sd","tennessee":"tn",
"texas":"tx","utah":"ut","vermont":"vt","virginia":"va","washington":"wa","west virginia":"wv",
"wisconsin":"wi","wyoming":"wy"}

def vdw(x):
    x=np.asarray(x,float); r=stats.rankdata(x); return stats.norm.ppf(r/(len(x)+1.0))

def pull_concern(indicator):
    where=(f"dsss_indicator_category_label='{indicator}' AND geographic_level='State' "
           f"AND dsss_group_variable_name='{GRP}' AND dsss_timeperiodlabel='{WIN}'")
    cmd=['curl','-s','--get',RES,'--data-urlencode',f'$where={where}','--data-urlencode','$limit=2000']
    d=json.loads(subprocess.run(cmd,capture_output=True,text=True,timeout=90).stdout)
    out={}
    for r in d:
        st=ABBR.get(str(r.get('geographic_label','')).strip().lower())
        if st and r.get('dsss_value') and str(r.get('suppressionflag','0'))=='0':
            lo=hi=np.nan
            try: lo,hi=[float(x.strip()) for x in str(r.get('dsss_confidenceinterval')).split('-')]
            except Exception: pass
            out[st]=(float(r['dsss_value']),lo,hi)
    return out

def pull_ed(signal):
    cmd=['curl','-s','--get','https://api.delphi.cmu.edu/epidata/covidcast/',
         '--data-urlencode','data_source=nssp','--data-urlencode',f'signal={signal}',
         '--data-urlencode','time_type=week','--data-urlencode','geo_type=state',
         '--data-urlencode',f'time_values={EPI}','--data-urlencode','geo_value=*']
    d=json.loads(subprocess.run(cmd,capture_output=True,text=True,timeout=120).stdout)
    per={}
    for r in d.get('epidata',[]):
        if r.get('value') is not None: per.setdefault(r['geo_value'],[]).append(float(r['value']))
    return {s:float(np.mean(v)) for s,v in per.items() if v}

el=pd.read_csv('election_2024_by_state.csv'); el['rep_share']=(100+el['gop_margin'])/2
vote=el.set_index('state')['rep_share'].to_dict()

DIS={'flu':('Concerned about flu disease','pct_ed_visits_influenza'),
     'covid':('Concerned about COVID-19 disease','pct_ed_visits_covid')}
frames={}
print("disease  n   worry~vote  worry~actual")
for name,(ind,sig) in DIS.items():
    c=pull_concern(ind); a=pull_ed(sig)
    rows=[{'state':s,'worry':v[0],'ci_low':v[1],'ci_high':v[2],'actual':a.get(s),'vote':vote.get(s)}
          for s,v in c.items()]
    df=pd.DataFrame(rows).dropna(subset=['worry','actual','vote']).reset_index(drop=True)
    df['rank_worry']=stats.rankdata(df['worry']); df['rank_actual']=stats.rankdata(df['actual'])
    df['z_worry']=vdw(df['worry']); df['z_actual']=vdw(df['actual']); df['D_z']=df['z_worry']-df['z_actual']
    df['D_rank']=df['rank_worry']-df['rank_actual']
    # bootstrap the concern rank from published CIs
    n=len(df); rng=np.random.default_rng(SEED)
    sd=((df['ci_high']-df['ci_low'])/(2*1.959964)).fillna(0).values
    draws=rng.normal(df['worry'].values[None,:],np.where(sd>0,sd,1e-6)[None,:],size=(NBOOT,n))
    br=np.array([stats.rankdata(draws[i]) for i in range(NBOOT)])
    df['worry_lo']=np.percentile(br,2.5,axis=0); df['worry_hi']=np.percentile(br,97.5,axis=0)
    df['excludes_zero']=((df['rank_actual']<df['worry_lo'])|(df['rank_actual']>df['worry_hi']))
    df.to_csv(f'divergence_{name}.csv',index=False); frames[name]=df
    print(f"{name:7} {n:3}  {stats.pearsonr(df.worry,df.vote)[0]:+.2f}       {stats.pearsonr(df.worry,df.actual)[0]:+.2f}")

f=frames['flu'].set_index('state'); cv=frames['covid'].set_index('state'); com=f.index.intersection(cv.index)
print(f"\ncross-disease (n={len(com)}): flu worry vs covid worry r={stats.pearsonr(f.loc[com,'worry'],cv.loc[com,'worry'])[0]:+.2f}")
print("saved divergence_flu.csv, divergence_covid.csv (with bootstrap rank CIs)")
