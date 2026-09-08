#!/usr/bin/env python3
"""Audit tie sensitivity of split-panel ICLR proportional top-list concordance."""
from __future__ import annotations
import argparse, json, math, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def c(g): return g-.5*g*g

def v(g): return g-g*g+g**3/3

def se(n,g1,g2): return math.sqrt(v(g1)*v(g2)/(c(g1)*c(g2)*n))

def rank(score, ids, mode, rng=None):
    score=np.asarray(score,float); ids=np.asarray(ids,str)
    if mode=='ascending_id': order=np.lexsort((ids,-score))
    elif mode=='descending_id':
        # integer hash provides a deterministic reversed tie ordering
        h=np.array([int.from_bytes(x.encode()[:8].ljust(8,b'0'),'little') for x in ids],dtype=np.uint64)
        order=np.lexsort((-h,-score))
    elif mode=='random':
        if rng is None: raise ValueError
        order=np.lexsort((rng.random(len(score)),-score))
    else: raise ValueError(mode)
    r=np.empty(len(order),dtype=np.int32); r[order]=np.arange(1,len(order)+1)
    return r

def btau(r1,r2,m1,m2):
    a=np.minimum(r1,m1+1); b=np.minimum(r2,m2+1)
    return float(kendalltau(a,b,variant='b').statistic)

def records_from_file(p):
    out=[]
    try:
        if p.suffix.lower()=='.csv':
            df=pd.read_csv(p)
            cols={x.lower():x for x in df.columns}
            pid=next((cols[x] for x in ['paper_id','paperid','forum','id'] if x in cols),None)
            score=next((cols[x] for x in ['actual_score','score','rating'] if x in cols),None)
            if pid and score:
                year=next((cols[x] for x in ['year','conference_year'] if x in cols),None)
                for _,r in df.iterrows(): out.append({'paper_id':str(r[pid]),'score':float(r[score]),'year':str(r[year]) if year else None})
        elif p.suffix.lower() in {'.json','.jsonl'}:
            text=p.read_text(encoding='utf-8',errors='replace').strip()
            objs=[]
            if p.suffix.lower()=='.jsonl':
                objs=[json.loads(x) for x in text.splitlines() if x.strip()]
            else:
                x=json.loads(text); objs=x if isinstance(x,list) else x.get('records',x.get('data',[])) if isinstance(x,dict) else []
            for r in objs:
                if not isinstance(r,dict): continue
                pid=r.get('paper_id',r.get('paperid',r.get('forum',r.get('id'))))
                sc=r.get('actual_score',r.get('score',r.get('rating')))
                if pid is not None and sc is not None:
                    out.append({'paper_id':str(pid),'score':float(sc),'year':str(r.get('year')) if r.get('year') is not None else None})
    except Exception:
        return []
    if out:
        inferred=re.search(r'20(?:23|24)',str(p))
        for r in out:
            if not r['year'] and inferred: r['year']=inferred.group(0)
    return out

def load_records(root):
    allr=[]
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.csv','.json','.jsonl'}:
            allr.extend(records_from_file(p))
    if not allr: raise RuntimeError('No ICLR review records found')
    df=pd.DataFrame(allr).dropna(subset=['paper_id','score'])
    df['year']=df.year.fillna('unknown')
    return df.drop_duplicates()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--seeds',type=int,default=50); ap.add_argument('--random-tie-reps',type=int,default=20)
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    df=load_records(args.input_dir)
    gammas=[(.10,.10),(.20,.20),(.10,.20),(.20,.10)]
    rows=[]
    for year,dy in df.groupby('year'):
        groups={pid:g.score.to_numpy(float) for pid,g in dy.groupby('paper_id') if len(g)>=4}
        ids=np.array(sorted(groups))
        if len(ids)<200: continue
        for seed0 in range(args.seeds):
            seed=20260908+seed0
            panel_a=[]; panel_b=[]
            for pid in ids:
                x=groups[pid].copy(); rng=np.random.default_rng(abs(hash((seed,pid)))%(2**32)); rng.shuffle(x); h=len(x)//2; panel_a.append(x[:h].mean()); panel_b.append(x[h:].mean())
            a=np.asarray(panel_a); b=np.asarray(panel_b); n=len(ids)
            raa=rank(a,ids,'ascending_id'); rba=rank(b,ids,'ascending_id')
            rad=rank(a,ids,'descending_id'); rbd=rank(b,ids,'descending_id')
            tie_a=1-len(np.unique(a))/n; tie_b=1-len(np.unique(b))/n
            for g1,g2 in gammas:
                m1=max(2,min(n-1,round(g1*n))); m2=max(2,min(n-1,round(g2*n)))
                base=btau(raa,rba,m1,m2); rev=btau(rad,rbd,m1,m2)
                rand=[]
                for rep in range(args.random_tie_reps):
                    rng=np.random.default_rng(seed*1000+rep)
                    rand.append(btau(rank(a,ids,'random',rng),rank(b,ids,'random',rng),m1,m2))
                rand=np.asarray(rand)
                rows.append({'year':year,'seed':seed,'n':n,'m1':m1,'m2':m2,'gamma1':m1/n,'gamma2':m2/n,'tau_id_ascending':base,'tau_id_descending':rev,'tie_rule_abs_difference':abs(base-rev),'random_tie_tau_mean':rand.mean(),'random_tie_tau_sd':rand.std(ddof=1),'random_tie_tau_min':rand.min(),'random_tie_tau_max':rand.max(),'tie_rate_panel_a':tie_a,'tie_rate_panel_b':tie_b,'theorem3_se':se(n,m1/n,m2/n)})
    out=pd.DataFrame(rows)
    if out.empty: raise RuntimeError('No year with >=200 eligible papers')
    out.to_csv(args.output_dir/'iclr_tie_sensitivity.csv',index=False)
    s=out.groupby(['year','gamma1','gamma2']).agg(n=('n','median'),tau_mean=('tau_id_ascending','mean'),tie_rule_diff_mean=('tie_rule_abs_difference','mean'),tie_rule_diff_max=('tie_rule_abs_difference','max'),random_tie_sd_mean=('random_tie_tau_sd','mean'),random_tie_sd_max=('random_tie_tau_sd','max'),tie_rate_a_mean=('tie_rate_panel_a','mean'),tie_rate_b_mean=('tie_rate_panel_b','mean')).reset_index()
    s.to_csv(args.output_dir/'iclr_tie_sensitivity_summary.csv',index=False)
    meta={'records':len(df),'years':sorted(df.year.unique()),'eligible_years':sorted(out.year.unique()),'seeds':args.seeds,'random_tie_reps':args.random_tie_reps}
    (args.output_dir/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','rows':len(out),'max_tie_rule_difference':float(out.tie_rule_abs_difference.max())}))
if __name__=='__main__': main()
