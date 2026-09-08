#!/usr/bin/env python3
"""MovieLens split-user-panel candidate for proportional-truncation concordance.

Users are partitioned into disjoint panels.  Each panel ranks the same eligible
movies by its own mean rating.  Eligibility is determined symmetrically by a
minimum rating count in both panels.  The script evaluates top-list overlap,
bottom-tied Kendall tau_b, Theorem 3 random-ranking z references, tie sensitivity,
and finite random-permutation calibration.  It does not modify the manuscript.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm


def c(g: float) -> float:
    return g - 0.5 * g * g


def v(g: float) -> float:
    return g - g * g + g**3 / 3.0


def se_theorem3(n: int, g1: float, g2: float) -> float:
    return np.sqrt(v(g1) * v(g2) / (c(g1) * c(g2) * n))


def rank_from_score(score: np.ndarray, ids: np.ndarray, reverse_tie: bool = False) -> np.ndarray:
    tie = -ids if reverse_tie else ids
    order = np.lexsort((tie, -score))
    rank = np.empty(len(order), dtype=np.int32)
    rank[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return rank


def bottom_tied_tau(r1: np.ndarray, r2: np.ndarray, m1: int, m2: int) -> float:
    a = np.minimum(r1, m1 + 1)
    b = np.minimum(r2, m2 + 1)
    stat = kendalltau(a, b, variant="b").statistic
    return float(stat)


def load_ratings(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith("ratings.dat")]
        if len(names) != 1:
            raise RuntimeError(f"Expected one ratings.dat, found {names}")
        with z.open(names[0]) as fh:
            df = pd.read_csv(
                fh,
                sep="::",
                engine="python",
                names=["user", "movie", "rating", "timestamp"],
                usecols=[0, 1, 2, 3],
            )
    if len(df) != 1_000_209 or df.user.nunique() != 6040:
        raise RuntimeError(f"Unexpected MovieLens 1M dimensions: {df.shape}, users={df.user.nunique()}")
    return df


def one_split(df: pd.DataFrame, seed: int, min_count: int, gammas: list[tuple[float, float]]) -> list[dict]:
    rng = np.random.default_rng(seed)
    users = np.sort(df.user.unique())
    panel = rng.integers(0, 2, size=len(users), endpoint=False)
    # force both panels nonempty and near-balanced through a seeded permutation
    perm = rng.permutation(len(users))
    panel[:] = 1
    panel[perm[: len(users)//2]] = 0
    map_panel = pd.Series(panel, index=users)
    x = df.assign(panel=df.user.map(map_panel))
    agg = x.groupby(["panel", "movie"]).rating.agg(["mean", "count"]).reset_index()
    a = agg[agg.panel == 0].set_index("movie")
    b = agg[agg.panel == 1].set_index("movie")
    common = a.index.intersection(b.index)
    a = a.loc[common]
    b = b.loc[common]
    eligible = common[(a["count"] >= min_count).to_numpy() & (b["count"] >= min_count).to_numpy()]
    a = a.loc[eligible]
    b = b.loc[eligible]
    ids = eligible.to_numpy(dtype=int)
    n = len(ids)
    if n < 300:
        raise RuntimeError(f"Too few eligible movies ({n}) at min_count={min_count}")
    r1 = rank_from_score(a["mean"].to_numpy(), ids)
    r2 = rank_from_score(b["mean"].to_numpy(), ids)
    r1_rev = rank_from_score(a["mean"].to_numpy(), ids, reverse_tie=True)
    r2_rev = rank_from_score(b["mean"].to_numpy(), ids, reverse_tie=True)
    tie_rate_a = float(1.0 - a["mean"].nunique() / n)
    tie_rate_b = float(1.0 - b["mean"].nunique() / n)
    rows=[]
    for g1,g2 in gammas:
        m1=max(2,min(n-1,int(round(g1*n))))
        m2=max(2,min(n-1,int(round(g2*n))))
        tau=bottom_tied_tau(r1,r2,m1,m2)
        tau_rev=bottom_tied_tau(r1_rev,r2_rev,m1,m2)
        top1=set(np.flatnonzero(r1<=m1))
        top2=set(np.flatnonzero(r2<=m2))
        overlap=len(top1&top2)
        se=se_theorem3(n,m1/n,m2/n)
        z=tau/se
        rows.append({
            "seed":seed,"n":n,"min_count_per_panel":min_count,
            "gamma1_target":g1,"gamma2_target":g2,"m1":m1,"m2":m2,
            "gamma1":m1/n,"gamma2":m2/n,
            "bottom_tied_tau_b":tau,"theorem3_se":se,"theorem3_z":z,
            "theorem3_p_two_sided":2*norm.sf(abs(z)),
            "overlap_count":overlap,"overlap_over_min_depth":overlap/min(m1,m2),
            "tie_rate_panel_a":tie_rate_a,"tie_rate_panel_b":tie_rate_b,
            "reverse_id_tie_tau_b":tau_rev,"tie_break_abs_difference":abs(tau-tau_rev),
            "panel_a_users":int((panel==0).sum()),"panel_b_users":int((panel==1).sum()),
            "panel_a_median_ratings_per_movie":float(a['count'].median()),
            "panel_b_median_ratings_per_movie":float(b['count'].median()),
        })
    return rows


def null_calibration(n: int, gammas: list[tuple[float,float]], reps: int, seed: int) -> pd.DataFrame:
    rng=np.random.default_rng(seed)
    rows=[]
    base=np.arange(1,n+1,dtype=np.int32)
    for g1,g2 in gammas:
        m1=max(2,min(n-1,int(round(g1*n))))
        m2=max(2,min(n-1,int(round(g2*n))))
        se=se_theorem3(n,m1/n,m2/n)
        zs=[]
        for _ in range(reps):
            p1=rng.permutation(base)
            p2=rng.permutation(base)
            zs.append(bottom_tied_tau(p1,p2,m1,m2)/se)
        z=np.asarray(zs)
        rows.append({
            "n":n,"m1":m1,"m2":m2,"gamma1":m1/n,"gamma2":m2/n,
            "reps":reps,"null_z_mean":float(z.mean()),"null_z_sd":float(z.std(ddof=1)),
            "null_abs_z_gt_1_96":float(np.mean(np.abs(z)>1.96)),
            "null_q025":float(np.quantile(z,.025)),"null_q975":float(np.quantile(z,.975)),
        })
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--zip',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--seeds',type=int,default=100)
    ap.add_argument('--min-count',type=int,default=30)
    ap.add_argument('--null-reps',type=int,default=2000)
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    df=load_ratings(args.zip)
    gammas=[(.05,.05),(.10,.10),(.20,.20),(.05,.10),(.10,.20)]
    rows=[]
    for seed in range(args.seeds):
        rows.extend(one_split(df,20260908+seed,args.min_count,gammas))
    out=pd.DataFrame(rows)
    out.to_csv(args.output_dir/'movielens_split_panel_results.csv',index=False)
    summary=out.groupby(['gamma1_target','gamma2_target']).agg(
        n_median=('n','median'),tau_mean=('bottom_tied_tau_b','mean'),tau_sd=('bottom_tied_tau_b','std'),
        tau_median=('bottom_tied_tau_b','median'),tau_min=('bottom_tied_tau_b','min'),tau_max=('bottom_tied_tau_b','max'),
        overlap_mean=('overlap_over_min_depth','mean'),tie_break_max=('tie_break_abs_difference','max'),
        tie_break_mean=('tie_break_abs_difference','mean'),tie_rate_a_mean=('tie_rate_panel_a','mean'),tie_rate_b_mean=('tie_rate_panel_b','mean')
    ).reset_index()
    summary.to_csv(args.output_dir/'movielens_split_panel_summary.csv',index=False)
    n=int(round(out.n.median()))
    cal=null_calibration(n,gammas,args.null_reps,20260908)
    cal.to_csv(args.output_dir/'movielens_theorem3_calibration.csv',index=False)
    meta={
        'source_file':str(args.zip),'source_sha256':hashlib.sha256(args.zip.read_bytes()).hexdigest(),
        'ratings_rows':len(df),'users':int(df.user.nunique()),'movies':int(df.movie.nunique()),
        'seeds':args.seeds,'min_count_per_panel':args.min_count,'median_eligible_n':float(out.n.median()),
        'panel_assignment':'disjoint seeded 50/50 split of users','ranking_score':'panel-specific arithmetic mean rating',
        'tie_break':'movie ID; reversed-ID sensitivity also reported','manuscript_modified':False,
    }
    (args.output_dir/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','rows':len(out),'median_n':float(out.n.median()),'output':str(args.output_dir)}))

if __name__=='__main__': main()
