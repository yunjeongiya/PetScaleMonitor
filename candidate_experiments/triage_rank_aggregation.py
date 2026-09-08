#!/usr/bin/env python3
"""Evaluate TRC as a rank-aggregation metric in high-risk triage data.

Primary candidate: UCI Cardiotocography with ordinal expert-consensus outcome
Normal < Suspect < Pathological. Six unsupervised detectors produce rankings
without labels, and five relevance-blind aggregators combine them.
Secondary stress test: ADBench Mammography with binary anomaly labels.
No manuscript files are modified.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler


def comparable_pair_count(n: int, m: int) -> int:
    return m * (m - 1) // 2 + m * (n - m)


def stable_order_desc(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    if not np.isfinite(score).all():
        raise ValueError("non-finite anomaly score")
    return np.lexsort((np.arange(len(score)), -score)).astype(int)


def rank_vector(order: np.ndarray) -> np.ndarray:
    r = np.empty(len(order), dtype=np.int32)
    r[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return r


def trc_components(order: np.ndarray, outcome: np.ndarray, m: int) -> dict[str, float]:
    y = np.asarray(outcome, dtype=float)[order]
    n = len(y)
    m = int(max(1, min(m, n)))
    within = 0.0
    boundary = 0.0
    for i in range(m):
        if i + 1 < m:
            within += float(np.sign(y[i] - y[i + 1:m]).sum())
        if m < n:
            boundary += float(np.sign(y[i] - y[m:]).sum())
    d_within = m * (m - 1) // 2
    d_boundary = m * (n - m)
    d_total = d_within + d_boundary
    return {
        "trc": (within + boundary) / d_total if d_total else float("nan"),
        "within_component_over_total": within / d_total if d_total else float("nan"),
        "boundary_component_over_total": boundary / d_total if d_total else float("nan"),
        "within_pair_average": within / d_within if d_within else float("nan"),
        "boundary_pair_average": boundary / d_boundary if d_boundary else float("nan"),
        "within_numerator": within,
        "boundary_numerator": boundary,
        "comparable_pairs": d_total,
    }


def oracle_trc(outcome: np.ndarray, m: int) -> float:
    order = np.lexsort((np.arange(len(outcome)), -np.asarray(outcome, dtype=float)))
    return trc_components(order, outcome, m)["trc"]


def evaluate_order(order: np.ndarray, outcome: np.ndarray, m: int, pathological_value: float) -> dict[str, float]:
    rank = rank_vector(order)
    priority = -rank.astype(float)
    comp = trc_components(order, outcome, m)
    full = kendalltau(priority, outcome, variant="b").statistic
    top = order[:m]
    top_tau = kendalltau(priority[top], outcome[top], variant="b").statistic
    gain = np.asarray(outcome, dtype=float) - float(np.min(outcome))
    ndcg = ndcg_score(gain.reshape(1, -1), priority.reshape(1, -1), k=m)
    path = (np.asarray(outcome) == pathological_value).astype(int)
    abnormal = (np.asarray(outcome) > np.min(outcome)).astype(int)
    return {
        **comp,
        "oracle_normalized_trc": comp["trc"] / oracle_trc(outcome, m),
        "full_kendall_tau_b": float(full) if full is not None else float("nan"),
        "top_only_kendall_tau_b": float(top_tau) if top_tau is not None else float("nan"),
        "ndcg_at_m": float(ndcg),
        "ap_pathological": float(average_precision_score(path, priority)),
        "ap_abnormal": float(average_precision_score(abnormal, priority)),
        "auc_pathological": float(roc_auc_score(path, priority)),
        "precision_pathological_at_m": float(path[top].mean()),
        "precision_abnormal_at_m": float(abnormal[top].mean()),
        "pathological_recall_at_m": float(path[top].sum() / max(1, path.sum())),
        "abnormal_recall_at_m": float(abnormal[top].sum() / max(1, abnormal.sum())),
    }


def load_ctg(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    frame = pd.read_excel(path, sheet_name="Data", skipfooter=3, engine="xlrd")
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if "NSP" in numeric.columns:
        target = numeric["NSP"]
        candidates = [c for c in numeric.columns if c not in {"CLASS", "NSP"}]
        features = numeric[candidates]
    else:
        numeric = numeric.iloc[:, 1:24]
        target = numeric.iloc[:, -1]
        features = numeric.iloc[:, :-2]
    valid = target.isin([1, 2, 3]) & features.notna().all(axis=1)
    features = features.loc[valid]
    target = target.loc[valid]
    if len(features) != 2126:
        raise RuntimeError(f"Expected 2126 CTG cases, obtained {len(features)}")
    y = target.to_numpy(dtype=int)
    counts = dict(zip(*np.unique(y, return_counts=True)))
    if counts != {1: 1655, 2: 295, 3: 176}:
        raise RuntimeError(f"Unexpected CTG class counts: {counts}")
    return features.to_numpy(dtype=float), y.astype(float), [str(c) for c in features.columns]


def load_mammography(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = np.load(path, allow_pickle=False)
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"], dtype=int).reshape(-1)
    if set(np.unique(y)) != {0, 1}:
        raise RuntimeError("Mammography labels must be binary")
    return X, y.astype(float), [f"x{i+1}" for i in range(X.shape[1])]


def hbos_score(X: np.ndarray, bins: int = 20) -> np.ndarray:
    n, d = X.shape
    score = np.zeros(n, dtype=float)
    eps = 1e-12
    for j in range(d):
        x = X[:, j]
        lo, hi = float(np.min(x)), float(np.max(x))
        if not hi > lo:
            continue
        hist, edges = np.histogram(x, bins=bins, range=(lo, hi), density=True)
        idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(hist) - 1)
        score += -np.log(hist[idx] + eps)
    return score


def detector_scores(X_raw: np.ndarray) -> dict[str, np.ndarray]:
    X = StandardScaler().fit_transform(X_raw)
    Xr = RobustScaler().fit_transform(X_raw)
    n, d = X.shape
    k = min(35, max(10, int(round(math.sqrt(n)))))
    iso = IsolationForest(n_estimators=300, max_samples="auto", contamination="auto", random_state=20260908, n_jobs=-1).fit(X)
    s_iso = -iso.score_samples(X)
    lof = LocalOutlierFactor(n_neighbors=k, novelty=False, n_jobs=-1)
    lof.fit_predict(X)
    s_lof = -lof.negative_outlier_factor_
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X)
    dist, _ = nn.kneighbors(X)
    s_knn = dist[:, -1]
    n_components = max(1, min(d - 1, int(math.ceil(0.75 * d)))) if d > 1 else 1
    pca = PCA(n_components=n_components, random_state=20260908).fit(X)
    recon = pca.inverse_transform(pca.transform(X))
    s_pca = np.sum((X - recon) ** 2, axis=1)
    scores = {
        "Isolation forest": s_iso,
        "Local outlier factor": s_lof,
        "kNN distance": s_knn,
        "PCA reconstruction": s_pca,
        "HBOS": hbos_score(X, bins=20),
        "Robust diagonal distance": np.sum(np.abs(Xr), axis=1),
    }
    for name, score in scores.items():
        if len(score) != n or not np.isfinite(score).all():
            raise RuntimeError(f"Invalid detector scores for {name}")
    return scores


def aggregate_orders(base_scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = list(base_scores)
    n = len(next(iter(base_scores.values())))
    ranks = np.column_stack([rank_vector(stable_order_desc(base_scores[name])) for name in names]).astype(float)
    pct = ranks / n
    wins = np.zeros(n, dtype=float)
    for col in range(ranks.shape[1]):
        wins += n - ranks[:, col]
    return {
        "Borda mean rank": stable_order_desc(-pct.mean(axis=1)),
        "Median rank": stable_order_desc(-np.median(pct, axis=1)),
        "Reciprocal-rank fusion": stable_order_desc(np.sum(1.0 / (60.0 + ranks), axis=1)),
        "Rank product": stable_order_desc(-np.mean(np.log(ranks), axis=1)),
        "Pairwise win score": stable_order_desc(wins),
    }


def run_dataset(dataset: str, X: np.ndarray, y: np.ndarray, primary_m: int, sensitivity_m: list[int], pathological_value: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = detector_scores(X)
    aggregate = aggregate_orders(base)
    orders = {**{f"Base: {name}": stable_order_desc(score) for name, score in base.items()}, **aggregate}
    rows = []
    for method, order in orders.items():
        for m in sorted(set([primary_m, *sensitivity_m])):
            rows.append({"dataset": dataset, "method": method, "kind": "base" if method.startswith("Base:") else "aggregate", "m": m, "primary_depth": m == primary_m, **evaluate_order(order, y, m, pathological_value)})
    metrics_df = pd.DataFrame(rows)
    loo_rows = []
    for omitted in base:
        reduced = {k: v for k, v in base.items() if k != omitted}
        for method, order in aggregate_orders(reduced).items():
            loo_rows.append({"dataset": dataset, "omitted_detector": omitted, "method": method, "m": primary_m, **evaluate_order(order, y, primary_m, pathological_value)})
    loo_df = pd.DataFrame(loo_rows)
    primary = metrics_df[(metrics_df["kind"] == "aggregate") & (metrics_df["primary_depth"])].set_index("method")
    methods = sorted(primary.index)
    comp_rows = []
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            da, db = primary.loc[a], primary.loc[b]
            top_a, top_b = da["top_only_kendall_tau_b"], db["top_only_kendall_tau_b"]
            comp_rows.append({
                "dataset": dataset,
                "method_a": a,
                "method_b": b,
                "delta_trc": float(da["trc"] - db["trc"]),
                "delta_full_kendall": float(da["full_kendall_tau_b"] - db["full_kendall_tau_b"]),
                "delta_top_only_kendall": float(top_a - top_b) if np.isfinite(top_a) and np.isfinite(top_b) else float("nan"),
                "delta_ndcg": float(da["ndcg_at_m"] - db["ndcg_at_m"]),
                "delta_ap_pathological": float(da["ap_pathological"] - db["ap_pathological"]),
                "delta_precision_pathological_at_m": float(da["precision_pathological_at_m"] - db["precision_pathological_at_m"]),
                "top_only_tie_or_undefined_but_trc_diff": bool((not np.isfinite(top_a) or not np.isfinite(top_b) or abs(top_a - top_b) < 1e-12) and abs(da["trc"] - db["trc"]) > 1e-6),
                "full_kendall_near_tie_but_trc_diff": bool(abs(da["full_kendall_tau_b"] - db["full_kendall_tau_b"]) < 0.002 and abs(da["trc"] - db["trc"]) > 0.005),
            })
    return metrics_df, loo_df, pd.DataFrame(comp_rows)


def plot_primary(all_metrics: pd.DataFrame, output: Path) -> None:
    agg = all_metrics[(all_metrics["kind"] == "aggregate") & (all_metrics["primary_depth"])].copy()
    datasets = list(agg["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.1 * len(datasets), 4.8), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        g = agg[agg["dataset"] == dataset].sort_values("trc")
        y = np.arange(len(g))
        ax.barh(y, g["boundary_component_over_total"], label="Top-vs-rest component")
        ax.barh(y, g["within_component_over_total"], left=g["boundary_component_over_total"], label="Within-top component")
        ax.set_yticks(y, g["method"])
        ax.set_xlabel("Contribution to rank-outcome TRC")
        ax.set_title(dataset)
        ax.axvline(0, linewidth=0.8)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctg-xls", type=Path, required=True)
    parser.add_argument("--mammography-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    X_ctg, y_ctg, ctg_features = load_ctg(args.ctg_xls)
    n_ctg = len(y_ctg)
    primary_ctg = int(round(0.10 * n_ctg))
    ctg_metrics, ctg_loo, ctg_comp = run_dataset("UCI Cardiotocography (ordinal)", X_ctg, y_ctg, primary_ctg, [int(np.sum(y_ctg == 3)), int(round(0.20 * n_ctg))], 3.0)
    X_mam, y_mam, mam_features = load_mammography(args.mammography_npz)
    n_mam = len(y_mam)
    primary_mam = int(np.sum(y_mam == 1))
    mam_metrics, mam_loo, mam_comp = run_dataset("ADBench Mammography (binary stress test)", X_mam, y_mam, primary_mam, [max(10, primary_mam // 2), min(n_mam - 1, primary_mam * 2)], 1.0)
    metrics = pd.concat([ctg_metrics, mam_metrics], ignore_index=True)
    loo = pd.concat([ctg_loo, mam_loo], ignore_index=True)
    comparisons = pd.concat([ctg_comp, mam_comp], ignore_index=True)
    metrics.to_csv(args.output_dir / "triage_ra_metrics.csv", index=False)
    loo.to_csv(args.output_dir / "triage_ra_leave_one_detector_out.csv", index=False)
    comparisons.to_csv(args.output_dir / "triage_ra_pairwise_discrimination.csv", index=False)
    primary = metrics[(metrics["kind"] == "aggregate") & (metrics["primary_depth"])].copy()
    primary.to_csv(args.output_dir / "triage_ra_primary_summary.csv", index=False)
    align_rows = []
    for dataset, group in primary.groupby("dataset"):
        for metric in ["ndcg_at_m", "ap_pathological", "precision_pathological_at_m", "full_kendall_tau_b", "top_only_kendall_tau_b"]:
            valid = group[["trc", metric]].dropna()
            rho = spearmanr(valid["trc"], valid[metric]).statistic if len(valid) >= 3 else float("nan")
            align_rows.append({"dataset": dataset, "comparison_metric": metric, "spearman_with_trc": float(rho) if rho is not None else float("nan")})
    pd.DataFrame(align_rows).to_csv(args.output_dir / "triage_ra_metric_alignment.csv", index=False)
    plot_primary(metrics, args.output_dir / "triage_ra_candidate.png")
    summary = {
        "candidate": "Unsupervised detector rank aggregation for clinical high-risk triage",
        "primary_dataset": {"name": "UCI Cardiotocography", "n": n_ctg, "features": len(ctg_features), "outcome_levels": {"normal": int(np.sum(y_ctg == 1)), "suspect": int(np.sum(y_ctg == 2)), "pathological": int(np.sum(y_ctg == 3))}, "primary_m": primary_ctg, "primary_fraction": primary_ctg / n_ctg},
        "boundary_stress_test": {"name": "ADBench Mammography", "n": n_mam, "features": len(mam_features), "anomalies": int(np.sum(y_mam == 1)), "primary_m": primary_mam},
        "base_rankers": list(detector_scores(X_ctg).keys()),
        "aggregators": ["Borda mean rank", "Median rank", "Reciprocal-rank fusion", "Rank product", "Pairwise win score"],
        "primary_results": primary.to_dict(orient="records"),
        "discrimination_counts": {"top_only_tie_or_undefined_but_trc_diff": int(comparisons["top_only_tie_or_undefined_but_trc_diff"].sum()), "full_kendall_near_tie_but_trc_diff": int(comparisons["full_kendall_near_tie_but_trc_diff"].sum())},
        "interpretation": [
            "Cardiotocography supplies an ordered external outcome, so TRC assesses both ordering among high-risk cases and separation of the selected set from the remainder.",
            "The binary mammography stress test isolates the boundary-separation role when within-class ordering is not identified.",
            "Outcome labels are never supplied to detectors or aggregation rules; they are used only for evaluation and a prevalence-matched sensitivity depth.",
            "TRC is complementary to NDCG and average precision. It does not encode gain or discount weights, but it still requires application-defined depth m."
        ]
    }
    (args.output_dir / "triage_ra_candidate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert len(ctg_metrics) > 0 and len(mam_metrics) > 0
    assert metrics["trc"].between(-1, 1).all()
    assert metrics["oracle_normalized_trc"].between(-1.01, 1.01).all()
    assert set(primary["dataset"]) == {"UCI Cardiotocography (ordinal)", "ADBench Mammography (binary stress test)"}
    print(json.dumps({"status": "PASS", "metrics_rows": len(metrics), "primary_rows": len(primary), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
