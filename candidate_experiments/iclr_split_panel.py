#!/usr/bin/env python3
"""Split-panel peer-review experiment for the proportional-truncation CLT.

The experiment forms two disjoint two-review panels for every paper with at
least four valid ICLR reviews. The panels independently rank the same paper
universe. Agreement of the two top lists is evaluated by bottom-tied
Kendall tau_b, and Theorem 3's random-ranking reference is calibrated by
Monte Carlo permutations at the observed n and truncation depths.

No manuscript files are modified by this script.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm


def bottom_tied_rank_vector(order: np.ndarray, m: int) -> np.ndarray:
    n = len(order)
    r = np.full(n, m + 1, dtype=np.int32)
    r[order[:m]] = np.arange(1, m + 1, dtype=np.int32)
    return r


def bottom_tied_tau(order_a: np.ndarray, order_b: np.ndarray, m1: int, m2: int) -> float:
    ra = bottom_tied_rank_vector(order_a, m1)
    rb = bottom_tied_rank_vector(order_b, m2)
    stat = kendalltau(ra, rb, variant="b").statistic
    return float(stat) if stat is not None else float("nan")


def theorem3_variance(gamma1: float, gamma2: float) -> float:
    if not (0 < gamma1 <= 1 and 0 < gamma2 <= 1):
        raise ValueError("gamma values must be in (0,1]")
    c = lambda g: g - g * g / 2.0
    v = lambda g: g - g * g + g**3 / 3.0
    return v(gamma1) * v(gamma2) / (c(gamma1) * c(gamma2))


def parse_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, (int, float, np.integer, np.floating)):
        x = float(value)
        return x if math.isfinite(x) else None
    text = str(value).strip()
    if not text:
        return None
    token = text.split(":", 1)[0].strip()
    try:
        x = float(token)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def hash_uint(*parts: object) -> int:
    text = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big", signed=False)


def read_reviews(path: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            paper = obj.get("paper_id") or obj.get("_paper_forum")
            review = obj.get("review_id") or obj.get("id")
            score = parse_number(obj.get("actual_score"))
            if score is None:
                content = obj.get("content") or {}
                score = parse_number(content.get("rating") if isinstance(content, dict) else None)
            confidence = parse_number(obj.get("confidence"))
            if confidence is None:
                content = obj.get("content") or {}
                confidence = parse_number(content.get("confidence") if isinstance(content, dict) else None)
            if paper is None or review is None or score is None:
                continue
            records.append({
                "paper_id": str(paper),
                "review_id": str(review),
                "score": float(score),
                "confidence": 0.0 if confidence is None else float(confidence),
            })
    out = pd.DataFrame.from_records(records)
    if out.empty:
        raise RuntimeError(f"No valid reviews parsed from {path}")
    return out.drop_duplicates(["paper_id", "review_id"], keep="first")


def panel_rankings(reviews: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for paper_id, group in reviews.groupby("paper_id", sort=True):
        if len(group) < 4:
            continue
        g = group.copy()
        g["split_key"] = [hash_uint("split", seed, paper_id, rid) for rid in g["review_id"]]
        g = g.sort_values(["split_key", "review_id"], kind="mergesort").iloc[:4]
        a = g.iloc[:2]
        b = g.iloc[2:4]
        rows.append({
            "paper_id": paper_id,
            "score_a": float(a["score"].mean()),
            "score_b": float(b["score"].mean()),
            "confidence_a": float(a["confidence"].mean()),
            "confidence_b": float(b["confidence"].mean()),
            "tie_a": hash_uint("tie-a", seed, paper_id),
            "tie_b": hash_uint("tie-b", seed, paper_id),
        })
    panel = pd.DataFrame.from_records(rows).sort_values("paper_id", kind="mergesort").reset_index(drop=True)
    if len(panel) < 100:
        raise RuntimeError(f"Only {len(panel)} papers have >=4 valid reviews")
    order_a = np.lexsort((panel["tie_a"].to_numpy(), -panel["confidence_a"].to_numpy(), -panel["score_a"].to_numpy()))
    order_b = np.lexsort((panel["tie_b"].to_numpy(), -panel["confidence_b"].to_numpy(), -panel["score_b"].to_numpy()))
    return order_a.astype(int), order_b.astype(int), panel


def rank_vector(order: np.ndarray) -> np.ndarray:
    r = np.empty(len(order), dtype=np.int32)
    r[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return r


def overlap_stats(order_a: np.ndarray, order_b: np.ndarray, m1: int, m2: int) -> dict[str, float]:
    a = set(order_a[:m1].tolist())
    b = set(order_b[:m2].tolist())
    ov = len(a & b)
    return {"overlap": ov, "overlap_over_min_depth": ov / min(m1, m2), "jaccard": ov / len(a | b)}


def pair_tie_fraction(values: np.ndarray) -> float:
    _, counts = np.unique(values, return_counts=True)
    tied = np.sum(counts * (counts - 1) // 2)
    total = len(values) * (len(values) - 1) // 2
    return float(tied / total) if total else 0.0


def run_observed(year: int, reviews: pd.DataFrame, seeds: int, depth_pairs: list[tuple[float, float]]) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    for seed in range(seeds):
        order_a, order_b, panel = panel_rankings(reviews, seed)
        n = len(panel)
        full_tau = float(kendalltau(rank_vector(order_a), rank_vector(order_b), variant="b").statistic)
        for gamma1, gamma2 in depth_pairs:
            m1 = max(1, min(n, int(round(gamma1 * n))))
            m2 = max(1, min(n, int(round(gamma2 * n))))
            tau = bottom_tied_tau(order_a, order_b, m1, m2)
            variance = theorem3_variance(m1 / n, m2 / n)
            z = math.sqrt(n) * tau / math.sqrt(variance)
            out.append({
                "year": year,
                "seed": seed,
                "n_papers": n,
                "m1": m1,
                "m2": m2,
                "gamma1_realized": m1 / n,
                "gamma2_realized": m2 / n,
                "bottom_tied_tau_b": tau,
                "theorem3_z": z,
                "theorem3_p_two_sided": 2.0 * norm.sf(abs(z)),
                "full_kendall_tau_b": full_tau,
                "score_tie_fraction_a": pair_tie_fraction(panel["score_a"].to_numpy()),
                "score_tie_fraction_b": pair_tie_fraction(panel["score_b"].to_numpy()),
                **overlap_stats(order_a, order_b, m1, m2),
            })
    return pd.DataFrame(out)


def run_null_calibration(year: int, n: int, depth_pairs: list[tuple[float, float]], reps: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    identity = np.arange(n, dtype=int)
    for gamma1, gamma2 in depth_pairs:
        m1 = max(1, min(n, int(round(gamma1 * n))))
        m2 = max(1, min(n, int(round(gamma2 * n))))
        variance = theorem3_variance(m1 / n, m2 / n)
        zs = np.empty(reps, dtype=float)
        taus = np.empty(reps, dtype=float)
        for b in range(reps):
            perm = rng.permutation(n)
            tau = bottom_tied_tau(identity, perm, m1, m2)
            z = math.sqrt(n) * tau / math.sqrt(variance)
            taus[b] = tau
            zs[b] = z
            detail.append({"year": year, "replicate": b, "n_papers": n, "m1": m1, "m2": m2, "gamma1_realized": m1 / n, "gamma2_realized": m2 / n, "tau_b": tau, "z": z})
        summary.append({
            "year": year,
            "n_papers": n,
            "m1": m1,
            "m2": m2,
            "gamma1_realized": m1 / n,
            "gamma2_realized": m2 / n,
            "reps": reps,
            "tau_mean": float(np.mean(taus)),
            "tau_sd_empirical": float(np.std(taus, ddof=1)),
            "tau_se_theory": math.sqrt(variance / n),
            "sd_ratio_empirical_to_theory": float(np.std(taus, ddof=1) / math.sqrt(variance / n)),
            "z_mean": float(np.mean(zs)),
            "z_sd": float(np.std(zs, ddof=1)),
            "tail_abs_gt_1_96": float(np.mean(np.abs(zs) > 1.96)),
            "z_q025": float(np.quantile(zs, 0.025)),
            "z_q975": float(np.quantile(zs, 0.975)),
        })
    return pd.DataFrame(detail), pd.DataFrame(summary)


def plot_results(observed: pd.DataFrame, calibration: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    primary = observed[(observed["m1"] / observed["n_papers"] > 0.28) & (observed["m1"] / observed["n_papers"] < 0.32) & (observed["m1"] == observed["m2"])]
    for year, g in primary.groupby("year"):
        axes[0].hist(g["bottom_tied_tau_b"], bins=12, alpha=0.55, label=str(year))
    axes[0].set_xlabel(r"Bottom-tied Kendall $\tau_b$ at $\gamma=0.30$")
    axes[0].set_ylabel("Split-panel replicates")
    axes[0].set_title("Observed split-panel agreement")
    axes[0].legend(frameon=False)
    x = np.arange(len(calibration))
    axes[1].errorbar(x, calibration["z_sd"], fmt="o", label="Empirical SD of Z")
    axes[1].axhline(1.0, linewidth=1, linestyle="--", label="N(0,1) target")
    labels = [f"{int(y)}\n{m1}/{m2}" for y, m1, m2 in zip(calibration["year"], calibration["m1"], calibration["m2"])]
    axes[1].set_xticks(x, labels, rotation=45, ha="right")
    axes[1].set_ylabel("Standard deviation")
    axes[1].set_title("Theorem 3 null calibration")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--null-reps", type=int, default=1000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    depth_pairs = [(0.10, 0.10), (0.20, 0.20), (0.30, 0.30), (0.20, 0.30), (0.30, 0.20)]
    observed_parts = []
    calibration_parts = []
    calibration_detail_parts = []
    data_summary: dict[str, object] = {}
    rng = np.random.default_rng(20260908)
    for year in (2023, 2024):
        reviews = read_reviews(args.input_root / f"ICLR_{year}" / "reviews.jsonl.gz")
        counts = reviews.groupby("paper_id").size()
        data_summary[str(year)] = {
            "valid_reviews": int(len(reviews)),
            "papers_with_valid_reviews": int(counts.size),
            "papers_with_at_least_four_reviews": int((counts >= 4).sum()),
            "median_valid_reviews_per_paper": float(counts.median()),
        }
        obs = run_observed(year, reviews, args.seeds, depth_pairs)
        observed_parts.append(obs)
        n = int(obs.loc[obs["seed"] == 0, "n_papers"].iloc[0])
        cal_detail, cal_summary = run_null_calibration(year, n, [(0.10, 0.10), (0.30, 0.30), (0.20, 0.30)], args.null_reps, rng)
        calibration_detail_parts.append(cal_detail)
        calibration_parts.append(cal_summary)
    observed = pd.concat(observed_parts, ignore_index=True)
    calibration = pd.concat(calibration_parts, ignore_index=True)
    calibration_detail = pd.concat(calibration_detail_parts, ignore_index=True)
    observed.to_csv(args.output_dir / "iclr_split_panel_observed.csv", index=False)
    calibration.to_csv(args.output_dir / "iclr_theorem3_calibration.csv", index=False)
    calibration_detail.to_csv(args.output_dir / "iclr_theorem3_calibration_replicates.csv", index=False)
    aggregate = observed.groupby(["year", "m1", "m2", "gamma1_realized", "gamma2_realized"], as_index=False).agg(
        n_papers=("n_papers", "first"), tau_mean=("bottom_tied_tau_b", "mean"), tau_sd=("bottom_tied_tau_b", "std"), tau_min=("bottom_tied_tau_b", "min"), tau_max=("bottom_tied_tau_b", "max"), full_tau_mean=("full_kendall_tau_b", "mean"), overlap_fraction_mean=("overlap_over_min_depth", "mean"), jaccard_mean=("jaccard", "mean"), theorem3_z_mean=("theorem3_z", "mean"))
    aggregate.to_csv(args.output_dir / "iclr_split_panel_summary.csv", index=False)
    primary = aggregate[(np.isclose(aggregate["gamma1_realized"], 0.30, atol=0.01)) & (aggregate["m1"] == aggregate["m2"])]
    decision = {
        "candidate": "ICLR split-reviewer panels",
        "years": [2023, 2024],
        "design": "two disjoint two-review panels per paper, repeated deterministic random splits",
        "primary_gamma": [0.30, 0.30],
        "data_summary": data_summary,
        "primary_results": primary.to_dict(orient="records"),
        "calibration": calibration.to_dict(orient="records"),
        "assumption_notes": [
            "Papers are closer to exchangeable item units than genes or molecular features, but topic and reviewer-calibration heterogeneity remain.",
            "Panel scores are discrete. Panel-specific independent tie breaking is used and split-panel variability is reported.",
            "The analytic reference is a conditional random-ranking benchmark, not a model of the full peer-review assignment process.",
            "Both panels evaluate the same paper after the same submission and rebuttal process; conditional independence is approximate rather than literal."
        ]
    }
    (args.output_dir / "iclr_candidate_summary.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    plot_results(observed, calibration, args.output_dir / "iclr_split_panel_candidate.png")
    assert set(observed["year"]) == {2023, 2024}
    assert observed["bottom_tied_tau_b"].between(-1, 1).all()
    assert calibration["z_sd"].between(0.75, 1.25).all(), calibration
    assert calibration["tail_abs_gt_1_96"].between(0.02, 0.09).all(), calibration
    print(json.dumps({"status": "PASS", "rows": len(observed), "calibration_rows": len(calibration), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
