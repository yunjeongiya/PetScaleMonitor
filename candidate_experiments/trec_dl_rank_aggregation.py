#!/usr/bin/env python3
"""Evaluate rank-outcome TRC on TREC Deep Learning passage-run aggregation.

The script uses official submitted run files as rank inputs and official graded
qrels only for evaluation.  It restricts the primary analysis universe to
judged passages, so unjudged documents are never silently coded as irrelevant.
It also performs three deterministic stress tests: tail-only permutations,
within-top permutations, and top-boundary swaps.
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
from scipy.stats import kendalltau


def stable_order_desc(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    if not np.isfinite(score).all():
        raise ValueError("non-finite score")
    return np.lexsort((np.arange(len(score)), -score)).astype(int)


def rank_vector(order: np.ndarray) -> np.ndarray:
    out = np.empty(len(order), dtype=np.int32)
    out[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return out


def trc_components(order: np.ndarray, outcome: np.ndarray, m: int) -> dict[str, float]:
    y = np.asarray(outcome, dtype=float)[order]
    n = len(y)
    m = int(max(1, min(m, n - 1)))
    within = 0.0
    boundary = 0.0
    for i in range(m):
        if i + 1 < m:
            within += float(np.sign(y[i] - y[i + 1:m]).sum())
        boundary += float(np.sign(y[i] - y[m:]).sum())
    d_within = m * (m - 1) // 2
    d_boundary = m * (n - m)
    d_total = d_within + d_boundary
    return {
        "trc": (within + boundary) / d_total,
        "within_component_over_total": within / d_total,
        "boundary_component_over_total": boundary / d_total,
        "within_pair_average": within / d_within if d_within else float("nan"),
        "boundary_pair_average": boundary / d_boundary if d_boundary else float("nan"),
        "within_numerator": within,
        "boundary_numerator": boundary,
        "comparable_pairs": d_total,
    }


def dcg(rels: np.ndarray) -> float:
    rels = np.asarray(rels, dtype=float)
    if not len(rels):
        return 0.0
    gains = np.power(2.0, rels) - 1.0
    discounts = np.log2(np.arange(2, len(rels) + 2, dtype=float))
    return float(np.sum(gains / discounts))


def ndcg_at(order: np.ndarray, y: np.ndarray, m: int) -> float:
    m = min(m, len(order))
    num = dcg(y[order[:m]])
    den = dcg(np.sort(y)[::-1][:m])
    return num / den if den > 0 else float("nan")


def ap_at(order: np.ndarray, binary: np.ndarray, m: int) -> float:
    m = min(m, len(order))
    y = np.asarray(binary, dtype=int)[order[:m]]
    total = int(np.asarray(binary, dtype=int).sum())
    if total == 0:
        return float("nan")
    hit = 0
    s = 0.0
    for i, v in enumerate(y, 1):
        if v:
            hit += 1
            s += hit / i
    return s / min(total, m)


def evaluate(order: np.ndarray, y: np.ndarray, m: int) -> dict[str, float]:
    rank = rank_vector(order)
    priority = -rank.astype(float)
    comp = trc_components(order, y, m)
    full = kendalltau(priority, y, variant="b").statistic
    top = order[:m]
    top_tau = kendalltau(priority[top], y[top], variant="b").statistic
    high = (y >= 2).astype(int)
    anyrel = (y >= 1).astype(int)
    return {
        **comp,
        "full_kendall_tau_b": float(full) if np.isfinite(full) else float("nan"),
        "top_only_kendall_tau_b": float(top_tau) if np.isfinite(top_tau) else float("nan"),
        "ndcg_at_m": ndcg_at(order, y, m),
        "ap_high_at_m": ap_at(order, high, m),
        "ap_any_at_m": ap_at(order, anyrel, m),
        "precision_high_at_m": float(high[top].mean()),
        "precision_any_at_m": float(anyrel[top].mean()),
    }


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    with open_text(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, docid, rel = parts[0], parts[2], int(float(parts[3]))
            out.setdefault(qid, {})[docid] = rel
    if not out:
        raise RuntimeError(f"No qrels parsed from {path}")
    return out


def load_run(path: Path, top_depth: int = 1000) -> dict[str, list[str]]:
    rows: dict[str, list[tuple[int, str]]] = {}
    with open_text(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, docid = parts[0], parts[2]
            try:
                rank = int(float(parts[3]))
            except ValueError:
                continue
            if rank <= top_depth:
                rows.setdefault(qid, []).append((rank, docid))
    return {q: [d for _, d in sorted(v)] for q, v in rows.items()}


def query_rank_matrix(
    docs: list[str], runs: dict[str, dict[str, list[str]]], qid: str
) -> tuple[np.ndarray, list[str]]:
    index = {d: i for i, d in enumerate(docs)}
    cols = []
    names = []
    n = len(docs)
    for name, run in runs.items():
        order = []
        seen = set()
        for doc in run.get(qid, []):
            if doc in index and doc not in seen:
                order.append(index[doc])
                seen.add(doc)
        if len(order) < min(5, max(2, n // 10)):
            continue
        ranks = np.full(n, n + 1.0)
        for r, i in enumerate(order, 1):
            ranks[i] = r
        cols.append(ranks)
        names.append(name)
    if len(cols) < 3:
        return np.empty((n, 0)), []
    return np.column_stack(cols), names


def aggregate_orders(ranks: np.ndarray) -> dict[str, np.ndarray]:
    n, k = ranks.shape
    pct = ranks / (n + 1.0)
    listed = ranks <= n
    borda = -pct.mean(axis=1)
    rrf = np.sum(np.where(listed, 1.0 / (60.0 + ranks), 0.0), axis=1)
    median = -np.median(pct, axis=1)
    rankprod = -np.mean(np.log(np.maximum(ranks, 1.0)), axis=1)
    best = np.min(pct, axis=1)
    mean = np.mean(pct, axis=1)
    return {
        "Borda mean rank": stable_order_desc(borda),
        "Reciprocal-rank fusion": stable_order_desc(rrf),
        "Median rank": stable_order_desc(median),
        "Rank product": stable_order_desc(rankprod),
        "Best-rank fusion": np.lexsort((np.arange(n), mean, best)).astype(int),
    }


def stress_tail(order: np.ndarray, y: np.ndarray, m: int, reps: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    base = evaluate(order, y, m)
    trcs = []
    full = []
    within = []
    for _ in range(reps):
        p = order.copy()
        p[m:] = rng.permutation(p[m:])
        e = evaluate(p, y, m)
        trcs.append(e["trc"])
        full.append(e["full_kendall_tau_b"])
        within.append(e["top_only_kendall_tau_b"])
    return {
        "base_trc": base["trc"],
        "tail_trc_max_abs_change": float(np.max(np.abs(np.asarray(trcs) - base["trc"]))),
        "tail_full_kendall_sd": float(np.nanstd(full, ddof=1)),
        "tail_top_only_max_abs_change": float(np.nanmax(np.abs(np.asarray(within) - base["top_only_kendall_tau_b"]))) if np.isfinite(base["top_only_kendall_tau_b"]) else float("nan"),
    }


def stress_within(order: np.ndarray, y: np.ndarray, m: int, reps: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    base = evaluate(order, y, m)["trc"]
    vals = []
    for _ in range(reps):
        p = order.copy()
        p[:m] = rng.permutation(p[:m])
        vals.append(evaluate(p, y, m)["trc"])
    vals = np.asarray(vals)
    return {
        "base_trc": base,
        "within_permutation_trc_sd": float(np.std(vals, ddof=1)),
        "within_permutation_mean_abs_change": float(np.mean(np.abs(vals - base))),
    }


def beneficial_boundary_swap(order: np.ndarray, y: np.ndarray, m: int) -> dict[str, float]:
    base = evaluate(order, y, m)
    top = order[:m]
    rest = order[m:]
    i_top = int(np.argmin(y[top]))
    i_rest = int(np.argmax(y[rest]))
    p = order.copy()
    p[i_top], p[m + i_rest] = p[m + i_rest], p[i_top]
    after = evaluate(p, y, m)
    return {
        "base_trc": base["trc"],
        "swapped_trc": after["trc"],
        "delta_trc": after["trc"] - base["trc"],
        "delta_boundary_component": after["boundary_component_over_total"] - base["boundary_component_over_total"],
        "delta_within_component": after["within_component_over_total"] - base["within_component_over_total"],
        "top_relevance_removed": float(y[top[i_top]]),
        "rest_relevance_inserted": float(y[rest[i_rest]]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", required=True)
    ap.add_argument("--qrels", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--tail-reps", type=int, default=200)
    ap.add_argument("--within-reps", type=int, default=200)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    qrels = load_qrels(args.qrels)
    run_paths = sorted([p for p in args.run_dir.iterdir() if p.is_file()])
    runs = {}
    for p in run_paths:
        try:
            r = load_run(p)
        except (OSError, EOFError):
            continue
        overlap = len(set(r) & set(qrels))
        if overlap >= 5:
            runs[p.name] = r
    if len(runs) < 3:
        raise RuntimeError(f"Only {len(runs)} qualifying runs")

    rows = []
    stress_rows = []
    skipped = {}
    for qid, judgments in sorted(qrels.items()):
        docs = sorted(judgments)
        y = np.asarray([judgments[d] for d in docs], dtype=float)
        if len(docs) < max(20, args.m + 5) or len(np.unique(y)) < 2:
            skipped[qid] = "insufficient judged universe or relevance variation"
            continue
        matrix, used = query_rank_matrix(docs, runs, qid)
        if matrix.shape[1] < 3:
            skipped[qid] = "fewer than three qualifying system rankings"
            continue
        orders = aggregate_orders(matrix)
        for method, order in orders.items():
            e = evaluate(order, y, args.m)
            rows.append({
                "year": args.year,
                "qid": qid,
                "method": method,
                "n_judged": len(docs),
                "n_input_runs": matrix.shape[1],
                "m": min(args.m, len(docs) - 1),
                **e,
            })
        # Stress tests on RRF, a prespecified common fusion baseline.
        order = orders["Reciprocal-rank fusion"]
        seed = int(hashlib.sha256(f"{args.year}|{qid}".encode()).hexdigest()[:8], 16)
        stress_rows.append({
            "year": args.year,
            "qid": qid,
            "n_judged": len(docs),
            "n_input_runs": matrix.shape[1],
            "m": min(args.m, len(docs) - 1),
            **stress_tail(order, y, args.m, args.tail_reps, seed),
            **{k: v for k, v in stress_within(order, y, args.m, args.within_reps, seed + 1).items() if k != "base_trc"},
            **{f"swap_{k}": v for k, v in beneficial_boundary_swap(order, y, args.m).items() if k != "base_trc"},
        })

    df = pd.DataFrame(rows)
    stress = pd.DataFrame(stress_rows)
    if df.empty:
        raise RuntimeError("No eligible queries")
    df.to_csv(args.output_dir / "query_method_metrics.csv", index=False)
    stress.to_csv(args.output_dir / "tail_boundary_stress.csv", index=False)

    metrics = [
        "trc", "within_component_over_total", "boundary_component_over_total",
        "within_pair_average", "boundary_pair_average", "full_kendall_tau_b",
        "top_only_kendall_tau_b", "ndcg_at_m", "ap_high_at_m", "ap_any_at_m",
        "precision_high_at_m", "precision_any_at_m",
    ]
    macro = df.groupby("method")[metrics].agg(["mean", "std", "median", "count"])
    macro.to_csv(args.output_dir / "macro_summary.csv")

    methods = sorted(df.method.unique())
    pair_rows = []
    for qid, g in df.groupby("qid"):
        g = g.set_index("method")
        for i, a in enumerate(methods):
            for b in methods[i + 1:]:
                if a not in g.index or b not in g.index:
                    continue
                ra, rb = g.loc[a], g.loc[b]
                ta, tb = ra.top_only_kendall_tau_b, rb.top_only_kendall_tau_b
                pair_rows.append({
                    "year": args.year, "qid": qid, "method_a": a, "method_b": b,
                    "delta_trc": ra.trc - rb.trc,
                    "delta_full_kendall": ra.full_kendall_tau_b - rb.full_kendall_tau_b,
                    "delta_top_only_kendall": ta - tb if np.isfinite(ta) and np.isfinite(tb) else float("nan"),
                    "delta_ndcg": ra.ndcg_at_m - rb.ndcg_at_m,
                    "top_only_tie_or_undefined_but_trc_diff": bool((not np.isfinite(ta) or not np.isfinite(tb) or abs(ta - tb) < 1e-12) and abs(ra.trc - rb.trc) > 1e-6),
                    "full_near_tie_but_trc_diff": bool(abs(ra.full_kendall_tau_b - rb.full_kendall_tau_b) < 0.005 and abs(ra.trc - rb.trc) > 0.01),
                    "trc_ndcg_same_direction": bool((ra.trc - rb.trc) * (ra.ndcg_at_m - rb.ndcg_at_m) > 0),
                })
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(args.output_dir / "pairwise_discrimination.csv", index=False)

    # Figure: macro metric comparison and stress-test invariance.
    order_methods = df.groupby("method").trc.mean().sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    g = df.groupby("method")[["trc", "full_kendall_tau_b", "top_only_kendall_tau_b"]].mean().loc[order_methods]
    yloc = np.arange(len(g))
    for col, marker in zip(g.columns, ["o", "s", "^"]):
        axes[0].plot(g[col], yloc, marker=marker, linestyle="none", label=col)
    axes[0].set_yticks(yloc, g.index)
    axes[0].axvline(0, linewidth=0.8)
    axes[0].set_xlabel("Mean coefficient across queries")
    axes[0].set_title(f"TREC DL {args.year}: aggregate-ranking evaluation")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].scatter(stress.tail_full_kendall_sd, stress.tail_trc_max_abs_change, alpha=0.7)
    axes[1].set_xlabel("SD of full Kendall under tail-only permutations")
    axes[1].set_ylabel("Maximum absolute TRC change")
    axes[1].set_title("Tail-noise stress test (RRF)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "trec_dl_candidate.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "year": args.year,
        "qrels": str(args.qrels),
        "run_dir": str(args.run_dir),
        "qualifying_run_count": len(runs),
        "qualifying_run_names": sorted(runs),
        "eligible_query_count": int(df.qid.nunique()),
        "skipped_queries": skipped,
        "primary_universe": "judged passages only",
        "labels_used_in_aggregation": False,
        "tail_permutation_property": "TRC and top-only Kendall are invariant when only ranks below m are permuted",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", **{k: meta[k] for k in ["year", "qualifying_run_count", "eligible_query_count"]}}))


if __name__ == "__main__":
    main()
