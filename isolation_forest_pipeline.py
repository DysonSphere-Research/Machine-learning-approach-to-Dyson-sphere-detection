#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isolation Forest — Dyson-centric / Normal-centric with Weighted Fusion (single-run)

Usage (examples):
  # Run both regimes and save only the fused ranking with default weights (0.5)
  python isolation_forest_pipeline.py --data-dir /path/data --out-dir /path/out

  # Save dyson + normal + fused with multiple weights and metrics for Dyson as target
  python isolation_forest_pipeline.py --data-dir /path/data --out-dir /path/out \
      --emit dyson,normal,fused --fuse-weights 0.9,0.7,0.5,0.3 \
      --emit-metrics --metrics-target dyson

  # Save only dyson ranking
  python isolation_forest_pipeline.py --data-dir /path/data --out-dir /path/out --emit dyson

Inputs in --data-dir:
  - train.csv           (used when mode=dyson)     columns: source_id, <numeric features...>
  - trainNormal.csv     (used when mode=normal)    columns: source_id, <numeric features...>
  - test_normal.csv     (evaluation set)           columns: source_id, <numeric features...>
  - num_ds.txt          (first integer = number of Dyson spies at top of test)
  - numNorm.txt         (first integer = number of Normal spies after the Dyson block; used for metrics target=normal)

Outputs in --out-dir (examples):
  - isoforest_ranking_dyson.csv
  - isoforest_ranking_normal.csv
  - isoforest_ranking_fused_wDy0p70_wN0p30_minmax.csv
  - isoforest_metrics_dyson.csv
  - isoforest_metrics_normal.csv
  - isoforest_model_dyson.joblib
  - isoforest_model_normal.joblib
  - isoforest_dyson.log / isoforest_normal.log / isoforest_fused.log
"""

from __future__ import annotations
import argparse
import os
import re
import time
from datetime import timedelta
from typing import Optional, Tuple, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


# --------------------------- Logging ---------------------------

def _log(msg: str, log_file: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)


# ---------------------- File helpers --------------------------

_INT_RE = re.compile(r"(-?\d+)")

def _read_int(path: str, key_hint: Optional[str] = None) -> int:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    txt = open(path, "r", encoding="utf-8").read()
    m = _INT_RE.search(txt)
    if not m:
        raise ValueError(f"No integer found in {path}. Content: {txt!r}")
    return int(m.group(1))


def _require_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Present: {list(df.columns)}")


# ---------------------- Data loading --------------------------

def load_data(
    data_dir: str,
    train_dyson: str = "train.csv",
    train_normal: str = "trainNormal.csv",
    test_name: str = "test_normal.csv",
    num_ds_file: str = "num_ds.txt",
    num_norm_file: str = "numNorm.txt",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
    train_d = pd.read_csv(os.path.join(data_dir, train_dyson))
    train_n = pd.read_csv(os.path.join(data_dir, train_normal))
    test_df = pd.read_csv(os.path.join(data_dir, test_name))

    for df in (train_d, train_n, test_df):
        _require_columns(df, ["source_id"])

    num_ds = _read_int(os.path.join(data_dir, num_ds_file))
    num_norm = _read_int(os.path.join(data_dir, num_norm_file))

    return train_d, train_n, test_df, num_ds, num_norm


# -------------------- Spy masks --------------------------

def build_spy_labels(test_df: pd.DataFrame, num_ds: int, num_norm: int) -> pd.DataFrame:
    """
    Return DataFrame with spy labels for each test star:
      - is_dyson_spy
      - is_normal_spy
    Collapses duplicate IDs with logical OR so labels are unique per ID.
    """
    n = len(test_df)
    is_dyson = np.zeros(n, dtype=bool)
    is_normal = np.zeros(n, dtype=bool)

    if num_ds > 0:
        is_dyson[:num_ds] = True
    if num_norm > 0:
        start, stop = num_ds, min(n, num_ds + num_norm)
        is_normal[start:stop] = True

    labels = pd.DataFrame({
        "ID": test_df["source_id"].values,
        "is_dyson_spy": is_dyson,
        "is_normal_spy": is_normal,
    })
    # Deduplicate by ID to avoid one-to-many merges later
    labels = (labels.groupby("ID", as_index=False)
                    .agg(is_dyson_spy=("is_dyson_spy", "any"),
                         is_normal_spy=("is_normal_spy", "any")))
    return labels


# -------------------- Model & Scoring -------------------------

def fit_isolation_forest(
    X_train: pd.DataFrame,
    n_estimators: int = 100,
    max_samples: str | int | float = "auto",
    contamination: str | float = "auto",
    random_state: int = 42,
    n_jobs: Optional[int] = None,
) -> IsolationForest:
    X = X_train.drop(columns=["source_id"]).values
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X)
    return model


def score_rank(model: IsolationForest, X_test: pd.DataFrame) -> pd.DataFrame:
    X = X_test.drop(columns=["source_id"]).values
    scores = model.decision_function(X)
    out = pd.DataFrame({"ID": X_test["source_id"].values, "score": scores})
    out = out.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    return out


# -------------------- Normalization -------------------------

def normalize_series(s: pd.Series, scheme: str) -> pd.Series:
    if scheme == "none":
        return s
    if scheme == "zscore":
        mu, sigma = s.mean(), s.std(ddof=0)
        return (s - mu) / (sigma if sigma > 0 else 1.0)
    smin, smax = float(s.min()), float(s.max())
    if smax == smin:
        return pd.Series(np.zeros_like(s, dtype=float), index=s.index)
    return (s - smin) / (smax - smin)


def format_weight(x: float) -> str:
    return f"{x:.2f}".replace(".", "p")


# -------------------- Fusion & Metrics ------------------------

def fuse_rankings(
    dyson_rank: pd.DataFrame,
    normal_rank: pd.DataFrame,
    alpha_dyson: float,
    norm_scheme: str = "minmax",
) -> pd.DataFrame:
    df = pd.merge(dyson_rank.rename(columns={"score": "score_dyson"}),
                  normal_rank.rename(columns={"score": "score_normal"}),
                  on="ID", how="inner", validate="one_to_one")

    s_d = normalize_series(df["score_dyson"].astype(float), norm_scheme)
    s_n = normalize_series(df["score_normal"].astype(float), norm_scheme)
    fused = alpha_dyson * s_d + (1.0 - alpha_dyson) * (1.0 - s_n)

    fused_df = pd.DataFrame({"ID": df["ID"], "score": fused})
    fused_df = fused_df.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    return fused_df


def spy_mask(n_test: int, num_ds: int, num_norm: int, target: str) -> np.ndarray:
    mask = np.zeros(n_test, dtype=bool)
    if target == "dyson":
        mask[:max(0, num_ds)] = True
    else:
        start, stop = max(0, num_ds), min(n_test, num_ds + max(0, num_norm))
        mask[start:stop] = True
    return mask


def metrics_from_ranking(
    ranking: pd.DataFrame,
    test_df: pd.DataFrame,
    num_ds: int,
    num_norm: int,
    metrics_target: str,
) -> pd.DataFrame:
    """
    Compute cumulative Precision/Recall/F1 for each k (1..N) given the
    implicit spy mask over the original test order.

    Robust to duplicate source_id in test_df: maps each ID to the FIRST
    occurrence position in test_df.
    """
    # Build positives mask on the original test order
    pos_mask = spy_mask(len(test_df), num_ds, num_norm, metrics_target)

    # Map: ID -> first position in test order (handles duplicate IDs)
    # Option A (groupby-first)
    pos_map = (
        pd.Series(np.arange(len(test_df)), index=test_df["source_id"].values)
          .groupby(level=0)
          .first()
    )
    # In alternativa:
    # tmp = test_df.reset_index().rename(columns={"index": "pos"})
    # pos_map = tmp.drop_duplicates("source_id", keep="first").set_index("source_id")["pos"]

    # Keep only IDs that exist in the mapping
    rnk = ranking.copy()
    rnk["pos"] = rnk["ID"].map(pos_map)

    # Drop IDs that didn't map (NaN) and cast to int
    rnk = rnk.dropna(subset=["pos"]).copy()
    rnk["pos"] = rnk["pos"].astype(int)

    # True Positive at each rank position
    is_tp_ordered = pos_mask[rnk["pos"].values]
    cum_tp = np.cumsum(is_tp_ordered.astype(int))
    k = np.arange(1, len(rnk) + 1)
    denom_pos = int(pos_mask.sum())
    precision = cum_tp / k
    recall = cum_tp / (denom_pos if denom_pos > 0 else 1)
    f1 = (2 * precision * recall) / np.clip(precision + recall, 1e-12, None)

    return pd.DataFrame({"k": k, "precision": precision, "recall": recall, "f1": f1})



def ensure_unique_ids(ranking: pd.DataFrame, policy: str, label: str) -> pd.DataFrame:
    if ranking["ID"].is_unique:
        return ranking
    dup_cnt = ranking["ID"].duplicated().sum()
    print(f"[WARN] {label}: found {dup_cnt} duplicated IDs. Applying policy: {policy}")
    if policy == "error":
        raise ValueError(f"Duplicate IDs in {label} ranking.")
    if policy == "drop-keep-best":
        return ranking.sort_values("score", ascending=False).drop_duplicates("ID").reset_index(drop=True)
    if policy in {"mean", "max", "min"}:
        rk = ranking.groupby("ID", as_index=False).agg(score=("score", policy))
        return rk.sort_values("score", ascending=False).reset_index(drop=True)
    return ranking.drop_duplicates("ID").reset_index(drop=True)


# -------------------- Save utilities --------------------------

def save_ranking(out_dir: str, name: str, df: pd.DataFrame) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    df.to_csv(path, index=False)
    return path


def save_model(out_dir: str, mode: str, model: IsolationForest, meta: Dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"isoforest_model_{mode}.joblib")
    joblib.dump({"model": model, "meta": meta}, path)
    return path


# ------------------------------ Main --------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Isolation Forest (dyson | normal) with weighted fusion")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--emit", type=str, default="fused_only")
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--max-samples", type=str, default="auto")
    p.add_argument("--contamination", type=str, default="auto")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--fuse-weights", type=str, default="0.50")
    p.add_argument("--norm-scheme", type=str, choices=["minmax", "zscore", "none"], default="minmax")
    p.add_argument("--emit-metrics", action="store_true")
    p.add_argument("--metrics-target", type=str, choices=["dyson", "normal"], default="dyson")
    p.add_argument("--train-file-dyson", type=str, default="train.csv")
    p.add_argument("--train-file-normal", type=str, default="trainNormal.csv")
    p.add_argument("--test-file", type=str, default="test_normal.csv")
    p.add_argument("--num-ds-file", type=str, default="num_ds.txt")
    p.add_argument("--num-norm-file", type=str, default="numNorm.txt")
    p.add_argument("--on-duplicate", type=str, choices=["error", "drop-keep-best", "mean", "max", "min"],
                   default="drop-keep-best")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    emit = {e.strip().lower() for e in args.emit.split(",")}
    if emit == {"fused_only"}:
        emit = {"fused"}

    t0 = time.time()
    train_d, train_n, test_df, num_ds, num_norm = load_data(
        args.data_dir, args.train_file_dyson, args.train_file_normal,
        args.test_file, args.num_ds_file, args.num_norm_file
    )
    spy_labels = build_spy_labels(test_df, num_ds, num_norm)

    logs = {m: os.path.join(args.out_dir, f"isoforest_{m}.log") for m in ["dyson", "normal", "fused"]}

    # Dyson model
    open(logs["dyson"], "w").write("Isolation Forest (dyson) — log\n")
    _log(f"Training dyson IF | train shape={train_d.shape}", logs["dyson"])
    model_d = fit_isolation_forest(train_d, args.n_estimators, args.max_samples,
                                   args.contamination, args.random_state, args.n_jobs)
    rank_d = score_rank(model_d, test_df)
    rank_d = ensure_unique_ids(rank_d, args.on_duplicate, "dyson")
    _log("Ranking computed (dyson).", logs["dyson"])

    # Normal model
    open(logs["normal"], "w").write("Isolation Forest (normal) — log\n")
    _log(f"Training normal IF | train shape={train_n.shape}", logs["normal"])
    model_n = fit_isolation_forest(train_n, args.n_estimators, args.max_samples,
                                   args.contamination, args.random_state, args.n_jobs)
    rank_n = score_rank(model_n, test_df)
    rank_n = ensure_unique_ids(rank_n, args.on_duplicate, "normal")
    _log("Ranking computed (normal).", logs["normal"])

    # Save Dyson ranking (do not overwrite rank_d used for fusion)
    if "dyson" in emit:
        rank_d_out = rank_d.merge(spy_labels[["ID", "is_dyson_spy"]], on="ID", how="left")
        save_ranking(args.out_dir, "isoforest_ranking_dyson.csv", rank_d_out)
        save_model(args.out_dir, "dyson", model_d, vars(args))
        _log("Saved dyson ranking & model.", logs["dyson"])

    # Save Normal ranking (do not overwrite rank_n used for fusion)
    if "normal" in emit:
        rank_n_out = rank_n.merge(spy_labels[["ID", "is_dyson_spy"]], on="ID", how="left")
        save_ranking(args.out_dir, "isoforest_ranking_normal.csv", rank_n_out)
        save_model(args.out_dir, "normal", model_n, vars(args))
        _log("Saved normal ranking & model.", logs["normal"])

    # Fused rankings
    if "fused" in emit:
        open(logs["fused"], "w").write("Isolation Forest (fused) — log\n")
        alphas = [float(x) for x in args.fuse_weights.split(",")]

        # Safety: ensure uniqueness right before merging
        rank_d_unique = ensure_unique_ids(rank_d, args.on_duplicate, "dyson")
        rank_n_unique = ensure_unique_ids(rank_n, args.on_duplicate, "normal")

        for a in alphas:
            fused_df = fuse_rankings(rank_d_unique, rank_n_unique, a, args.norm_scheme)
            fused_df = fused_df.merge(spy_labels, on="ID", how="left")  # adds is_dyson_spy and is_normal_spy
            name = f"isoforest_ranking_fused_wDy{format_weight(a)}_wN{format_weight(1.0-a)}_{args.norm_scheme}.csv"
            save_ranking(args.out_dir, name, fused_df)
            _log(f"Saved fused ranking: {name}", logs["fused"])
            if args.emit_metrics:
                m = metrics_from_ranking(fused_df, test_df, num_ds, num_norm, args.metrics_target)
                metrics_name = f"isoforest_metrics_{args.metrics_target}_wDy{format_weight(a)}_{args.norm_scheme}.csv"
                save_ranking(args.out_dir, metrics_name, m)
                _log(f"Saved metrics: {metrics_name}", logs["fused"])

    dt = timedelta(seconds=int(time.time() - t0))
    for k in logs:
        _log(f"Elapsed: {dt}", logs[k])


if __name__ == "__main__":
    main()
