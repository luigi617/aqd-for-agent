#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Set, Any, Dict, List

import pandas as pd


def load_queries_from_json(json_path: Path) -> Set[str]:
    """Load a JSON file that is expected to be a list of dicts with a 'query' key."""
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read JSON {json_path}: {e}") from e

    if not isinstance(data, list):
        raise ValueError(f"JSON {json_path} must be a list, got {type(data).__name__}")

    queries: Set[str] = set()
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            # Skip non-dicts (or raise if you prefer strict)
            continue
        q = item.get("query")
        if isinstance(q, str) and q.strip():
            queries.add(q.strip())

    return queries


def main(
    csv_path: str = "unhelpful_data/results/final_merged_results.csv",
    json_dir: str = "unhelpful_data",
    output_csv: str = "unhelpful_data/results/unhelpful_data_stats.csv",
) -> None:
    csv_path_p = Path(csv_path)
    json_dir_p = Path(json_dir)

    if not csv_path_p.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path_p.resolve()}")
    if not json_dir_p.exists() or not json_dir_p.is_dir():
        raise NotADirectoryError(f"JSON directory not found: {json_dir_p.resolve()}")

    # Load CSV
    df = pd.read_csv(csv_path_p)

    required_cols = {"query", "avg_score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    # Normalize types
    df["query"] = df["query"].astype(str).str.strip()
    df["avg_score"] = pd.to_numeric(df["avg_score"], errors="coerce")

    # (Optional) drop rows where avg_score is NaN
    df = df.dropna(subset=["avg_score"])

    # Pre-index by query for speed
    # (We still use isin, but this ensures query normalization is consistent.)
    all_csv_queries = set(df["query"].unique())

    results: List[Dict[str, Any]] = []

    json_files = sorted([p for p in json_dir_p.glob("*.json") if p.is_file()])
    if not json_files:
        print(f"No .json files found in {json_dir_p.resolve()}")
        return

    for jp in json_files:
        queries = load_queries_from_json(jp)

        # Only consider queries that actually exist in the CSV
        matched_queries = queries.intersection(all_csv_queries)

        subset = df[df["query"].isin(matched_queries)]
        n_rows = int(len(subset))
        n_unique_queries = int(subset["query"].nunique())

        if n_rows > 0:
            mean_val = float(subset["avg_score"].mean())
            std_val = float(subset["avg_score"].std(ddof=1)) if n_rows > 1 else float("nan")
        else:
            mean_val = float("nan")
            std_val = float("nan")

        results.append(
            {
                "json_file": jp.name,
                "json_queries_total": len(queries),
                "matched_queries_unique": n_unique_queries,
                "matched_rows": n_rows,
                "avg_score_mean": mean_val,
                "avg_score_std": std_val,
            }
        )

    out_df = pd.DataFrame(results).sort_values(
        by=["matched_rows", "matched_queries_unique"], ascending=[False, False]
    )

    # Print nicely
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(out_df.to_string(index=False))

    out_df.to_csv(output_csv, index=False)
    print(f"\nWrote summary to: {Path(output_csv).resolve()}")


if __name__ == "__main__":
    main()
