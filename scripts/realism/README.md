# Realism Scripts

All scripts in this folder read the dataset directory from:

- dataset_config.py

To switch datasets, edit:

```python
DATA_DIR = "gemma_unhelpful_data"
```

Then run the scripts normally from the repo root

## Recommended Usage

### 1. Full scoring pipeline

Run the LLM scoring pass:

```bash
python3 scripts/realism/realism.py
```

This writes one result file per input JSON plus `DATA_DIR/results/_combined.json`.

Then merge with human annotations and compute correlations:

```bash
python3 scripts/realism/extract_result.py
```

This writes `DATA_DIR/results/final_merged_results.csv` and `DATA_DIR/results/correlation_results.csv`.

Then summarize by source file:

```bash
python3 scripts/realism/stats_per_method.py
```

This writes `DATA_DIR/results/<dataset>_stats.csv`.

### 2. Overlap pipeline

Create the majority-vote annotation file:

```bash
python3 scripts/realism/majority_annotatation.py
```

Extract the subset of annotated queries that also exist in the JSON inputs:

```bash
python3 scripts/realism/get_overlap_queries.py
```

If the overlap file is non-empty, score it with the three LLMs:

```bash
python3 scripts/realism/overlap_query_llm_judge.py
```

Then print the correlations:

```bash
python3 scripts/realism/overlap_query_correlation.py
```

Optional count-only utility:

```bash
python3 scripts/realism/count.py
```

## What Each Output Contains

- `_combined.json`: all judged rows plus provider metadata and file progress
- `final_merged_results.csv`: judged rows merged with human labels and per-model scores
- `correlation_results.csv`: correlation and summary statistics versus `avg_score`
- `merged_annotations.csv`: majority-vote realism and relevance labels
- `queries_in_jsons_with_realism.json`: overlap list in the format `[{ "query": ..., "realism": ... }]`
- `queries_with_llm_realism.json`: overlap list with human realism, LLM average realism, and per-model scores

## Known Caveats

These are important if you want reproducible or publication-quality results.

1. `overlap_query_llm_judge.py` has `DEV = True` by default.
   It still processes the full file when `DEV_MAX_ITEMS` is `None`, but it uses the smaller dev models (`gpt-5-nano`, `deepseek-chat`, `gemini-2.0-flash-lite`) instead of the stronger models used in `realism.py`.

2. `majority_annotatation.py` and `extract_result.py` do not handle 3-way annotation ties explicitly.
   If Luigi, Omar, and Lucy all pick different labels, the chosen "majority" is arbitrary instead of being marked as a tie or unresolved item.

3. `get_overlap_queries.py` matches only on exact query string.
   It does not normalize casing, punctuation, or whitespace, and if the same query appears more than once in the CSV it keeps only the last realism value because it builds a plain `dict`.

4. `extract_result.py` merges on `query` only.
   If the same query appears in multiple categories or multiple rows, the merge can duplicate rows or attach human labels ambiguously.

5. In the current checked-in workspace, the overlap file for `unhelpful_data` is empty.
   If `DATA_DIR = "unhelpful_data"`, then `unhelpful_data/results/queries_in_jsons_with_realism.json` currently contains `[]`, so the overlap-scoring pipeline cannot proceed unless the source data changes or the matching logic is widened.

6. The LLM scripts require `aiohttp` and `aiofiles`, but those packages are not listed in `pyproject.toml`.