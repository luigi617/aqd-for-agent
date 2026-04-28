import json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from dataset_config import DATA_DIR

RESULTS_DIR = f"{DATA_DIR}/results"
ANNOTATIONS_DIR = f"{DATA_DIR}/realism_annotations"

# -------------------------
# Load main JSON file
# -------------------------
with open(f"{RESULTS_DIR}/_combined.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# -------------------------
# Extract base fields
# -------------------------
extracted = []
for item in data.get("results", {}).values():
    extracted.append({
        "query": item.get("query"),
        "file": item.get("file"),
        "avg_score": item.get("avg_score"),
        "scores": item.get("scores"),
        "reasons": item.get("reasons"),
    })

# Convert to DataFrame for easier merging
df_extracted = pd.DataFrame(extracted)

# -------------------------
# Load annotation CSVs
# -------------------------
luigi = pd.read_csv(f"{ANNOTATIONS_DIR}/annotate_data_luigi.csv")
omar  = pd.read_csv(f"{ANNOTATIONS_DIR}/annotate_data_omar.csv")
lucy  = pd.read_csv(f"{ANNOTATIONS_DIR}/annotate_data_lucy.csv")

# Keep only needed columns + rename
luigi = luigi[["query", "realism"]].rename(
    columns={"realism": "luigi_realism"}
)
omar = omar[["query", "realism"]].rename(
    columns={"realism": "omar_realism"}
)
lucy = lucy[["query", "realism"]].rename(
    columns={"realism": "lucy_realism"}
)

# -------------------------
# Merge on query
# -------------------------
df_final = df_extracted.merge(luigi, on="query", how="left")
df_final = df_final.merge(omar,  on="query", how="left")
df_final = df_final.merge(lucy,  on="query", how="left")

# -------------------------------------------------
# Expand LLM scores into separate columns
# -------------------------------------------------
df_final["gpt_score"] = df_final["scores"].apply(
    lambda x: x.get("gpt") if isinstance(x, dict) else np.nan
)
df_final["deepseek_score"] = df_final["scores"].apply(
    lambda x: x.get("deepseek") if isinstance(x, dict) else np.nan
)
df_final["gemini_score"] = df_final["scores"].apply(
    lambda x: x.get("gemini") if isinstance(x, dict) else np.nan
)

# -------------------------------------------------
# Compute majority vote (realism)
# -------------------------------------------------
def majority_vote(row):
    values = [row["luigi_realism"], row["omar_realism"], row["lucy_realism"]]
    values = [v for v in values if pd.notna(v)]
    if len(values) == 0:
        return np.nan
    # Majority vote; if tie, this returns one of the tied values (deterministic by set order not guaranteed)
    return max(set(values), key=values.count)

df_final["majority_realism"] = df_final.apply(majority_vote, axis=1)

# -------------------------------------------------
# Columns to analyze (include avg_score so we can compute its mean too)
# -------------------------------------------------
columns_to_test = [
    "luigi_realism",
    "omar_realism",
    "lucy_realism",
    "majority_realism",
    "gpt_score",
    "deepseek_score",
    "gemini_score",
    "avg_score"
]

# -------------------------------------------------
# Compute correlations vs avg_score + summary stats
# (Includes avg_score row with mean/std/count and self-correlation)
# -------------------------------------------------
results = []

for col in columns_to_test:
    # For stats, we need avg_score to exist; for avg_score itself, we just dropna on avg_score
    if col == "avg_score":
        temp = df_final[[col]].dropna()
        corr, pval = 1.0, 0.0
        mean_val = temp[col].mean() if len(temp) > 0 else np.nan
        std_val  = temp[col].std()  if len(temp) > 1 else np.nan
        count    = len(temp)
    else:
        temp = df_final[[col, "avg_score"]].dropna()
        if len(temp) > 1:
            corr, pval = spearmanr(temp[col], temp["avg_score"])
            mean_val = temp[col].mean()
            std_val  = temp[col].std()
        else:
            corr, pval, mean_val, std_val = np.nan, np.nan, np.nan, np.nan
        count = len(temp)

    results.append({
        "variable": col,
        "correlation_with_avg": corr,
        "p_value": pval,
        "mean": mean_val,
        "std": std_val,
        "count": count
    })

df_stats = pd.DataFrame(results)

print("\n=== Correlation Results (including avg_score mean) ===")
print(df_stats)

df_final.to_csv(f"{RESULTS_DIR}/final_merged_results.csv", index=False)
df_stats.to_csv(f"{RESULTS_DIR}/correlation_results.csv", index=False)
