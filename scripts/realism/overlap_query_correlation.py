import json
import pandas as pd
from dataset_config import DATA_DIR

file_path = f"{DATA_DIR}/results/queries_with_llm_realism.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)


df["gpt_score"] = df["per_model_scores"].apply(
    lambda x: x.get("gpt") if isinstance(x, dict) else None
)

df["deepseek_score"] = df["per_model_scores"].apply(
    lambda x: x.get("deepseek") if isinstance(x, dict) else None
)

df["gemini_score"] = df["per_model_scores"].apply(
    lambda x: x.get("gemini") if isinstance(x, dict) else None
)

cols = [
    "realism",
    "llm_realism",
    "gpt_score",
    "deepseek_score",
    "gemini_score",
]

for c in cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")


def spearman_vs_realism(column_name):
    df_clean = df.dropna(subset=["realism", column_name])
    if len(df_clean) < 2:
        return None, len(df_clean)
    corr = df_clean["realism"].corr(df_clean[column_name], method="spearman")
    return corr, len(df_clean)


results = {}

for col in ["llm_realism", "gpt_score", "deepseek_score", "gemini_score"]:
    corr, n = spearman_vs_realism(col)
    results[col] = {"n": n, "spearman": corr}

# ---- Print results ----
print("Spearman correlations vs realism:\n")
for k, v in results.items():
    print(f"{k}: n={v['n']}, spearman={v['spearman']}")
