import pandas as pd
import glob
from dataset_config import DATA_DIR

# Load all CSV files from the directory
files = glob.glob(f"{DATA_DIR}/realism_annotations/*.csv")

dfs = [pd.read_csv(file) for file in files]

# Concatenate all files
combined = pd.concat(dfs)

# Group by the fixed columns
grouped = combined.groupby(
    ["idx", "query", "product_category"],
    as_index=False
)

# Majority vote function
def majority_vote(series):
    return series.value_counts().idxmax()

# Apply majority voting
final_df = grouped.agg({
    "realism": majority_vote,
    "relevance": majority_vote
})
# Save the result
final_df.to_csv(f"{DATA_DIR}/results/merged_annotations.csv", index=False)

print("Merged file saved as merged_annotations.csv")
