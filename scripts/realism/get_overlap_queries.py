import json
import glob
import pandas as pd

MERGED_CSV = "unhelpful_data/results/merged_annotations.csv"
JSON_DIR = "unhelpful_data"
OUTPUT_JSON = "unhelpful_data/results/queries_in_jsons_with_realism.json"


df = pd.read_csv(MERGED_CSV)

realism_by_query = dict(zip(df["query"], df["realism"]))

csv_queries = set(realism_by_query.keys())

json_files = sorted(glob.glob(f"{JSON_DIR}/*.json"))
if not json_files:
    raise FileNotFoundError(f"No .json files found in {JSON_DIR}/")

all_json_queries = set()

for path in json_files:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if isinstance(item, dict):
            q = item.get("query")
            if q:
                all_json_queries.add(q)

final_queries = sorted(all_json_queries & csv_queries)

output = [{"query": q, "realism": realism_by_query[q]} for q in final_queries]

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"JSON files read: {len(json_files)}")
print(f"Unique queries across JSON files: {len(all_json_queries)}")
print(f"Queries also found in CSV: {len(final_queries)}")
print(f"Wrote: {OUTPUT_JSON}")
