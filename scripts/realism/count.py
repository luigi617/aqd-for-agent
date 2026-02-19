import os
import json
import glob

INPUT_DIR = "unhelpful_data"

def count_queries_in_file(path: str) -> tuple[int, int]:
    """
    Returns (num_valid_queries, num_total_items).
    Valid query = dict item with item["query"] as non-empty string after strip().
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be a list.")

    total_items = len(data)
    valid_queries = 0

    for item in data:
        if not isinstance(item, dict):
            continue
        q = item.get("query")
        if isinstance(q, str) and q.strip():
            valid_queries += 1

    return valid_queries, total_items

def main():
    paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    if not paths:
        raise SystemExit(f"No JSON files found in: {INPUT_DIR}")

    grand_valid = 0
    grand_total = 0

    for p in paths:
        file_name = os.path.basename(p)
        try:
            valid, total = count_queries_in_file(p)
            print(f"{file_name}: valid_queries={valid}  total_items={total}")
            grand_valid += valid
            grand_total += total
        except Exception as e:
            print(f"{file_name}: ERROR: {e}")

    print(f"\nTOTAL: valid_queries={grand_valid}  total_items={grand_total}")

if __name__ == "__main__":
    main()
