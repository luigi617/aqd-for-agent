import argparse
import json
import random
import os
import numpy as np
try:
    import torch  # optional, used only for serialization convenience
except Exception:  # pragma: no cover
    torch = None

from scripts.eval.intent_diversity import evaluate_intent_diversity
from scripts.eval.lexical_diversity import evaluate_lexical_diversity
from scripts.eval.semantic_diversity import evaluate_semantic_diversity

def _json_default(obj):
    """Best-effort conversion for NumPy/PyTorch objects to JSON-serializable types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if torch is not None:
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, (torch.int32, torch.int64, torch.float32, torch.float64)):  # type: ignore[attr-defined]
            return obj.item()  # type: ignore[union-attr]
    # Let json raise for unknown types
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def main(args):
    with open(args.input_file, "r") as f:
        data = json.load(f)

    # select 1000 random questions
    # set seed
    # random.seed(42)
    # subsets = random.sample(data, 50)

    # evaluate intent diversity and summarize the intents
    intent_labeled_data, intents_statistics = evaluate_intent_diversity(data, args.model_path)
    print(intents_statistics)
    # save the intent labeled data
    with open(os.path.join(args.output_dir, "intent_labeled_data.json"), "w") as f:
        json.dump(intent_labeled_data, f, indent=4)
    with open(os.path.join(args.output_dir, "intents_statistics.json"), "w") as f:
        json.dump(intents_statistics, f, indent=4)

    # evaluate lexical diversity
    lexical_diversity = evaluate_lexical_diversity(data)
    print(lexical_diversity)
    with open(os.path.join(args.output_dir, "lexical_diversity.json"), "w") as f:
        json.dump(lexical_diversity, f, indent=4, default=_json_default)

    # evaluate semantic diversity
    embedding_diversity, n_clusters, cluster_diversity = evaluate_semantic_diversity(data, args.embedding_model_path)
    print(f"Embedding Diversity: {embedding_diversity}, Number of Clusters: {n_clusters}, Cluster Diversity: {cluster_diversity}")

    with open(os.path.join(args.output_dir, "semantic_diversity.json"), "w") as f:
        json.dump({
            "embedding_diversity": embedding_diversity,
            "n_clusters": n_clusters,
            "cluster_diversity": cluster_diversity
        }, f, indent=4, default=_json_default)

    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="/home/yunanlu/code/aqd-for-agent/output_data/gpt5-mini_1126/cleaned_data.json", help="path to the input file")
    parser.add_argument("--model_path", type=str, default="gpt-5-mini", help="path to the model")
    parser.add_argument("--output_dir", type=str, default="/home/yunanlu/code/aqd-for-agent/output_data/gpt5-mini_1126", help="path to the output file")
    parser.add_argument("--embedding_model_path", type=str, default="Qwen/Qwen3-Embedding-0.6B", help="path to the embedding model")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        data = json.load(f)

    main(args)

'''
python -m scripts.eval.diversity_analysis \
--input_file /home/yunanlu/code/aqd-for-agent/output_data/gpt5-mini_1126/human_annotation/real_unhelpful_logs_1293.json \
--output_dir /home/yunanlu/code/aqd-for-agent/output_data/gpt5-mini_1126/human_annotation \
--embedding_model_path Qwen/Qwen3-Embedding-0.6B
'''