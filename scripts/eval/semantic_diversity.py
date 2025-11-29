import json
# load the embedding models and calculate index
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import AgglomerativeClustering
try:
    import torch  # optional, used only for type checks/conversions
except Exception:  # pragma: no cover
    torch = None

CLUSTERING_THRESHOLD = 0.5

def _to_numpy(x):
    """Convert torch.Tensor to numpy if needed; otherwise return np.asarray(x)."""
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def calculate_embedding_diversity(embedding_model, embeddings):
    embedding_similarity = embedding_model.similarity(embeddings, embeddings)
    sim_np = _to_numpy(embedding_similarity)
    return 1 - float(np.mean(sim_np))


def calculate_cluster_diversity(embedding_model, embeddings):
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=CLUSTERING_THRESHOLD, 
        linkage='average',  # Uses average distance between sets to decide merges
        metric='cosine'
    )
    labels = clustering.fit_predict(embeddings)
    n_clusters = clustering.n_clusters_
    cluster_distances = []

    # The distance between points in a cluster
    for cluster_id in range(n_clusters):
        # Get points in this cluster
        cluster_points = embeddings[labels == cluster_id]
        
        if len(cluster_points) > 1:
            # Calculate all pairwise distances in this cluster
            dists = embedding_model.similarity(cluster_points, cluster_points)
            dists_np = _to_numpy(dists)
            
            # Get the upper triangle of the distance matrix (excluding diagonal 0s)
            # to get unique pairs
            unique_distances = dists_np[np.triu_indices(len(cluster_points), k=1)]
            avg_dist = float(np.mean(unique_distances))
            cluster_distances.append(avg_dist)
        else:
            print(f"Cluster {cluster_id}: 1 item (Singleton)")
    return n_clusters, (float(np.mean(cluster_distances)) if cluster_distances else 0.0)


def evaluate_semantic_diversity(data, embedding_model_path):
    embedding_model = SentenceTransformer(embedding_model_path)
    # calculate the semantic diversity
    question_strings = []
    for item in data:
        question = item["query"]
        question_strings.append(question)
    question_embeddings = embedding_model.encode(question_strings)

    # embedding-based semantic diversity
    print(embedding_model.similarity_fn_name)
    embedding_diversity = calculate_embedding_diversity(embedding_model, question_embeddings)
    print(f"Embedding Diversity: {embedding_diversity}")


    # cluster the questions
    n_clusters, cluster_diversity = calculate_cluster_diversity(embedding_model, question_embeddings)
    print(f"Number of Clusters: {n_clusters}, Cluster Diversity: {cluster_diversity}")


    return embedding_diversity, n_clusters, cluster_diversity