"""Test clustering with Sentence Transformer embeddings and visualization."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from umap import UMAP


# === CONFIGURATION ===
MODEL_NAME = "sentence-transformers/LaBSE"
# MODEL_NAME = "BAAI/bge-m3"
# MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
CSV_PATH = "experiments/data/cluster_test_verbs.csv"  # or "experiments/cluster_test_sentence_transformer.csv"
N_CLUSTERS = 8  # 8 for verbs, 9 for nouns
REDUCTION_METHOD = "umap"  # "umap" or "tsne"


def load_data(csv_path: str) -> pd.DataFrame:
    """Load CSV with spanish, german, definition columns."""
    return pd.read_csv(csv_path, header=None, names=["spanish", "german", "definition"])


def compute_embeddings(df: pd.DataFrame, model: SentenceTransformer) -> tuple:
    """Compute embeddings for each word meaning using the definition."""
    word_meanings = []
    embeddings = []

    for _, row in df.iterrows():
        spanish = row["spanish"].strip()
        german = row["german"].strip()
        definition = row["definition"].strip()

        embedding = model.encode(definition, convert_to_numpy=True)
        word_meanings.append(f"{spanish}, {german}")
        embeddings.append(embedding)

    return word_meanings, np.array(embeddings)


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """Cluster embeddings using KMeans."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(embeddings)


def reduce_dimensions(embeddings: np.ndarray, method: str = "tsne") -> np.ndarray:
    """Reduce embeddings to 2D for visualization."""
    if method == "tsne":
        reducer = TSNE(
            n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1)
        )
    elif method == "umap":
        reducer = UMAP(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")

    return reducer.fit_transform(embeddings)


def plot_clusters(
    coords_2d: np.ndarray,
    clusters: np.ndarray,
    word_meanings: list,
    title: str,
    output_path: str,
):
    """Create scatter plot with cluster coloring and word labels."""
    plt.figure(figsize=(14, 10))

    scatter = plt.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=clusters,
        cmap="tab10",
        s=100,
        alpha=0.7,
    )

    # Add labels for each point
    for i, label in enumerate(word_meanings):
        plt.annotate(
            label,
            (coords_2d[i, 0], coords_2d[i, 1]),
            fontsize=8,
            alpha=0.8,
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.colorbar(scatter, label="Cluster")
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to {output_path}")


def print_clusters(clusters: np.ndarray, word_meanings: list, n_clusters: int):
    """Print cluster contents to console."""
    print("\n" + "=" * 80)
    print("CLUSTERING RESULTS")
    print("=" * 80)

    for cluster_id in range(n_clusters):
        indices = [i for i, c in enumerate(clusters) if c == cluster_id]
        print(f"\nCLUSTER {cluster_id} ({len(indices)} words):")
        for idx in indices:
            print(f"  {word_meanings[idx]}")


if __name__ == "__main__":
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"\nLoading data from {CSV_PATH}")
    df = load_data(CSV_PATH)
    print(f"Found {len(df)} word meanings")

    print("\nComputing embeddings...")
    word_meanings, embeddings = compute_embeddings(df, model)

    print(f"\nClustering into {N_CLUSTERS} groups...")
    clusters = cluster_embeddings(embeddings, N_CLUSTERS)

    print(f"\nReducing dimensions with {REDUCTION_METHOD.upper()}...")
    coords_2d = reduce_dimensions(embeddings, REDUCTION_METHOD)

    dataset_name = CSV_PATH.split("/")[-1].replace(".csv", "")
    output_path = (
        f"experiments/outputs/clustering_{dataset_name}_{REDUCTION_METHOD}.png"
    )
    plot_clusters(
        coords_2d,
        clusters,
        word_meanings,
        f"Word Meaning Clusters ({REDUCTION_METHOD.upper()})",
        output_path,
    )

    print_clusters(clusters, word_meanings, N_CLUSTERS)
