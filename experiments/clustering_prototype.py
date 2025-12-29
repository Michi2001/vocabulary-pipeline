"""
Prototype for semantic word clustering with POS-diversity penalty.

Approach C: Modified distance matrix where same-POS words receive a penalty,
encouraging POS-diverse clusters suitable for story generation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path
from collections import Counter

from vocab import Vocabulary, WordVariant
from sentence_transformers import SentenceTransformer

# === CONFIGURATION ===
EMBEDDING_MODEL = "sentence-transformers/LaBSE"  # Model for generating embeddings
TRANSLATIONS_PATH = (
    "output/spanish_lemma_frequencies/slice_6000_6400_with_vectors/translations.json"
)
OUTPUT_DIR = "experiments/outputs/clustering"

# Clustering parameters
MAX_GROUP_SIZE = 5  # Maximum words per group
POS_PENALTY = 1.0  # Penalty factor for same-POS pairs (1.0 = no penalty)
MAX_DISTANCE = (
    0.6  # Maximum distance threshold for clustering - tune based on percentiles
)
SAMPLE_SIZE = 50  # Number of words to cluster (for testing)
RANDOM_SEED = 43

# Visualization
POS_MARKERS = {
    "VERB": "o",
    "NOUN": "s",
    "ADJ": "^",
    "ADV": "D",
    "ADP": "v",
    "CONJ": "p",
}
POS_COLORS = {
    "VERB": "#e41a1c",
    "NOUN": "#377eb8",
    "ADJ": "#4daf4a",
    "ADV": "#984ea3",
    "ADP": "#ff7f00",
    "CONJ": "#a65628",
}


def get_embeddings_and_variants(
    vocab: Vocabulary,
    n: int,
    seed: int,
    model: SentenceTransformer,
    use_precomputed: bool = False,
) -> tuple[np.ndarray, list[WordVariant]]:
    """
    Sample n variants and return embeddings matrix.

    Args:
        vocab: Vocabulary instance
        n: Number of variants to sample
        seed: Random seed
        model: SentenceTransformer model for generating embeddings
        use_precomputed: If True, use embeddings from translations.json; if False, regenerate

    Returns:
        Tuple of (embeddings matrix, list of variants)
    """
    variants = vocab.sample_word_variants(n, seed=seed)

    if use_precomputed:
        # Use existing embeddings from file
        valid = [v for v in variants if v.embedding is not None]
        embeddings = np.array([v.embedding for v in valid])
        print(f"Using precomputed embeddings from file")
    else:
        # Generate new embeddings from definition_es
        print(f"Generating fresh embeddings using {model}")
        definitions = []
        valid = []

        for v in variants:
            if v.meaning.definition_es:
                definitions.append(v.meaning.definition_es)
                valid.append(v)

        if not definitions:
            raise ValueError("No definitions found to embed")

        # Encode all definitions
        embeddings = model.encode(
            definitions,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32,
        )

        print(
            f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}"
        )

    return embeddings, valid


def compute_modified_distance_matrix(
    embeddings: np.ndarray,
    variants: list[WordVariant],
    pos_penalty: float = 1.5,
) -> np.ndarray:
    """
    Compute distance matrix with POS penalty.

    Same-POS pairs get their distance multiplied by pos_penalty,
    making them appear "further apart" to encourage POS diversity.
    """
    # Base cosine distances (0 = identical, 1 = orthogonal, 2 = opposite)
    distances = cosine_distances(embeddings)

    # Apply POS penalty
    n = len(variants)
    for i in range(n):
        for j in range(i + 1, n):
            if variants[i].pos == variants[j].pos:
                distances[i, j] *= pos_penalty
                distances[j, i] *= pos_penalty

    return distances


def save_distance_matrix(
    distance_matrix: np.ndarray,
    variants: list[WordVariant],
    output_path: str,
):
    """Save distance matrix to a readable text file."""
    n = len(variants)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        # Header with word labels
        f.write("DISTANCE MATRIX\n")
        f.write("=" * 80 + "\n\n")

        # Legend: index -> word
        f.write("INDEX LEGEND:\n")
        f.write("-" * 40 + "\n")
        for i, v in enumerate(variants):
            f.write(f"{i:3d}: {v.lemma} ({v.pos}) -> {v.primary_translation}\n")
        f.write("\n")

        # Matrix header
        f.write("MATRIX (row, col -> distance):\n")
        f.write("-" * 40 + "\n")

        # Column headers (indices)
        f.write("     ")
        for j in range(n):
            f.write(f"{j:6d}")
        f.write("\n")

        # Matrix rows
        for i in range(n):
            f.write(f"{i:3d}: ")
            for j in range(n):
                f.write(f"{distance_matrix[i, j]:6.3f}")
            f.write(f"  | {variants[i].lemma}\n")

        f.write("\n")

        # Sorted list of all pairwise distances
        f.write("ALL PAIRWISE DISTANCES (sorted):\n")
        f.write("-" * 40 + "\n")

        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((distance_matrix[i, j], i, j))
        pairs.sort()

        for dist, i, j in pairs:
            vi, vj = variants[i], variants[j]
            f.write(f"{dist:.3f}: {vi.lemma}({vi.pos}) <-> {vj.lemma}({vj.pos})\n")

    print(f"Distance matrix saved to: {output_path}")


def analyze_distance_distribution(
    distance_matrix: np.ndarray, variants: list[WordVariant]
):
    """Print statistics about the distance distribution to help tune MAX_DISTANCE."""
    n = len(distance_matrix)
    # Get upper triangle (excluding diagonal)
    upper_tri = distance_matrix[np.triu_indices(n, k=1)]

    print("\n" + "-" * 70)
    print("DISTANCE DISTRIBUTION (modified with POS penalty)")
    print("-" * 70)
    print(f"Min distance:  {np.min(upper_tri):.3f}")
    print(f"Max distance:  {np.max(upper_tri):.3f}")
    print(f"Mean distance: {np.mean(upper_tri):.3f}")
    print(f"Median distance: {np.median(upper_tri):.3f}")
    print(f"Std deviation: {np.std(upper_tri):.3f}")

    # Percentiles
    percentiles = [10, 25, 50, 75, 90]
    print(f"\nPercentiles:")
    for p in percentiles:
        print(f"  {p}th: {np.percentile(upper_tri, p):.3f}")

    # Find closest pairs (for debugging)
    print(f"\nClosest 5 pairs:")
    flat_indices = np.argsort(upper_tri)[:5]
    for idx in flat_indices:
        i, j = np.triu_indices(n, k=1)
        vi, vj = variants[i[idx]], variants[j[idx]]
        dist = upper_tri[idx]
        print(f"  {dist:.3f}: {vi.lemma}({vi.pos}) <-> {vj.lemma}({vj.pos})")


def cluster_words(
    distance_matrix: np.ndarray,
    max_distance: float,
) -> np.ndarray:
    """
    Perform agglomerative clustering with distance threshold.

    Returns cluster labels (-1 for noise/singletons if using HDBSCAN-like behavior).
    """
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=max_distance,
        metric="precomputed",
        linkage="complete",
    )
    labels = clustering.fit_predict(distance_matrix)
    return labels


def split_large_clusters(
    labels: np.ndarray,
    distance_matrix: np.ndarray,
    variants: list[WordVariant],
    max_size: int,
) -> np.ndarray:
    """Split clusters that exceed max_size using sub-clustering."""
    new_labels = labels.copy()
    max_label = labels.max()

    unique_labels = np.unique(labels)
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        if len(indices) > max_size:
            # Sub-cluster this group
            sub_distances = distance_matrix[np.ix_(indices, indices)]
            n_subclusters = int(np.ceil(len(indices) / max_size))

            sub_clustering = AgglomerativeClustering(
                n_clusters=n_subclusters,
                metric="precomputed",
                linkage="average",
            )
            sub_labels = sub_clustering.fit_predict(sub_distances)

            # Assign new labels
            for i, idx in enumerate(indices):
                new_labels[idx] = max_label + 1 + sub_labels[i]
            max_label += n_subclusters

    # Renumber labels to be consecutive
    unique_new = np.unique(new_labels)
    label_map = {old: new for new, old in enumerate(unique_new)}
    return np.array([label_map[l] for l in new_labels])


def analyze_clusters(
    labels: np.ndarray,
    variants: list[WordVariant],
    distance_matrix: np.ndarray,
) -> dict:
    """Analyze cluster quality and statistics."""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    stats = {
        "n_clusters": n_clusters,
        "cluster_sizes": [],
        "pos_diversity": [],  # Number of unique POS per cluster
        "avg_intra_distance": [],  # Average distance within cluster
        "clusters": [],
    }

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        cluster_variants = [variants[i] for i in indices]

        # Size
        size = len(indices)
        stats["cluster_sizes"].append(size)

        # POS diversity
        pos_types = set(v.pos for v in cluster_variants)
        stats["pos_diversity"].append(len(pos_types))

        # Intra-cluster distance
        if size > 1:
            sub_dist = distance_matrix[np.ix_(indices, indices)]
            # Get upper triangle (excluding diagonal)
            upper_tri = sub_dist[np.triu_indices(size, k=1)]
            avg_dist = np.mean(upper_tri)
        else:
            avg_dist = 0.0
        stats["avg_intra_distance"].append(avg_dist)

        # Store cluster details
        stats["clusters"].append(
            {
                "label": label,
                "size": size,
                "pos_diversity": len(pos_types),
                "pos_counts": Counter(v.pos for v in cluster_variants),
                "avg_distance": avg_dist,
                "words": [
                    (v.lemma, v.pos, v.primary_translation) for v in cluster_variants
                ],
            }
        )

    return stats


def print_cluster_report(stats: dict):
    """Print a human-readable cluster report."""
    print("\n" + "=" * 70)
    print("CLUSTERING REPORT")
    print("=" * 70)
    print(f"Total clusters: {stats['n_clusters']}")
    print(
        f"Cluster sizes: min={min(stats['cluster_sizes'])}, "
        f"max={max(stats['cluster_sizes'])}, "
        f"avg={np.mean(stats['cluster_sizes']):.1f}"
    )
    print(
        f"POS diversity: min={min(stats['pos_diversity'])}, "
        f"max={max(stats['pos_diversity'])}, "
        f"avg={np.mean(stats['pos_diversity']):.1f}"
    )
    print(f"Avg intra-cluster distance: {np.mean(stats['avg_intra_distance']):.3f}")

    print("\n" + "-" * 70)
    print("CLUSTER DETAILS")
    print("-" * 70)

    for cluster in sorted(stats["clusters"], key=lambda x: -x["size"]):
        print(
            f"\nCluster {cluster['label']} (size={cluster['size']}, "
            f"POS diversity={cluster['pos_diversity']}, "
            f"avg_dist={cluster['avg_distance']:.3f})"
        )
        print(f"  POS: {dict(cluster['pos_counts'])}")
        for lemma, pos, translation in cluster["words"]:
            print(f"    - {lemma} ({pos}): {translation}")


def visualize_clusters(
    embeddings: np.ndarray,
    variants: list[WordVariant],
    labels: np.ndarray,
    title: str,
    output_path: str,
):
    """Visualize clusters with UMAP reduction."""
    from umap import UMAP

    # Reduce dimensions
    n_neighbors = min(15, len(embeddings) - 1)
    reducer = UMAP(n_components=2, random_state=RANDOM_SEED, n_neighbors=n_neighbors)
    coords_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(16, 12))

    # Get unique clusters and assign colors
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, max(20, n_clusters)))

    # Plot each point
    for i, (variant, coord, label) in enumerate(zip(variants, coords_2d, labels)):
        marker = POS_MARKERS.get(variant.pos, "x")
        color = cluster_colors[label % len(cluster_colors)]

        plt.scatter(
            coord[0],
            coord[1],
            marker=marker,
            c=[color],
            s=150,
            alpha=0.8,
            edgecolors="white",
            linewidths=1,
        )

        # Add label
        plt.annotate(
            f"{variant.primary_translation}",
            (coord[0], coord[1]),
            fontsize=7,
            alpha=0.9,
            xytext=(5, 5),
            textcoords="offset points",
        )

    # Create legend for POS markers
    legend_elements = []
    from matplotlib.lines import Line2D

    for pos, marker in POS_MARKERS.items():
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="gray",
                linestyle="None",
                markersize=10,
                label=pos,
            )
        )
    plt.legend(handles=legend_elements, loc="upper right", title="POS")

    plt.title(title)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def run_clustering(
    vocab: Vocabulary,
    embedding_model: SentenceTransformer,
    sample_size: int = SAMPLE_SIZE,
    max_group_size: int = MAX_GROUP_SIZE,
    pos_penalty: float = POS_PENALTY,
    max_distance: float = MAX_DISTANCE,
    seed: int = RANDOM_SEED,
    use_precomputed_embeddings: bool = False,
    visualize: bool = True,
    output_suffix: str = "",
) -> dict:
    """Run the full clustering pipeline."""
    print(f"\n{'='*70}")
    print(f"CLUSTERING PARAMETERS")
    print(f"{'='*70}")
    print(f"Sample size: {sample_size}")
    print(f"Max group size: {max_group_size}")
    print(f"POS penalty: {pos_penalty}")
    print(f"Max distance: {max_distance}")
    print(f"Use precomputed embeddings: {use_precomputed_embeddings}")

    # Get embeddings
    embeddings, variants = get_embeddings_and_variants(
        vocab, sample_size, seed, embedding_model, use_precomputed_embeddings
    )
    print(f"Variants with embeddings: {len(variants)}")

    # Compute modified distance matrix
    print("\nComputing modified distance matrix...")
    distance_matrix = compute_modified_distance_matrix(
        embeddings, variants, pos_penalty
    )

    # Analyze distance distribution
    analyze_distance_distribution(distance_matrix, variants)

    # Save distance matrix to file
    matrix_path = f"{OUTPUT_DIR}/distance_matrix_n{sample_size}.txt"
    save_distance_matrix(distance_matrix, variants, matrix_path)

    # Initial clustering
    print("Performing agglomerative clustering...")
    labels = cluster_words(distance_matrix, max_distance)
    print(f"Initial clusters: {len(np.unique(labels))}")

    # Split large clusters
    print(f"Splitting clusters larger than {max_group_size}...")
    labels = split_large_clusters(labels, distance_matrix, variants, max_group_size)
    print(f"Final clusters: {len(np.unique(labels))}")

    # Analyze
    stats = analyze_clusters(labels, variants, distance_matrix)
    print_cluster_report(stats)

    # Visualize
    if visualize:
        suffix = f"_penalty{pos_penalty}_dist{max_distance}{output_suffix}"
        output_path = f"{OUTPUT_DIR}/clusters_n{sample_size}{suffix}.png"
        title = (
            f"Word Clusters (n={len(variants)}, penalty={pos_penalty}, "
            f"max_dist={max_distance})"
        )
        visualize_clusters(embeddings, variants, labels, title, output_path)

    return {
        "labels": labels,
        "variants": variants,
        "embeddings": embeddings,
        "distance_matrix": distance_matrix,
        "stats": stats,
    }


def compare_penalties(
    vocab: Vocabulary, embedding_model: SentenceTransformer, penalties: list[float]
):
    """Compare clustering results with different POS penalties."""
    print("\n" + "=" * 70)
    print("COMPARING POS PENALTIES")
    print("=" * 70)

    results = []
    for penalty in penalties:
        result = run_clustering(
            vocab,
            embedding_model,
            pos_penalty=penalty,
            visualize=True,
            output_suffix=f"_compare",
        )
        results.append(
            {
                "penalty": penalty,
                "n_clusters": result["stats"]["n_clusters"],
                "avg_pos_diversity": np.mean(result["stats"]["pos_diversity"]),
                "avg_intra_distance": np.mean(result["stats"]["avg_intra_distance"]),
            }
        )

    print("\n" + "-" * 70)
    print("COMPARISON SUMMARY")
    print("-" * 70)
    print(
        f"{'Penalty':<10} {'Clusters':<10} {'Avg POS Div':<15} {'Avg Intra Dist':<15}"
    )
    for r in results:
        print(
            f"{r['penalty']:<10.1f} {r['n_clusters']:<10} "
            f"{r['avg_pos_diversity']:<15.2f} {r['avg_intra_distance']:<15.3f}"
        )


if __name__ == "__main__":
    print(f"Loading vocabulary from {TRANSLATIONS_PATH}")
    vocab = Vocabulary.from_file(TRANSLATIONS_PATH)
    print(f"Loaded: {vocab}")

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print(
        f"Model loaded. Dimension: {embedding_model.get_sentence_embedding_dimension()}"
    )

    # Single run with FRESH embeddings (regenerate from definition_es)
    print("\n" + "=" * 70)
    print("RUNNING WITH FRESH EMBEDDINGS")
    print("=" * 70)
    run_clustering(vocab, embedding_model, use_precomputed_embeddings=False)

    # Optionally: Compare with precomputed embeddings
    print("\n" + "=" * 70)
    print("RUNNING WITH PRECOMPUTED EMBEDDINGS")
    print("=" * 70)
    run_clustering(vocab, embedding_model, use_precomputed_embeddings=True)

    # Compare different penalties
    # compare_penalties(vocab, embedding_model, [1.0, 1.5, 2.0, 2.5])
