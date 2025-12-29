"""Visualize word embeddings with dimensionality reduction (no clustering)."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP
from pathlib import Path

from vocab import Vocabulary, WordVariant

TRANSLATIONS_PATH = (
    "output/spanish_lemma_frequencies/slice_6000_6400_with_vectors/translations.json"
)
SAMPLE_SIZES = [10, 20, 50, 100, 200]  # Different sample sizes to visualize
REDUCTION_METHOD = "umap"  # "umap" or "tsne"
RANDOM_SEED = 42
OUTPUT_DIR = "experiments/outputs"

# POS to marker mapping
POS_MARKERS = {
    "VERB": "o",  # circle
    "NOUN": "s",  # square
    "ADJ": "^",  # triangle up
    "ADV": "D",  # diamond
    "ADP": "v",  # triangle down
    "CONJ": "p",  # pentagon
}
DEFAULT_MARKER = "x"

# POS to color mapping
POS_COLORS = {
    "VERB": "#e41a1c",  # red
    "NOUN": "#377eb8",  # blue
    "ADJ": "#4daf4a",  # green
    "ADV": "#984ea3",  # purple
    "ADP": "#ff7f00",  # orange
    "CONJ": "#a65628",  # brown
}
DEFAULT_COLOR = "#999999"


def load_variants_with_embeddings(vocab: Vocabulary, n: int) -> list[WordVariant]:
    """Sample n variants that have embeddings."""
    all_variants = [v for v in vocab.get_all_word_variants() if v.embedding is not None]
    if len(all_variants) < n:
        print(
            f"Warning: Only {len(all_variants)} variants with embeddings, requested {n}"
        )
        n = len(all_variants)
    return vocab.sample_word_variants(n, seed=RANDOM_SEED)


def get_embeddings_matrix(
    variants: list[WordVariant],
) -> tuple[np.ndarray, list[WordVariant]]:
    """Extract embeddings matrix from variants, filtering out those without embeddings."""
    valid_variants = [v for v in variants if v.embedding is not None]
    if not valid_variants:
        raise ValueError("No variants with embeddings found")
    embeddings = np.array([v.embedding for v in valid_variants])
    return embeddings, valid_variants


def reduce_dimensions(embeddings: np.ndarray, method: str = "umap") -> np.ndarray:
    """Reduce embeddings to 2D for visualization."""
    n_samples = len(embeddings)
    if method == "tsne":
        perplexity = min(30, n_samples - 1)
        reducer = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=perplexity)
    elif method == "umap":
        n_neighbors = min(15, n_samples - 1)
        reducer = UMAP(
            n_components=2, random_state=RANDOM_SEED, n_neighbors=n_neighbors
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return reducer.fit_transform(embeddings)


def plot_embeddings(
    coords_2d: np.ndarray,
    variants: list[WordVariant],
    title: str,
    output_path: str,
    show_labels: bool = True,
):
    """Create scatter plot with POS-based markers and colors."""
    plt.figure(figsize=(16, 12))

    # Group variants by POS for legend
    pos_groups = {}
    for i, v in enumerate(variants):
        pos = v.pos
        if pos not in pos_groups:
            pos_groups[pos] = {"x": [], "y": [], "labels": []}
        pos_groups[pos]["x"].append(coords_2d[i, 0])
        pos_groups[pos]["y"].append(coords_2d[i, 1])
        pos_groups[pos]["labels"].append(f"{v.primary_translation}")

    # Plot each POS group separately
    for pos, data in pos_groups.items():
        marker = POS_MARKERS.get(pos, DEFAULT_MARKER)
        color = POS_COLORS.get(pos, DEFAULT_COLOR)
        plt.scatter(
            data["x"],
            data["y"],
            marker=marker,
            c=color,
            s=100,
            alpha=0.7,
            label=f"{pos} ({len(data['x'])})",
            edgecolors="white",
            linewidths=0.5,
        )

        # Add labels
        if show_labels and len(variants) <= 100:
            for x, y, label in zip(data["x"], data["y"], data["labels"]):
                plt.annotate(
                    label,
                    (x, y),
                    fontsize=7,
                    alpha=0.8,
                    xytext=(5, 5),
                    textcoords="offset points",
                )

    plt.legend(loc="upper right", title="POS (count)")
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    vocab = Vocabulary.from_file(TRANSLATIONS_PATH)

    # Count variants with embeddings
    variants_with_emb = [
        v for v in vocab.get_all_word_variants() if v.embedding is not None
    ]
    print(f"Variants with embeddings: {len(variants_with_emb)}")

    for sample_size in SAMPLE_SIZES:
        print(f"\n{'='*60}")
        print(f"Processing sample size: {sample_size}")
        print(f"{'='*60}")

        # Sample variants
        variants = vocab.sample_word_variants(sample_size, seed=RANDOM_SEED)

        # Get embeddings
        embeddings, valid_variants = get_embeddings_matrix(variants)
        print(f"Valid variants with embeddings: {len(valid_variants)}")

        # Reduce dimensions
        print(f"Reducing dimensions with {REDUCTION_METHOD.upper()}...")
        coords_2d = reduce_dimensions(embeddings, REDUCTION_METHOD)

        # Plot
        output_path = f"{OUTPUT_DIR}/embeddings_{sample_size}_{REDUCTION_METHOD}.png"
        title = f"Word Embeddings ({REDUCTION_METHOD.upper()}, n={len(valid_variants)})"
        show_labels = sample_size <= 100
        plot_embeddings(coords_2d, valid_variants, title, output_path, show_labels)

        # Print POS distribution
        pos_counts = {}
        for v in valid_variants:
            pos_counts[v.pos] = pos_counts.get(v.pos, 0) + 1
        print("POS distribution:")
        for pos, count in sorted(pos_counts.items(), key=lambda x: -x[1]):
            print(f"  {pos}: {count}")
