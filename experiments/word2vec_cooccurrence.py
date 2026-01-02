"""
Compute word co-occurrence probabilities using Word2Vec.

Samples WordVariants and computes pairwise co-occurrence probabilities
using Word2Vec's predict_output_word, storing results as an adjacency matrix.
"""

import numpy as np
from pathlib import Path
from gensim.models import Word2Vec

from vocab import Vocabulary

# === CONFIGURATION ===
TRANSLATIONS_PATH = "output/spanish_lemma_frequencies/20251229_185353/translations.json"
WORD2VEC_MODEL_PATH = "models/spanish_word2vec.model"
OUTPUT_DIR = "experiments/outputs/cooccurrence"

SAMPLE_SIZE = 100
RANDOM_SEED = 42


def load_word2vec_model(model_path: str) -> Word2Vec:
    """Load a pre-trained Word2Vec model."""
    print(f"Loading Word2Vec model from {model_path}")
    model = Word2Vec.load(model_path)
    print(f"Vocabulary size: {len(model.wv)}")
    return model


def get_cooccurrence_probability(
    model: Word2Vec,
    context_word: str,
    target_word: str,
) -> float:
    """
    Get probability that target_word appears in context of context_word.
    Returns 0.0 if either word is not in vocabulary.
    """
    if context_word not in model.wv or target_word not in model.wv:
        return 0.0

    try:
        predictions = model.predict_output_word([context_word], topn=None)
        if predictions is None:
            return 0.0

        # predictions is list of (word, probability)
        prob_dict = dict(predictions)
        return prob_dict.get(target_word, 0.0)
    except Exception:
        return 0.0


def compute_cooccurrence_matrix(
    model: Word2Vec,
    lemmas: list[str],
) -> tuple[np.ndarray, list[str], list[int]]:
    """
    Compute co-occurrence probability matrix for given lemmas.

    Returns:
        matrix: NxN probability matrix where matrix[i,j] = P(lemma_j | lemma_i)
        valid_lemmas: lemmas that exist in Word2Vec vocabulary
        valid_indices: original indices of valid lemmas
    """
    # Filter lemmas that exist in vocabulary
    valid_lemmas = []
    valid_indices = []
    for i, lemma in enumerate(lemmas):
        if lemma in model.wv:
            valid_lemmas.append(lemma)
            valid_indices.append(i)

    n = len(valid_lemmas)
    print(f"Valid lemmas in vocabulary: {n}/{len(lemmas)}")

    if n == 0:
        return np.array([]), [], []

    # Compute pairwise probabilities
    matrix = np.zeros((n, n))

    print(f"Computing {n}x{n} co-occurrence matrix...")
    for i, context_word in enumerate(valid_lemmas):
        if (i + 1) % 10 == 0:
            print(f"  Processing {i + 1}/{n}...")

        # Get all predictions for this context word at once
        try:
            predictions = model.predict_output_word([context_word], topn=None)
            if predictions:
                prob_dict = dict(predictions)
                for j, target_word in enumerate(valid_lemmas):
                    matrix[i, j] = prob_dict.get(target_word, 0.0)
        except Exception:
            pass

    return matrix, valid_lemmas, valid_indices


def save_adjacency_matrix(
    matrix: np.ndarray,
    lemmas: list[str],
    output_path: str,
):
    """Save adjacency matrix to readable text file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    n = len(lemmas)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("CO-OCCURRENCE PROBABILITY MATRIX\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Size: {n} x {n}\n")
        f.write(f"Entry [i,j] = P(lemma_j appears in context of lemma_i)\n\n")

        # Legend
        f.write("INDEX LEGEND:\n")
        f.write("-" * 40 + "\n")
        for i, lemma in enumerate(lemmas):
            f.write(f"{i:3d}: {lemma}\n")
        f.write("\n")

        # Matrix as CSV-style
        f.write("MATRIX (rows=context, cols=target):\n")
        f.write("-" * 40 + "\n")

        # Header row
        f.write("," + ",".join(lemmas) + "\n")

        # Data rows
        for i, lemma in enumerate(lemmas):
            row_values = [f"{matrix[i, j]:.6f}" for j in range(n)]
            f.write(f"{lemma}," + ",".join(row_values) + "\n")

    print(f"Saved matrix to: {output_path}")


def save_adjacency_matrix_numpy(
    matrix: np.ndarray,
    lemmas: list[str],
    output_path: str,
):
    """Save adjacency matrix as numpy .npz file for efficient loading."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        matrix=matrix,
        lemmas=np.array(lemmas, dtype=object),
    )
    print(f"Saved numpy matrix to: {output_path}")


def print_top_cooccurrences(
    matrix: np.ndarray,
    lemmas: list[str],
    top_k: int = 20,
):
    """Print top co-occurring pairs."""
    n = len(lemmas)
    pairs = []

    for i in range(n):
        for j in range(n):
            if i != j and matrix[i, j] > 0:
                pairs.append((matrix[i, j], lemmas[i], lemmas[j]))

    pairs.sort(reverse=True)

    print(f"\nTop {top_k} co-occurrence probabilities:")
    print("-" * 50)
    for prob, context, target in pairs[:top_k]:
        print(f"  P({target} | {context}) = {prob:.6f}")


def print_matrix_stats(matrix: np.ndarray):
    """Print statistics about the co-occurrence matrix."""
    # Exclude diagonal
    n = matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off_diag = matrix[mask]

    nonzero = off_diag[off_diag > 0]

    print("\nMatrix Statistics (excluding diagonal):")
    print("-" * 40)
    print(f"  Total pairs: {len(off_diag)}")
    print(f"  Non-zero pairs: {len(nonzero)} ({100*len(nonzero)/len(off_diag):.1f}%)")
    if len(nonzero) > 0:
        print(f"  Min (non-zero): {nonzero.min():.8f}")
        print(f"  Max: {nonzero.max():.6f}")
        print(f"  Mean (non-zero): {nonzero.mean():.8f}")
        print(f"  Median (non-zero): {np.median(nonzero):.8f}")


def main():
    # Load vocabulary
    print(f"Loading vocabulary from {TRANSLATIONS_PATH}")
    vocab = Vocabulary.from_file(TRANSLATIONS_PATH)
    print(f"Loaded: {vocab}")

    # Sample word variants
    print(f"\nSampling {SAMPLE_SIZE} word variants (seed={RANDOM_SEED})")
    variants = vocab.sample_word_variants(SAMPLE_SIZE, seed=RANDOM_SEED)

    # Extract unique lemmas
    lemmas = [v.lemma for v in variants]
    unique_lemmas = list(dict.fromkeys(lemmas))  # Preserve order, remove duplicates
    print(f"Unique lemmas: {len(unique_lemmas)}")

    # Load Word2Vec model
    model = load_word2vec_model(WORD2VEC_MODEL_PATH)

    # Compute co-occurrence matrix
    matrix, valid_lemmas, valid_indices = compute_cooccurrence_matrix(
        model, unique_lemmas
    )

    if len(valid_lemmas) == 0:
        print("No lemmas found in Word2Vec vocabulary!")
        return

    # Print statistics
    print_matrix_stats(matrix)
    print_top_cooccurrences(matrix, valid_lemmas)

    # Save results
    output_base = f"{OUTPUT_DIR}/cooccurrence_n{len(valid_lemmas)}"
    save_adjacency_matrix(matrix, valid_lemmas, f"{output_base}.txt")
    save_adjacency_matrix_numpy(matrix, valid_lemmas, f"{output_base}.npz")


if __name__ == "__main__":
    main()
