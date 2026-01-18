"""
Integration test for the full pipeline.

Tests the pipeline with a sample corpus, running translation and disambiguation
to verify the complete flow and co-occurrence matrix generation.
"""

import sys
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import datetime

from main import load_config, run_pipeline
from test_utils import Tee

TEST_DIR = Path("test")
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = TEST_DIR / "outputs" / "test_pipeline" / TIMESTAMP
OUTPUT_LOG_PATH = OUTPUT_DIR / "pipeline.log"


class TestPipelineIntegration(unittest.TestCase):
    """Integration test for the full pipeline with LLM translation."""

    config_path = TEST_DIR / "test_config.yaml"
    cache_dir = TEST_DIR / "cache"

    def test_full_pipeline(self):
        """Test full pipeline with LLM translation and disambiguation."""
        config = load_config(str(self.config_path))

        # Override output directory
        config["output"]["directory"] = str(OUTPUT_DIR)

        result = run_pipeline(config)

        # Basic assertions
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.vocabulary)
        self.assertIsNotNone(result.cooccurrence_matrix)
        self.assertGreater(len(result.vocabulary.translations), 0)
        self.assertGreater(len(result.vocabulary.word_variants), 0)

        # Verify output files exist
        output_path = Path(result.output_dir)
        self.assertTrue((output_path / "translations.json").exists())
        self.assertTrue((output_path / "vocabulary" / "translations.json").exists())
        self.assertTrue(
            (output_path / "vocabulary" / "cooccurrence_matrix.npz").exists()
        )

        # Generate co-occurrence matrix visualization
        self._visualize_cooccurrence_matrix(result)

        # Print co-occurrence details for each variant
        self._print_cooccurrences(result)

        print(f"\nPipeline completed successfully!")
        print(f"Output directory: {result.output_dir}")
        print(f"Translations: {len(result.vocabulary.translations)}")
        print(f"Word variants: {len(result.vocabulary.word_variants)}")

    def _visualize_cooccurrence_matrix(self, result):
        """Create and save a visualization of the co-occurrence matrix."""
        vocab = result.vocabulary
        matrix = result.cooccurrence_matrix

        # Sort variants by translation_id then meaning_idx (group meanings together)
        sorted_variants = sorted(
            vocab.word_variants,
            key=lambda v: (v.source_translation_id, v.meaning_idx),
        )
        # old_id -> new_index mapping
        reorder_map = {v.id: new_idx for new_idx, v in enumerate(sorted_variants)}

        # Get labels for word variants (with German translation and frequency)
        labels = []
        for variant in sorted_variants:
            translation = variant.primary_translation or "?"
            label = f"{variant.lemma} - {translation} ({variant.frequency})"
            labels.append(label)

        # Convert sparse matrix to dense and reorder to match sorted variants
        dense_matrix = matrix.toarray()
        n = len(labels)
        reordered_matrix = np.zeros((n, n))
        for old_i in range(n):
            for old_j in range(n):
                new_i = reorder_map[old_i]
                new_j = reorder_map[old_j]
                reordered_matrix[new_i, new_j] = dense_matrix[old_i, old_j]
        dense_matrix = reordered_matrix

        # Create figure
        _, ax = plt.subplots(figsize=(12, 10))

        # Plot heatmap
        im = ax.imshow(dense_matrix, cmap="YlOrRd", aspect="auto")

        # Add grid lines between cells
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", size=0)

        # Add cell values
        for i in range(n):
            for j in range(n):
                value = dense_matrix[i, j]
                if value > 0:
                    text_color = "white" if value > dense_matrix.max() / 2 else "black"
                    ax.text(
                        j,
                        i,
                        f"{int(value)}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=7,
                    )

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Co-occurrence count", rotation=-90, va="bottom")

        # Set ticks and labels
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

        # Add title
        ax.set_title("Word Variant Co-occurrence Matrix", fontsize=14, pad=20)

        # Adjust layout
        plt.tight_layout()

        # Save to output directory
        output_path = Path(result.output_dir) / "cooccurrence_matrix.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"\nCo-occurrence matrix visualization saved to: {output_path}")

    def _print_cooccurrences(self, result):
        """Print co-occurrence details for each word variant."""
        vocab = result.vocabulary

        print("\n" + "=" * 80)
        print("CO-OCCURRENCE DETAILS")
        print("=" * 80)

        # Sort variants by frequency descending
        sorted_variants = sorted(vocab.word_variants, key=lambda v: -v.frequency)

        for variant in sorted_variants:
            # Get co-occurrences for this variant
            cooccurrences = vocab.get_cooccurrences(variant.id, top_k=50)

            # Format variant label with frequency
            variant_label = f"{variant.lemma} ({variant.primary_translation}) [{variant.frequency}x]"

            if not cooccurrences:
                print(f"{variant_label}: (no co-occurrences)")
                continue

            # Format co-occurrence list
            cooc_parts = []
            for other_id, count in cooccurrences:
                other = vocab.get_word_variant(other_id)
                cooc_parts.append(
                    f"{other.lemma} ({other.primary_translation}) [{int(count)}]"
                )

            print(f"{variant_label}: {' - '.join(cooc_parts)}")

        print("=" * 80)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_file = open(OUTPUT_LOG_PATH, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    try:
        unittest.main(verbosity=2)
    finally:
        sys.stdout = original_stdout
        log_file.close()
