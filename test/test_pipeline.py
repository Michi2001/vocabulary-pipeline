"""
Integration test for the full pipeline.

Tests the pipeline with a sample corpus, running translation and disambiguation
to verify the complete flow and co-occurrence matrix generation.
"""

import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from main import load_config, run_pipeline


class TestPipelineIntegration(unittest.TestCase):
    """Integration test for the full pipeline with LLM translation."""

    test_dir = Path(__file__).parent
    config_path = test_dir / "test_config.yaml"
    cache_dir = test_dir / "cache"
    output_dir = test_dir / "outputs" / "test_pipeline"

    def test_full_pipeline(self):
        """Test full pipeline with LLM translation and disambiguation."""
        config = load_config(str(self.config_path))

        # Override output directory to fixed path (no timestamp)
        config["output"]["directory"] = str(self.output_dir)

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

        print(f"\nPipeline completed successfully!")
        print(f"Output directory: {result.output_dir}")
        print(f"Translations: {len(result.vocabulary.translations)}")
        print(f"Word variants: {len(result.vocabulary.word_variants)}")

    def _visualize_cooccurrence_matrix(self, result):
        """Create and save a visualization of the co-occurrence matrix."""
        vocab = result.vocabulary
        matrix = result.cooccurrence_matrix

        # Get labels for word variants
        labels = []
        for variant in vocab.word_variants:
            label = f"{variant.lemma} ({variant.pos})"
            if len(vocab.get_variant_ids_for_lemma(variant.lemma_token)) > 1:
                label += f" [{variant.meaning_idx}]"
            labels.append(label)

        # Convert sparse matrix to dense for visualization
        dense_matrix = matrix.toarray()
        n = len(labels)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
