"""Integration test for the full pipeline."""

import datetime
import sys
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from main import load_config, run_pipeline, PipelineResult
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
        config["output"]["directory"] = str(OUTPUT_DIR)

        result = run_pipeline(config)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.vocabulary)
        self.assertIsNotNone(result.cooccurrence_matrix)
        self.assertGreater(len(result.vocabulary.translations), 0)
        self.assertGreater(len(result.vocabulary.word_variants), 0)

        output_path = Path(result.output_dir)
        self.assertTrue((output_path / "translations.json").exists())
        # Light vocabulary (no embeddings, no private fields, no cooccurrence)
        self.assertTrue((output_path / "vocabulary" / "translations.json").exists())
        # Full vocabulary (with embeddings, private fields, and cooccurrence)
        self.assertTrue(
            (output_path / "vocabulary_full" / "translations.json").exists()
        )
        self.assertTrue(
            (output_path / "vocabulary_full" / "cooccurrence_matrix.npz").exists()
        )

        self._visualize_cooccurrence_matrix(result)
        self._print_cooccurrences(result)
        self._print_store_contents(result)
        self._print_word_variants(result)

        print(f"\nPipeline completed successfully!")
        print(f"Output directory: {result.output_dir}")
        print(f"Translations: {len(result.vocabulary.translations)}")
        print(f"Word variants: {len(result.vocabulary.word_variants)}")

    def _visualize_cooccurrence_matrix(self, result: PipelineResult):
        """Create and save a visualization of the co-occurrence matrix."""
        vocab = result.vocabulary
        matrix = result.cooccurrence_matrix

        sorted_variants = sorted(
            vocab.word_variants,
            key=lambda v: (v.source_translation_id, v.meaning_idx),
        )
        reorder_map = {
            v.id: new_idx for new_idx, v in enumerate(sorted_variants)
        }  # old_id -> new_idx

        labels = []
        for variant in sorted_variants:
            translation = variant.primary_translation or "?"
            labels.append(f"{variant.lemma} - {translation} ({variant.frequency})")

        dense_matrix = matrix.toarray()
        n = len(labels)
        reordered_matrix = np.zeros((n, n))
        for old_i in range(n):
            for old_j in range(n):
                new_i = reorder_map[old_i]
                new_j = reorder_map[old_j]
                reordered_matrix[new_i, new_j] = dense_matrix[old_i, old_j]
        dense_matrix = reordered_matrix

        _, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(dense_matrix, cmap="YlOrRd", aspect="auto")

        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", size=0)

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

        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Co-occurrence count", rotation=-90, va="bottom")

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title("Word Variant Co-occurrence Matrix", fontsize=14, pad=20)

        plt.tight_layout()

        output_path = Path(result.output_dir) / "cooccurrence_matrix.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"\nCo-occurrence matrix visualization saved to: {output_path}")

    def _print_cooccurrences(self, result: PipelineResult):
        """Print co-occurrence details for each word variant."""
        vocab = result.vocabulary

        print("\n" + "=" * 80)
        print("CO-OCCURRENCE DETAILS")
        print("=" * 80)

        sorted_variants = sorted(vocab.word_variants, key=lambda v: -v.frequency)

        for variant in sorted_variants:
            cooccurrences = vocab.get_cooccurrences(variant.id, top_k=50)
            variant_label = f"{variant.lemma} ({variant.primary_translation}) [{variant.frequency}x]"

            if not cooccurrences:
                print(f"{variant_label}: (no co-occurrences)")
                continue

            cooc_parts = []
            for other_id, count in cooccurrences:
                other = vocab.get_word_variant(other_id)
                cooc_parts.append(
                    f"{other.lemma} ({other.primary_translation}) [{int(count)}]"
                )

            print(f"{variant_label}: {' - '.join(cooc_parts)}")

        print("=" * 80)

    def _print_store_contents(self, result):
        """Print contents of all stores."""
        print("\n" + "=" * 80)
        print("STORE CONTENTS")
        print("=" * 80)

        for corpus_name, (sent_store, lemma_store, _) in result.corpus_stores.items():
            print(f"\n--- Corpus: {corpus_name} ---")

            dis_store = result.disambiguated_stores.get(corpus_name)

            sentences = list(sent_store)
            lemmas = list(lemma_store)
            disambiguated = list(dis_store) if dis_store else None

            for i, (sid, tokens) in enumerate(sentences):
                _, lemma_tokens = lemmas[i]
                variants = disambiguated[i][1] if disambiguated else None

                print(f"\n  Sentence {sid}:")
                print(f"    Tokens:  {tokens}")
                print(f"    Lemmas:  {lemma_tokens}")
                if variants is not None:
                    # Mark polysemous variants with [POL]
                    variant_strs = []
                    for v_id in variants:
                        if v_id is None:
                            variant_strs.append("None")
                        else:
                            variant = result.vocabulary.get_word_variant(v_id)
                            num_meanings = len(variant.translation.meanings)
                            pol_marker = "[POL]" if num_meanings > 1 else ""
                            variant_strs.append(f"{v_id}{pol_marker}")
                    print(f"    Variants: [{', '.join(variant_strs)}]")
                    # Show all translations for each variant, mark polysemous with [POL]
                    translation_strs = []
                    for v_id in variants:
                        if v_id is None:
                            translation_strs.append("None")
                        else:
                            variant = result.vocabulary.get_word_variant(v_id)
                            num_meanings = len(variant.translation.meanings)
                            pol_marker = "[POL]" if num_meanings > 1 else ""
                            translation_strs.append(
                                f"{variant.all_translations}{pol_marker}"
                            )
                    print(f"    Translations: [{', '.join(translation_strs)}]")

    def _print_word_variants(self, result):
        """Print all WordVariants with their IDs."""
        print("\n" + "=" * 80)
        print("WORD VARIANTS")
        print("=" * 80)

        for variant in result.vocabulary.word_variants:
            # Check if this lemma has multiple meanings (polysemous)
            num_meanings = len(variant.translation.meanings)
            polysemy_marker = " [POL]" if num_meanings > 1 else ""
            print(
                f"  ID {variant.id}: {variant.lemma}|{variant.pos} "
                f"-> {variant.all_translations} "
                f"(meaning_idx={variant.meaning_idx}, freq={variant.frequency}, "
                f"string_id={variant.string_id}){polysemy_marker}"
            )


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
