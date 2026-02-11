"""Integration test for the full pipeline."""

import datetime
import sys
import unittest
from pathlib import Path
from src.vocab import Vocabulary

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
        self.assertGreater(len(result.vocabulary.vocab_entries), 0)

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
        self._print_vocab_entries(result)

        print(f"\nPipeline completed successfully!")
        print(f"Output directory: {result.output_dir}")
        print(f"Translations: {len(result.vocabulary.translations)}")
        print(f"Word variants: {len(result.vocabulary.vocab_entries)}")

    def _visualize_cooccurrence_matrix(self, result: PipelineResult):
        """Create and save a visualization of the co-occurrence matrix."""
        vocab = result.vocabulary
        matrix = result.cooccurrence_matrix

        sorted_variants = sorted(
            vocab.vocab_entries,
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

        sorted_variants = sorted(vocab.vocab_entries, key=lambda v: -v.frequency)

        for variant in sorted_variants:
            cooccurrences = vocab.get_cooccurrences(variant.id, top_k=50)
            variant_label = f"{variant.lemma} ({variant.primary_translation}) [{variant.frequency}x]"

            if not cooccurrences:
                print(f"{variant_label}: (no co-occurrences)")
                continue

            cooc_parts = []
            for other_id, count in cooccurrences:
                other = vocab.get_vocab_entry(other_id)
                cooc_parts.append(
                    f"{other.lemma} ({other.primary_translation}) [{int(count)}]"
                )

            print(f"{variant_label}: {' - '.join(cooc_parts)}")

        print("=" * 80)

    def _print_store_contents(self, result):
        """Print contents of all stores in a readable table format."""
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

                print(f"\n  Sentence {sid}: \"{' '.join(tokens)}\"")

                if variants is None:
                    for token, (lemma, pos) in zip(tokens, lemma_tokens):
                        print(f"    {token:<16} {pos:<5}")
                    continue

                skipped_pos = []
                for j, (token, (lemma, pos)) in enumerate(zip(tokens, lemma_tokens)):
                    v_id = variants[j]
                    if v_id is None:
                        skipped_pos.append(pos)
                        continue
                    entry = result.vocabulary.get_vocab_entry(v_id)
                    pol = " [POL]" if entry.is_disambiguated else ""
                    translation = entry.all_translations
                    print(
                        f"    {token:<16} {pos:<5} "
                        f"\u2192 {translation:<35}{pol} #{v_id}"
                    )

                if skipped_pos:
                    counts = {}
                    for p in skipped_pos:
                        counts[p] = counts.get(p, 0) + 1
                    parts = [f"{c}x {p}" if c > 1 else p for p, c in counts.items()]
                    print(f"    ({len(skipped_pos)} skipped: {', '.join(parts)})")

    def _print_vocab_entries(self, result):
        """Print all WordVariants with their IDs."""
        print("\n" + "=" * 80)
        print("WORD VARIANTS")
        print("=" * 80)

        for entry in result.vocabulary.vocab_entries:
            pol = " [POL]" if entry.is_disambiguated else ""
            print(
                f"  ID {entry.id}: {entry.lemma}|{entry.pos} "
                f"-> {entry.all_translations} "
                f"(meaning_idx={entry.meaning_idx}, freq={entry.frequency}, "
                f"string_id={entry.string_id}){pol}"
            )

    def test_load_saved_vocabulary(self):
        """Test loading a saved vocabulary and verify data integrity."""
        # First run the pipeline to create vocabulary
        config = load_config(str(self.config_path))
        config["output"]["directory"] = str(OUTPUT_DIR)
        result = run_pipeline(config)

        self.assertIsNotNone(result.vocabulary)
        original_vocab = result.vocabulary

        vocab_full_path = Path(result.output_dir) / "vocabulary_full"
        loaded_vocab = Vocabulary.from_dir(str(vocab_full_path))

        # Verify translations count matches
        self.assertEqual(
            len(loaded_vocab.translations),
            len(original_vocab.translations),
            "Translation count mismatch",
        )

        # Verify word variants count matches
        self.assertEqual(
            len(loaded_vocab.vocab_entries),
            len(original_vocab.vocab_entries),
            "Word variant count mismatch",
        )

        # Verify co-occurrence matrix loaded
        self.assertIsNotNone(
            loaded_vocab.cooccurrence_matrix, "Co-occurrence matrix not loaded"
        )
        self.assertEqual(
            loaded_vocab.cooccurrence_matrix.shape,
            original_vocab.cooccurrence_matrix.shape,
            "Co-occurrence matrix shape mismatch",
        )

        # Verify IDs are unique
        translation_ids = [t.id for t in loaded_vocab.translations]
        self.assertEqual(
            len(translation_ids),
            len(set(translation_ids)),
            f"Duplicate translation IDs found: {[id for id in translation_ids if translation_ids.count(id) > 1]}",
        )

        # Verify each translation has correct data
        for orig, loaded in zip(original_vocab.translations, loaded_vocab.translations):
            self.assertEqual(orig.id, loaded.id, f"ID mismatch for {orig.lemma}")
            self.assertEqual(
                orig.lemma, loaded.lemma, f"Lemma mismatch for ID {orig.id}"
            )
            self.assertEqual(orig.pos, loaded.pos, f"POS mismatch for {orig.lemma}")
            self.assertEqual(
                orig.frequency, loaded.frequency, f"Frequency mismatch for {orig.lemma}"
            )
            self.assertEqual(
                len(orig.meanings),
                len(loaded.meanings),
                f"Meanings count mismatch for {orig.lemma}",
            )

        # Verify word variant lookups work
        for variant in loaded_vocab.vocab_entries:
            # Test lookup by lemma token
            lemma_token = (variant.lemma, variant.pos)
            variant_ids = loaded_vocab.get_entry_ids_for_lemma(lemma_token)
            self.assertIn(
                variant.id, variant_ids, f"Variant ID {variant.id} not in lookup"
            )

            # Test lookup by variant ID
            retrieved = loaded_vocab.get_vocab_entry(variant.id)
            self.assertEqual(
                retrieved.lemma, variant.lemma, "Variant lookup returned wrong lemma"
            )

        # Test loading light vocabulary (without embeddings)
        vocab_light_path = Path(result.output_dir) / "vocabulary"
        loaded_light = Vocabulary.from_dir(str(vocab_light_path))

        self.assertEqual(
            len(loaded_light.translations),
            len(original_vocab.translations),
            "Light vocab translation count mismatch",
        )
        # Light version should not have co-occurrence matrix
        self.assertIsNone(
            loaded_light.cooccurrence_matrix,
            "Light vocab should not have co-occurrence matrix",
        )

        print(f"\nVocabulary load test passed!")
        print(f"  Translations: {len(loaded_vocab.translations)}")
        print(f"  Word variants: {len(loaded_vocab.vocab_entries)}")
        print(f"  Unique IDs verified: {len(set(translation_ids))}")


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
