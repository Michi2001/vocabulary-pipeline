"""
Integration test for the full pipeline.

Tests the pipeline with a small sample corpus and mock translations
to verify co-occurrence matrix generation and vocabulary structure.
"""

import json
import shutil
import unittest
from pathlib import Path
from typing import List

from main import load_config, run_pipeline
from src.vocab import Translation, Meaning


class TestPipelineIntegration(unittest.TestCase):
    """Integration test for pipeline with mock translations."""

    test_dir = Path(__file__).parent
    config_path = test_dir / "test_config.yaml"
    cache_dir = test_dir / "cache"
    output_dir = test_dir / "output"

    @classmethod
    def setUpClass(cls):
        """Clean up previous test runs."""
        if cls.cache_dir.exists():
            shutil.rmtree(cls.cache_dir)
        if cls.output_dir.exists():
            shutil.rmtree(cls.output_dir)

    def _create_mock_translations(self) -> List[Translation]:
        """Create mock Translation objects with embeddings for test corpus words."""
        translations = []

        # Mock embedding (768-dim for BERT)
        emb_gato = [0.1] * 768
        emb_perro = [0.2] * 768
        emb_comer = [0.3] * 768
        emb_pescado = [0.4] * 768
        emb_carne = [0.5] * 768
        emb_dormir = [0.6] * 768
        emb_casa = [0.7] * 768
        emb_jardin = [0.8] * 768
        emb_jugar = [0.9] * 768

        # gato - appears in sentences 1, 3, 6
        translations.append(
            Translation(
                id="000001",
                lemma="gato",
                pos="NOUN",
                frequency=3,
                meanings=[
                    Meaning(
                        definition_es="Animal felino domestico",
                        estimated_share=1.0,
                        translations=[{"text": "Katze"}],
                        example_es="El gato come pescado.",
                        _bert_embedding=emb_gato,
                        _target_word="gato",
                        _target_start=3,
                        _target_end=7,
                    )
                ],
            )
        )

        # perro - appears in sentences 2, 3, 7
        translations.append(
            Translation(
                id="000002",
                lemma="perro",
                pos="NOUN",
                frequency=3,
                meanings=[
                    Meaning(
                        definition_es="Animal canino domestico",
                        estimated_share=1.0,
                        translations=[{"text": "Hund"}],
                        example_es="El perro come carne.",
                        _bert_embedding=emb_perro,
                        _target_word="perro",
                        _target_start=3,
                        _target_end=8,
                    )
                ],
            )
        )

        # comer - appears in sentences 1, 2
        translations.append(
            Translation(
                id="000003",
                lemma="comer",
                pos="VERB",
                frequency=2,
                meanings=[
                    Meaning(
                        definition_es="Ingerir alimentos",
                        estimated_share=1.0,
                        translations=[{"text": "essen"}],
                        example_es="El gato come pescado.",
                        _bert_embedding=emb_comer,
                        _target_word="come",
                        _target_start=9,
                        _target_end=13,
                    )
                ],
            )
        )

        # pescado - appears in sentences 1, 4
        translations.append(
            Translation(
                id="000004",
                lemma="pescado",
                pos="NOUN",
                frequency=2,
                meanings=[
                    Meaning(
                        definition_es="Pez preparado como alimento",
                        estimated_share=1.0,
                        translations=[{"text": "Fisch"}],
                        example_es="El pescado es fresco.",
                        _bert_embedding=emb_pescado,
                        _target_word="pescado",
                        _target_start=3,
                        _target_end=10,
                    )
                ],
            )
        )

        # carne - appears in sentences 2, 5
        translations.append(
            Translation(
                id="000005",
                lemma="carne",
                pos="NOUN",
                frequency=2,
                meanings=[
                    Meaning(
                        definition_es="Tejido muscular de animales",
                        estimated_share=1.0,
                        translations=[{"text": "Fleisch"}],
                        example_es="La carne es roja.",
                        _bert_embedding=emb_carne,
                        _target_word="carne",
                        _target_start=3,
                        _target_end=8,
                    )
                ],
            )
        )

        # dormir - appears in sentences 6, 7
        translations.append(
            Translation(
                id="000006",
                lemma="dormir",
                pos="VERB",
                frequency=2,
                meanings=[
                    Meaning(
                        definition_es="Estar en estado de reposo",
                        estimated_share=1.0,
                        translations=[{"text": "schlafen"}],
                        example_es="El gato duerme.",
                        _bert_embedding=emb_dormir,
                        _target_word="duerme",
                        _target_start=9,
                        _target_end=15,
                    )
                ],
            )
        )

        # casa - appears in sentences 6, 8
        translations.append(
            Translation(
                id="000007",
                lemma="casa",
                pos="NOUN",
                frequency=2,
                meanings=[
                    Meaning(
                        definition_es="Edificio para habitar",
                        estimated_share=1.0,
                        translations=[{"text": "Haus"}],
                        example_es="La casa es grande.",
                        _bert_embedding=emb_casa,
                        _target_word="casa",
                        _target_start=3,
                        _target_end=7,
                    )
                ],
            )
        )

        # jardin - appears in sentences 7, 8
        translations.append(
            Translation(
                id="000008",
                lemma="jardin",
                pos="NOUN",
                frequency=2,
                meanings=[
                    Meaning(
                        definition_es="Terreno con plantas",
                        estimated_share=1.0,
                        translations=[{"text": "Garten"}],
                        example_es="El jardin es grande.",
                        _bert_embedding=emb_jardin,
                        _target_word="jardin",
                        _target_start=3,
                        _target_end=9,
                    )
                ],
            )
        )

        # jugar - appears in sentence 3
        translations.append(
            Translation(
                id="000009",
                lemma="jugar",
                pos="VERB",
                frequency=1,
                meanings=[
                    Meaning(
                        definition_es="Hacer algo para divertirse",
                        estimated_share=1.0,
                        translations=[{"text": "spielen"}],
                        example_es="El gato y el perro juegan.",
                        _bert_embedding=emb_jugar,
                        _target_word="juegan",
                        _target_start=20,
                        _target_end=26,
                    )
                ],
            )
        )

        return translations

    def test_pipeline_runs_without_errors(self):
        """Test that pipeline completes without errors."""
        config = load_config(str(self.config_path))
        mock_translations = self._create_mock_translations()

        result = run_pipeline(config, translations_override=mock_translations)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.output_dir)
        self.assertTrue(Path(result.output_dir).exists())

    def test_vocabulary_files_created(self):
        """Test that vocabulary output files are created."""
        config = load_config(str(self.config_path))
        mock_translations = self._create_mock_translations()

        result = run_pipeline(config, translations_override=mock_translations)

        vocab_dir = Path(result.output_dir) / "vocabulary"
        self.assertTrue(vocab_dir.exists(), "vocabulary directory should exist")

        translations_file = vocab_dir / "translations.json"
        self.assertTrue(translations_file.exists(), "translations.json should exist")

        cooccurrence_file = vocab_dir / "cooccurrence_matrix.npz"
        self.assertTrue(
            cooccurrence_file.exists(), "cooccurrence_matrix.npz should exist"
        )

    def test_vocabulary_structure(self):
        """Test that vocabulary has expected structure."""
        config = load_config(str(self.config_path))
        mock_translations = self._create_mock_translations()

        result = run_pipeline(config, translations_override=mock_translations)

        self.assertIsNotNone(result.vocabulary)
        self.assertGreater(len(result.vocabulary.translations), 0)
        self.assertGreater(len(result.vocabulary.word_variants), 0)

        # Check translations.json content
        vocab_dir = Path(result.output_dir) / "vocabulary"
        with open(vocab_dir / "translations.json", "r", encoding="utf-8") as f:
            saved_translations = json.load(f)

        self.assertIsInstance(saved_translations, list)
        self.assertGreater(len(saved_translations), 0)

        first = saved_translations[0]
        self.assertIn("id", first)
        self.assertIn("lemma", first)
        self.assertIn("pos", first)
        self.assertIn("meanings", first)

    def test_cooccurrence_matrix_gato_perro(self):
        """Test that gato and perro co-occur (both in sentence 3)."""
        config = load_config(str(self.config_path))
        mock_translations = self._create_mock_translations()

        result = run_pipeline(config, translations_override=mock_translations)

        self.assertIsNotNone(result.cooccurrence_matrix)
        self.assertIsNotNone(result.vocabulary)

        vocab = result.vocabulary
        matrix = result.cooccurrence_matrix

        gato_ids = vocab.get_variant_ids_for_lemma(("gato", "NOUN"))
        perro_ids = vocab.get_variant_ids_for_lemma(("perro", "NOUN"))

        self.assertTrue(len(gato_ids) > 0, "gato should be in vocabulary")
        self.assertTrue(len(perro_ids) > 0, "perro should be in vocabulary")

        gato_id = gato_ids[0]
        perro_id = perro_ids[0]

        # gato and perro co-occur in "El gato y el perro juegan juntos"
        cooccurrence = matrix[gato_id, perro_id]
        self.assertGreater(
            cooccurrence, 0, "gato and perro should co-occur at least once"
        )

    def test_cooccurrence_matrix_gato_comer(self):
        """Test that gato and comer co-occur (both in sentence 1)."""
        config = load_config(str(self.config_path))
        mock_translations = self._create_mock_translations()

        result = run_pipeline(config, translations_override=mock_translations)

        vocab = result.vocabulary
        matrix = result.cooccurrence_matrix

        gato_ids = vocab.get_variant_ids_for_lemma(("gato", "NOUN"))
        comer_ids = vocab.get_variant_ids_for_lemma(("comer", "VERB"))

        self.assertTrue(len(gato_ids) > 0, "gato should be in vocabulary")
        self.assertTrue(len(comer_ids) > 0, "comer should be in vocabulary")

        gato_id = gato_ids[0]
        comer_id = comer_ids[0]

        # gato and comer co-occur in "El gato come pescado fresco"
        cooccurrence = matrix[gato_id, comer_id]
        self.assertGreater(
            cooccurrence, 0, "gato and comer should co-occur at least once"
        )

    def test_cooccurrence_matrix_casa_jardin(self):
        """Test that casa and jardin co-occur (both in sentence 8)."""
        config = load_config(str(self.config_path))
        mock_translations = self._create_mock_translations()

        result = run_pipeline(config, translations_override=mock_translations)

        vocab = result.vocabulary
        matrix = result.cooccurrence_matrix

        casa_ids = vocab.get_variant_ids_for_lemma(("casa", "NOUN"))
        jardin_ids = vocab.get_variant_ids_for_lemma(("jardin", "NOUN"))

        self.assertTrue(len(casa_ids) > 0, "casa should be in vocabulary")
        self.assertTrue(len(jardin_ids) > 0, "jardin should be in vocabulary")

        casa_id = casa_ids[0]
        jardin_id = jardin_ids[0]

        # casa and jardin co-occur in "La casa tiene un jardin grande"
        cooccurrence = matrix[casa_id, jardin_id]
        self.assertGreater(
            cooccurrence, 0, "casa and jardin should co-occur at least once"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
