"""Generate embeddings for translation meanings using sentence-transformers."""

import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for definition_es fields in translations."""

    def __init__(self, model_name: str, batch_size: int = 32):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        logger.info(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def generate(self, translations: list[dict]) -> list[dict]:
        """Add embedding vectors to each meaning's definition_es field."""
        # Collect all definitions with their location (lemma_idx, meaning_idx)
        definitions = []
        locations = []

        for lemma_idx, entry in enumerate(translations):
            for meaning_idx, meaning in enumerate(entry.get("meanings", [])):
                definition = meaning.get("definition_es", "")
                if definition:
                    definitions.append(definition)
                    locations.append((lemma_idx, meaning_idx))

        if not definitions:
            logger.warning("No definitions found to embed")
            return translations

        logger.info(f"Generating embeddings for {len(definitions)} definitions...")

        # Batch encode all definitions
        embeddings = self.model.encode(
            definitions,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # Write embeddings back to translations
        for (lemma_idx, meaning_idx), embedding in zip(locations, embeddings):
            translations[lemma_idx]["meanings"][meaning_idx]["embedding"] = embedding.tolist()

        logger.info(f"Added embeddings to {len(definitions)} meanings")
        return translations
