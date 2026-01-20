"""
Meaning Embedding Generator for Word Sense Disambiguation.

Pre-computes BERT embeddings for target words in example sentences.
Uses metadata (_target_start, _target_end) extracted by LemmaTranslator.
These embeddings are used later by the BatchDisambiguator for WSD.
"""

import logging
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from src.vocab import Translation

logger = logging.getLogger(__name__)


class MeaningEmbeddingGenerator:
    """
    Generates BERT embeddings for target words in example sentences.

    Uses the same BERT model and embedding extraction strategy as the disambiguator
    to ensure consistency between pre-computed meaning embeddings and runtime embeddings.
    """

    def __init__(
        self,
        model_name: str = "dccuchile/bert-base-spanish-wwm-cased",
        batch_size: int = 32,
    ):
        """
        Initialize with BERT model.

        Args:
            model_name: HuggingFace model identifier (default: BETO)
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load BERT model and tokenizer."""
        if self._model is None:
            logger.info(f"Loading BERT model: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            logger.info("BERT model loaded successfully")

    def _get_contextual_embedding(
        self, sentence: str, char_start: int, char_end: int, target_word: str
    ) -> Optional[List[float]]:
        """
        Get BERT contextual embedding for target word.

        Uses second-to-last hidden layer and averages subtoken embeddings.

        Args:
            sentence: Sentence text (without brackets)
            char_start: Character start position of target word
            char_end: Character end position of target word
            target_word: Target word text (for logging only)

        Returns:
            Embedding as list of floats, or None if extraction fails
        """
        self._load_model()

        # Tokenize sentence
        inputs = self._tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        # Find token positions for target word using character-to-token mapping
        target_positions = []
        for char_pos in range(char_start, char_end):
            token_idx = inputs.char_to_token(0, char_pos)
            if token_idx is not None and token_idx not in target_positions:
                target_positions.append(token_idx)

        if not target_positions:
            logger.warning(
                f"Target word '{target_word}' not found in tokenized sentence"
            )
            return None

        # Get BERT embeddings from second-to-last hidden layer
        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            # Use second-to-last layer (index -2)
            embeddings = hidden_states[-2][0]  # [seq_len, hidden_dim]

        # Average embeddings of target word tokens
        word_embedding = embeddings[target_positions].mean(dim=0)

        return word_embedding.tolist()

    def generate(self, translations: List[Translation]) -> List[Translation]:
        """
        Add averaged BERT embedding from multiple examples per meaning.

        Expects that LemmaTranslator has already extracted and stored:
        - _target_word: The target word
        - _target_starts: Character start positions (list, one per example)
        - _target_ends: Character end positions (list, one per example)
        - examples_es: List of example sentences

        Modifies Translation objects in-place.

        Args:
            translations: List of Translation objects

        Returns:
            Same list with averaged embeddings added to meanings
        """
        logger.info(f"Generating meaning embeddings for {len(translations)} lemmas...")

        total_meanings = 0
        successful_embeddings = 0
        missing_metadata = 0
        total_examples_processed = 0

        for translation in translations:
            if not translation.meanings:
                continue

            for meaning in translation.meanings:
                total_meanings += 1

                if not meaning.examples_es:
                    continue

                # Check if metadata exists (extracted by LemmaTranslator)
                if meaning._target_starts is None or meaning._target_ends is None:
                    missing_metadata += 1
                    continue

                target_word = meaning._target_word or "unknown"

                # Collect embeddings from all valid examples
                embeddings = []
                for i, example in enumerate(meaning.examples_es):
                    if i >= len(meaning._target_starts) or i >= len(meaning._target_ends):
                        break

                    char_start = meaning._target_starts[i]
                    char_end = meaning._target_ends[i]

                    # Skip examples with missing position data
                    if char_start is None or char_end is None:
                        continue

                    embedding = self._get_contextual_embedding(
                        example, char_start, char_end, target_word
                    )
                    if embedding is not None:
                        embeddings.append(embedding)
                        total_examples_processed += 1

                # Average all embeddings for this meaning
                if embeddings:
                    averaged = np.mean(embeddings, axis=0).tolist()
                    meaning._bert_embedding = averaged
                    successful_embeddings += 1

        logger.info(
            f"Meaning embeddings generated: {successful_embeddings}/{total_meanings} "
            f"(from {total_examples_processed} examples, missing metadata: {missing_metadata})"
        )

        return translations
