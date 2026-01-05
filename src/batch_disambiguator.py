"""
Batch Word Sense Disambiguator.

Disambiguates polysemous words in sentences using pre-computed meaning embeddings.
Processes sentences in batches for efficiency with large corpora.
"""

import gzip
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Generator, Set

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

from src.sentence_token_store import SentenceTokenStore, SentenceTokens

logger = logging.getLogger(__name__)


@dataclass
class DisambiguationTask:
    """A single token that needs disambiguation."""

    sentence_id: int
    token_idx: int
    lemma: str
    pos: str
    char_start: int
    char_end: int


@dataclass
class DisambiguatedToken:
    """Result of disambiguating a single token."""

    token_idx: int
    lemma: str
    pos: str
    meaning_idx: int
    word_variant_id: str
    confidence: float


class BatchDisambiguator:
    """
    Disambiguates polysemous words using BERT embeddings.

    Uses pre-computed meaning embeddings from translations.json and compares
    them with runtime embeddings extracted from corpus sentences.
    """

    def __init__(
        self,
        translations: List[Dict[str, Any]],
        model_name: str = "dccuchile/bert-base-spanish-wwm-cased",
        batch_size: int = 32,
    ):
        """
        Initialize disambiguator.

        Args:
            translations: List of translation dicts with meaning_embedding
            model_name: BERT model (must match MeaningEmbeddingGenerator)
            batch_size: Sentences per BERT batch
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

        # Build lookup structures
        self._build_lookups(translations)

    def _build_lookups(self, translations: List[Dict[str, Any]]):
        """Build lookup tables for efficient disambiguation."""
        # Meaning embeddings: {(lemma, pos): [(meaning_idx, word_variant_id, embedding), ...]}
        self.meaning_embeddings: Dict[
            Tuple[str, str], List[Tuple[int, str, np.ndarray]]
        ] = {}

        # Track which lemmas are polysemous (have >1 meaning with embeddings)
        self.polysemous_lemmas: Set[Tuple[str, str]] = set()

        # Word variant IDs for mono-semantic words: (lemma, pos) -> word_variant_id
        self.mono_variant_ids: Dict[Tuple[str, str], str] = {}

        for translation in translations:
            lemma = translation.get("lemma", "").lower()
            pos = translation.get("pos", "")
            trans_id = translation.get("id", "")

            if not lemma or not pos:
                continue

            meanings = translation.get("meanings", [])
            key = (lemma, pos)

            embeddings_for_lemma = []
            for meaning_idx, meaning in enumerate(meanings):
                embedding = meaning.get("meaning_embedding")
                if embedding is not None:
                    # Create word variant ID: translation_id + meaning index
                    variant_id = f"{trans_id}_{meaning_idx}"
                    embeddings_for_lemma.append(
                        (meaning_idx, variant_id, np.array(embedding, dtype=np.float32))
                    )

            if len(embeddings_for_lemma) > 1:
                self.meaning_embeddings[key] = embeddings_for_lemma
                self.polysemous_lemmas.add(key)
            elif len(embeddings_for_lemma) == 1:
                # Mono-semantic: store variant ID directly
                self.mono_variant_ids[key] = embeddings_for_lemma[0][1]
            elif len(meanings) == 1:
                # No embedding but single meaning - still mono-semantic
                variant_id = f"{trans_id}_0"
                self.mono_variant_ids[key] = variant_id

        logger.info(
            f"Built disambiguation lookups: {len(self.polysemous_lemmas)} polysemous lemmas, "
            f"{len(self.mono_variant_ids)} mono-semantic lemmas"
        )

    def _load_model(self):
        """Lazy load BERT model."""
        if self._model is None:
            logger.info(f"Loading BERT model for disambiguation: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()

    def _get_embedding_for_token(
        self,
        sentence_text: str,
        char_start: int,
        char_end: int,
    ) -> Optional[np.ndarray]:
        """
        Extract BERT embedding for a specific token in a sentence.

        Args:
            sentence_text: Full sentence text
            char_start: Character start position of target word
            char_end: Character end position of target word

        Returns:
            Embedding array or None if extraction fails
        """
        self._load_model()

        inputs = self._tokenizer(
            sentence_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        # Map character positions to token positions
        target_positions = []
        for char_pos in range(char_start, char_end):
            token_idx = inputs.char_to_token(0, char_pos)
            if token_idx is not None and token_idx not in target_positions:
                target_positions.append(token_idx)

        if not target_positions:
            return None

        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            embeddings = hidden_states[-2][0]  # Second-to-last layer

        word_embedding = embeddings[target_positions].mean(dim=0)
        return word_embedding.numpy()

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _disambiguate_token(
        self,
        sentence_text: str,
        token: Tuple[str, str, int, int],
        token_idx: int,
    ) -> Optional[DisambiguatedToken]:
        """
        Disambiguate a single token (mono-semantic or polysemous).

        Args:
            sentence_text: Full sentence text
            token: (lemma, pos, char_start, char_end)
            token_idx: Token index in sentence

        Returns:
            DisambiguatedToken or None if token not in vocabulary
        """
        lemma, pos, char_start, char_end = token
        key = (lemma.lower(), pos)

        # Check if mono-semantic
        if key in self.mono_variant_ids:
            return DisambiguatedToken(
                token_idx=token_idx,
                lemma=lemma,
                pos=pos,
                meaning_idx=0,
                word_variant_id=self.mono_variant_ids[key],
                confidence=1.0,
            )

        # Check if polysemous
        if key not in self.meaning_embeddings:
            return None

        # Get embedding for this token in context
        token_embedding = self._get_embedding_for_token(
            sentence_text, char_start, char_end
        )
        if token_embedding is None:
            return None

        # Compare with all meaning embeddings
        best_idx = 0
        best_variant_id = ""
        best_score = -1.0

        for meaning_idx, variant_id, meaning_emb in self.meaning_embeddings[key]:
            score = self._cosine_similarity(token_embedding, meaning_emb)
            if score > best_score:
                best_score = score
                best_idx = meaning_idx
                best_variant_id = variant_id

        return DisambiguatedToken(
            token_idx=token_idx,
            lemma=lemma,
            pos=pos,
            meaning_idx=best_idx,
            word_variant_id=best_variant_id,
            confidence=best_score,
        )

    def disambiguate_sentence(
        self,
        sentence_tokens: SentenceTokens,
        sentence_text: str,
    ) -> List[DisambiguatedToken]:
        """
        Disambiguate all tokens in a sentence.

        Args:
            sentence_tokens: Token information from SentenceTokenStore
            sentence_text: Original sentence text (needed for BERT)

        Returns:
            List of disambiguated tokens (including mono-semantic)
        """
        results = []

        for idx, token in enumerate(sentence_tokens.tokens):
            # Disambiguate token (handles both mono-semantic and polysemous)
            result = self._disambiguate_token(sentence_text, token, idx)
            if result:
                results.append(result)

        return results

    def process(
        self,
        sentence_store: SentenceTokenStore,
        corpus_reader,  # To get original sentence text
        output_path: Path,
    ):
        """
        Process all sentences and write disambiguation results.

        Args:
            sentence_store: Store with sentence-token mappings
            corpus_reader: Reader to get original sentence texts
            output_path: Path for output JSONL file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting batch disambiguation to {output_path}")

        # We need sentence texts - read corpus in parallel with token store
        sentence_texts = list(corpus_reader.read_sentences(batch_size=100000))
        flat_sentences = [s for batch in sentence_texts for s in batch]

        total_sentences = 0
        total_disambiguated = 0

        with gzip.open(output_path, "wt", encoding="utf-8") as f:
            for sentence_tokens in sentence_store.iter_sentences():
                sid = sentence_tokens.sid

                if sid >= len(flat_sentences):
                    logger.warning(f"Sentence ID {sid} out of range")
                    continue

                sentence_text = flat_sentences[sid]
                results = self.disambiguate_sentence(sentence_tokens, sentence_text)

                if results:
                    # Write result
                    variants = [
                        [r.word_variant_id, r.token_idx, r.confidence] for r in results
                    ]
                    line = json.dumps(
                        {"sid": sid, "variants": variants},
                        ensure_ascii=False,
                    )
                    f.write(line + "\n")
                    total_disambiguated += len(results)

                total_sentences += 1
                if total_sentences % 10000 == 0:
                    logger.info(
                        f"Processed {total_sentences:,} sentences, "
                        f"{total_disambiguated:,} tokens disambiguated"
                    )

        logger.info(
            f"Disambiguation complete: {total_sentences:,} sentences, "
            f"{total_disambiguated:,} tokens disambiguated"
        )


def load_disambiguations(path: Path) -> Generator[Dict[str, Any], None, None]:
    """
    Load disambiguation results from JSONL file.

    Yields:
        Dicts with 'sid' and 'variants' keys
    """
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
