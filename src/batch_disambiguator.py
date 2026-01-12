"""
Batch Word Sense Disambiguator.

Disambiguates polysemous words in sentences using pre-computed meaning embeddings.
Processes sentences in batches for efficiency with large corpora.
Uses index-aligned stores for efficient token lookup.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.stores.disambiguated_store import DisambiguatedStore
from src.stores.lemmatized_store import LemmatizedStore
from src.stores.sentence_store import SentenceStore
from src.vocab import LemmaToken, Vocabulary

logger = logging.getLogger(__name__)


class BatchDisambiguator:
    """
    Disambiguates polysemous words using BERT embeddings.

    Uses pre-computed meaning embeddings from Vocabulary and compares
    them with runtime embeddings extracted from corpus sentences.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        model_name: str = "dccuchile/bert-base-spanish-wwm-cased",
        batch_size: int = 32,
    ):
        """
        Initialize disambiguator.

        Args:
            vocabulary: Vocabulary with word variants and embeddings
            model_name: BERT model (must match MeaningEmbeddingGenerator)
            batch_size: Sentences per BERT batch
        """
        self.vocabulary = vocabulary
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

        # Build lookup structures
        self._build_lookups()

    def _build_lookups(self):
        """
        Build lookup tables using Vocabulary.variants_by_lemma_token.

        Uses integer variant IDs for direct array access.
        """
        # Use vocabulary's existing lookup: LemmaToken
        self.vocab_lemma_tokens: Set[LemmaToken] = set(
            self.vocabulary.variants_by_lemma_token.keys()
        )

        # Mono-semantic lemmas: LemmaToken -> variant_id (int)
        self.mono_variant_idx: Dict[LemmaToken, int] = {}

        # Polysemous lemmas: LemmaToken -> [(variant_id, embedding), ...]
        self.poly_variants: Dict[LemmaToken, List[Tuple[int, np.ndarray]]] = {}

        for lemma_token, variant_ids in self.vocabulary.variants_by_lemma_token.items():
            # Collect variants with embeddings
            variants_with_emb: List[Tuple[int, np.ndarray]] = []
            first_variant_id = None

            for vid in variant_ids:
                if first_variant_id is None:
                    first_variant_id = vid
                variant = self.vocabulary.word_variants[vid]
                emb = variant.bert_embedding
                if emb is not None:
                    variants_with_emb.append((vid, emb))

            if len(variants_with_emb) == 1:
                # Mono-semantic: single meaning with embedding
                self.mono_variant_idx[lemma_token] = variants_with_emb[0][0]
            elif len(variants_with_emb) > 1:
                # Polysemous: multiple meanings with embeddings
                self.poly_variants[lemma_token] = variants_with_emb
            elif first_variant_id is not None:
                # No embedding but still in vocab - treat as mono-semantic
                self.mono_variant_idx[lemma_token] = first_variant_id

        logger.info(
            f"Built disambiguation lookups: {len(self.poly_variants)} polysemous lemmas, "
            f"{len(self.mono_variant_idx)} mono-semantic lemmas, "
            f"{len(self.vocab_lemma_tokens)} total lemma tokens"
        )

    def _load_model(self):
        """Lazy load BERT model."""
        if self._model is None:
            logger.info(f"Loading BERT model for disambiguation: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()

    def _get_embedding_for_token_by_index(
        self,
        sentence_text: str,
        original_tokens: List[str],
        token_idx: int,
    ) -> Optional[np.ndarray]:
        """
        Extract BERT embedding for a token by its index.

        Uses the original tokens to reconstruct character positions.

        Args:
            sentence_text: Reconstructed sentence text
            original_tokens: List of original tokens
            token_idx: Index of target token in original_tokens

        Returns:
            Embedding array or None if extraction fails
        """
        self._load_model()

        # Calculate character position from tokens
        # Tokens are joined by space, so we calculate position
        char_start = 0
        for i in range(token_idx):
            char_start += len(original_tokens[i]) + 1  # +1 for space

        target_token = original_tokens[token_idx]
        char_end = char_start + len(target_token)

        inputs = self._tokenizer(
            sentence_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        # Map character positions to token positions
        target_positions = []
        for char_pos in range(char_start, char_end):
            bert_token_idx = inputs.char_to_token(0, char_pos)
            if bert_token_idx is not None and bert_token_idx not in target_positions:
                target_positions.append(bert_token_idx)

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
        original_tokens: List[str],
        token_idx: int,
        lemma_token: LemmaToken,
    ) -> Optional[int]:
        """
        Disambiguate a single token (mono-semantic or polysemous).

        Args:
            sentence_text: Reconstructed sentence text
            original_tokens: List of original tokens
            token_idx: Token index in sentence
            lemma_token: (lemma, pos) tuple

        Returns:
            Word variant ID (int) or None if token not in vocabulary
        """
        # Skip deleted tokens
        if lemma_token[0] == "_DELETED_":
            return None

        # Check if in vocabulary at all
        if lemma_token not in self.vocab_lemma_tokens:
            return None

        # Check if mono-semantic
        if lemma_token in self.mono_variant_idx:
            return self.mono_variant_idx[lemma_token]

        # Check if polysemous
        if lemma_token not in self.poly_variants:
            return None

        # Get embedding for this token in context
        token_embedding = self._get_embedding_for_token_by_index(
            sentence_text, original_tokens, token_idx
        )
        if token_embedding is None:
            return None

        # Compare with all meaning embeddings
        best_variant_id: Optional[int] = None
        best_score = -1.0

        for variant_id, meaning_emb in self.poly_variants[lemma_token]:
            score = self._cosine_similarity(token_embedding, meaning_emb)
            if score > best_score:
                best_score = score
                best_variant_id = variant_id

        return best_variant_id

    def process(
        self,
        sentence_store: SentenceStore,
        lemmatized_store: LemmatizedStore,
        output_path: Path,
        use_corrections: bool = True,
    ) -> DisambiguatedStore:
        """
        Process sentences and write disambiguation results.

        Args:
            sentence_store: Store with original tokens
            lemmatized_store: Store with lemmatized tokens
            output_path: Path for DisambiguatedStore output file
            use_corrections: Whether to use corrected lemmas (if available)

        Returns:
            DisambiguatedStore pointing to the created file
        """
        logger.info("Starting batch disambiguation...")

        # Determine which lemma source to use
        if use_corrections and lemmatized_store.has_corrections():
            logger.info("Using corrected lemmas for disambiguation")
            lemma_iterator = lemmatized_store.iter_corrected_with_sentences(
                sentence_store
            )
        else:
            logger.info("Using original lemmas for disambiguation")
            lemma_iterator = lemmatized_store.iter_with_sentences(sentence_store)

        total_sentences = 0
        total_disambiguated = 0

        disambiguated_store = DisambiguatedStore(output_path)

        with disambiguated_store.open_writer() as writer:
            for sid, original_tokens, lemma_tokens in lemma_iterator:
                # Reconstruct sentence text from original tokens
                sentence_text = " ".join(original_tokens)

                # Disambiguate each token
                variants: List[Optional[int]] = []
                for idx, lemma_token in enumerate(lemma_tokens):
                    variant_id = self._disambiguate_token(
                        sentence_text, original_tokens, idx, lemma_token
                    )
                    variants.append(variant_id)
                    if variant_id is not None:
                        total_disambiguated += 1

                writer.write(sid, variants)
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

        return disambiguated_store
