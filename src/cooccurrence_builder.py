"""
Co-Occurrence Matrix Builder.

Builds a sparse co-occurrence matrix from disambiguated sentences.
Matrix dimensions are num_word_variants x num_word_variants.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

import numpy as np
from scipy import sparse

from src.vocab import Vocabulary
from src.sentence_store import SentenceStore

logger = logging.getLogger(__name__)


class CoOccurrenceBuilder:
    """
    Builds sparse co-occurrence matrix from disambiguated sentences.

    Co-occurrence is defined at sentence level: all word variants in the same
    sentence are considered co-occurrent.
    """

    def __init__(self, vocabulary: Vocabulary):
        """
        Initialize with vocabulary to build variant index.

        Args:
            vocabulary: Vocabulary with word variants
        """
        self.vocabulary = vocabulary
        self._build_variant_index()

    def _build_variant_index(self):
        """Build mapping from word variant IDs to matrix indices."""
        # Use word variants from vocabulary (already sorted by frequency)
        self.variant_ids: List[str] = [v.id for v in self.vocabulary.word_variants]
        self.variant_info: Dict[str, Dict[str, Any]] = {}

        for variant in self.vocabulary.word_variants:
            self.variant_info[variant.id] = {
                "lemma": variant.lemma,
                "pos": variant.pos,
                "meaning_idx": variant.meaning_idx,
                "translation_id": variant.source_translation_id,
            }

        # Sort for consistent ordering
        self.variant_ids.sort()

        # Build index mapping
        self.variant_to_idx: Dict[str, int] = {
            vid: idx for idx, vid in enumerate(self.variant_ids)
        }

        logger.info(f"Built variant index with {len(self.variant_ids)} word variants")

    def build(self, sentence_store: SentenceStore) -> sparse.csr_matrix:
        """
        Build co-occurrence matrix from disambiguated sentences in store.

        Args:
            sentence_store: SentenceStore with disambiguated data

        Returns:
            Sparse co-occurrence matrix (symmetric)
        """
        if not sentence_store.has_disambiguated():
            raise ValueError("SentenceStore has no disambiguation data")

        n = len(self.variant_ids)
        logger.info(f"Building {n}x{n} co-occurrence matrix...")

        # Use lil_matrix for efficient incremental construction
        matrix = sparse.lil_matrix((n, n), dtype=np.float32)

        sentence_count = 0
        pair_count = 0

        for sid, variants in sentence_store.iter_disambiguated():
            # Filter out None values (tokens not in vocabulary)
            variant_ids_in_sentence = [v for v in variants if v is not None]

            # Count co-occurrences (all pairs)
            seen_pairs: Set[Tuple[int, int]] = set()
            for i, vid_i in enumerate(variant_ids_in_sentence):
                if vid_i not in self.variant_to_idx:
                    continue
                idx_i = self.variant_to_idx[vid_i]

                for j, vid_j in enumerate(variant_ids_in_sentence):
                    if i >= j:  # Skip self and duplicates (symmetric)
                        continue
                    if vid_j not in self.variant_to_idx:
                        continue
                    idx_j = self.variant_to_idx[vid_j]

                    # Ensure consistent ordering for symmetric matrix
                    if idx_i > idx_j:
                        idx_i, idx_j = idx_j, idx_i

                    pair_key = (idx_i, idx_j)
                    if pair_key not in seen_pairs:
                        matrix[idx_i, idx_j] += 1
                        matrix[idx_j, idx_i] += 1  # Symmetric
                        seen_pairs.add(pair_key)
                        pair_count += 1

            sentence_count += 1
            if sentence_count % 50000 == 0:
                logger.info(
                    f"Processed {sentence_count:,} sentences, "
                    f"{pair_count:,} co-occurrence pairs"
                )

        logger.info(
            f"Matrix construction complete: {sentence_count:,} sentences, "
            f"{pair_count:,} unique co-occurrence pairs"
        )

        # Convert to CSR for efficient storage and operations
        csr_matrix = matrix.tocsr()

        # Log sparsity statistics
        nnz = csr_matrix.nnz
        total = n * n
        sparsity = 1 - (nnz / total) if total > 0 else 1
        logger.info(
            f"Matrix statistics: {nnz:,} non-zero entries, {sparsity:.4%} sparsity"
        )

        return csr_matrix

    def save(self, matrix: sparse.csr_matrix, output_path: Path):
        """
        Save matrix and metadata to disk.

        Args:
            matrix: Co-occurrence matrix
            output_path: Path for .npz file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save matrix
        sparse.save_npz(output_path, matrix)
        logger.info(f"Saved co-occurrence matrix to {output_path}")

        # Save variant index as JSON
        index_path = output_path.with_suffix(".json")
        index_data = {
            "variant_ids": self.variant_ids,
            "variant_info": self.variant_info,
        }
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved variant index to {index_path}")

    @classmethod
    def load(cls, matrix_path: Path) -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
        """
        Load matrix and metadata from disk.

        Args:
            matrix_path: Path to .npz file

        Returns:
            Tuple of (matrix, index_data)
        """
        matrix_path = Path(matrix_path)

        matrix = sparse.load_npz(matrix_path)

        index_path = matrix_path.with_suffix(".json")
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        return matrix, index_data

    def get_cooccurrences(
        self,
        matrix: sparse.csr_matrix,
        variant_id: str,
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Get top co-occurring variants for a given variant.

        Args:
            matrix: Co-occurrence matrix
            variant_id: Word variant ID to query
            top_k: Number of top co-occurrences to return

        Returns:
            List of (variant_id, count) tuples, sorted by count descending
        """
        if variant_id not in self.variant_to_idx:
            return []

        idx = self.variant_to_idx[variant_id]
        row = matrix.getrow(idx).toarray().flatten()

        # Get top-k indices
        top_indices = np.argsort(row)[-top_k:][::-1]

        results = []
        for i in top_indices:
            if row[i] > 0:
                results.append((self.variant_ids[i], float(row[i])))

        return results
