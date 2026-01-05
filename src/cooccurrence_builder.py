"""
Co-Occurrence Matrix Builder.

Builds a sparse co-occurrence matrix from disambiguated sentences.
Matrix dimensions are num_word_variants x num_word_variants.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

import numpy as np
from scipy import sparse

from src.batch_disambiguator import load_disambiguations

logger = logging.getLogger(__name__)


class CoOccurrenceBuilder:
    """
    Builds sparse co-occurrence matrix from disambiguated sentences.

    Co-occurrence is defined at sentence level: all word variants in the same
    sentence are considered co-occurrent.
    """

    def __init__(self, translations: List[Dict[str, Any]]):
        """
        Initialize with translation data to build variant index.

        Args:
            translations: List of translation dicts
        """
        self._build_variant_index(translations)

    def _build_variant_index(self, translations: List[Dict[str, Any]]):
        """Build mapping from word variant IDs to matrix indices."""
        # Collect all possible variant IDs
        self.variant_ids: List[str] = []
        self.variant_info: Dict[str, Dict[str, Any]] = {}

        for translation in translations:
            trans_id = translation.get("id", "")
            lemma = translation.get("lemma", "")
            pos = translation.get("pos", "")

            meanings = translation.get("meanings", [])
            for meaning_idx, meaning in enumerate(meanings):
                variant_id = f"{trans_id}_{meaning_idx}"
                self.variant_ids.append(variant_id)
                self.variant_info[variant_id] = {
                    "lemma": lemma,
                    "pos": pos,
                    "meaning_idx": meaning_idx,
                    "translation_id": trans_id,
                }

        # Sort for consistent ordering
        self.variant_ids.sort()

        # Build index mapping
        self.variant_to_idx: Dict[str, int] = {
            vid: idx for idx, vid in enumerate(self.variant_ids)
        }

        logger.info(f"Built variant index with {len(self.variant_ids)} word variants")

    def build(self, disambiguations_path: Path) -> sparse.csr_matrix:
        """
        Build co-occurrence matrix from disambiguation results.

        Args:
            disambiguations_path: Path to sentence_disambiguations.jsonl.gz

        Returns:
            Sparse co-occurrence matrix (symmetric)
        """
        n = len(self.variant_ids)
        logger.info(f"Building {n}x{n} co-occurrence matrix...")

        # Use lil_matrix for efficient incremental construction
        matrix = sparse.lil_matrix((n, n), dtype=np.float32)

        sentence_count = 0
        pair_count = 0

        for disambiguation in load_disambiguations(disambiguations_path):
            variants = disambiguation.get("variants", [])

            # Extract variant IDs from this sentence
            variant_ids_in_sentence: List[str] = []
            for variant_data in variants:
                # Format: [variant_id, token_idx, confidence]
                variant_id = variant_data[0]
                if variant_id in self.variant_to_idx:
                    variant_ids_in_sentence.append(variant_id)

            # Count co-occurrences (all pairs)
            seen_pairs: Set[Tuple[int, int]] = set()
            for i, vid_i in enumerate(variant_ids_in_sentence):
                idx_i = self.variant_to_idx[vid_i]
                for j, vid_j in enumerate(variant_ids_in_sentence):
                    if i >= j:  # Skip self and duplicates (symmetric)
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
            f"Matrix statistics: {nnz:,} non-zero entries, " f"{sparsity:.4%} sparsity"
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
        import json

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
        import json

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
