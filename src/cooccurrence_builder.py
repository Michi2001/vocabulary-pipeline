"""
Co-Occurrence Matrix Builder.

Builds a sparse co-occurrence matrix from disambiguated sentences.
Matrix dimensions are num_word_variants x num_word_variants.
Variant IDs (integers) are used directly as matrix indices.
"""

import logging
from typing import List, Set, Tuple
import numpy as np

from scipy import sparse

from src.stores.disambiguated_store import DisambiguatedStore
from src.vocab import Vocabulary

logger = logging.getLogger(__name__)


class CoOccurrenceBuilder:
    """
    Builds sparse co-occurrence matrix from disambiguated sentences.

    Co-occurrence is defined at sentence level: all word variants in the same
    sentence are considered co-occurrent.

    Since WordVariant.id is now an integer index into Vocabulary.word_variants,
    no separate mapping is needed - variant IDs are used directly as matrix indices.
    """

    def __init__(self, vocabulary: Vocabulary):
        """
        Initialize with vocabulary.

        Args:
            vocabulary: Vocabulary with word variants
        """
        self.vocabulary = vocabulary
        self.n_variants = len(vocabulary.word_variants)
        logger.info(
            f"CoOccurrenceBuilder initialized with {self.n_variants} word variants"
        )

    def build(
        self, disambiguated_stores: List[DisambiguatedStore]
    ) -> sparse.csr_matrix:
        """
        Build co-occurrence matrix from multiple disambiguated stores.

        Args:
            disambiguated_stores: List of DisambiguatedStore objects (one per corpus)

        Returns:
            Sparse co-occurrence matrix (symmetric)
        """
        n = self.n_variants
        logger.info(
            f"Building {n}x{n} co-occurrence matrix from {len(disambiguated_stores)} corpora..."
        )

        # Use lil_matrix for efficient incremental construction
        matrix = sparse.lil_matrix((n, n), dtype=np.float32)

        total_sentence_count = 0
        total_pair_count = 0

        # Process all stores
        for store_idx, store in enumerate(disambiguated_stores, 1):
            logger.info(f"Processing store {store_idx}/{len(disambiguated_stores)}")

            sentence_count = 0
            pair_count = 0

            for sid, variants in store:
                # Filter out None values (tokens not in vocabulary)
                # variants are already integers (variant IDs = matrix indices)
                variant_ids = [v for v in variants if v is not None]

                # Count co-occurrences (all pairs)
                seen_pairs: Set[Tuple[int, int]] = set()
                for i, idx_i in enumerate(variant_ids):
                    for j, idx_j in enumerate(variant_ids):
                        if i >= j:  # Skip self and duplicates (symmetric)
                            continue

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
                        f"  Store {store_idx}: {sentence_count:,} sentences, "
                        f"{pair_count:,} pairs"
                    )

            total_sentence_count += sentence_count
            total_pair_count += pair_count
            logger.info(
                f"  Store {store_idx} complete: {sentence_count:,} sentences, "
                f"{pair_count:,} pairs"
            )

        logger.info(
            f"Matrix construction complete: {total_sentence_count:,} total sentences from "
            f"{len(disambiguated_stores)} corpora, {total_pair_count:,} unique co-occurrence pairs"
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
