"""
Co-Occurrence Matrix Builder.

Builds a sparse co-occurrence matrix from disambiguated sentences.
Matrix dimensions are num_vocab_entries x num_vocab_entries.
Entry IDs (integers) are used directly as matrix indices.

Supports mixed entries: disambiguated (specific meanings) and
non-disambiguated (entire lemmas).
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

    Co-occurrence is defined at sentence level: all vocab entries in the same
    sentence are considered co-occurrent.

    Supports mixed entries:
    - Disambiguated: specific meaning of a polysemous word
    - Non-disambiguated: entire lemma (all meanings treated as one)

    Entry IDs are used directly as matrix indices.
    """

    def __init__(self, vocabulary: Vocabulary):
        """
        Initialize with vocabulary.

        Args:
            vocabulary: Vocabulary with vocab entries
        """
        self.vocabulary = vocabulary
        self.n_entries = len(vocabulary.vocab_entries)
        disamb_count = sum(1 for e in vocabulary.vocab_entries if e.is_disambiguated)
        non_disamb_count = self.n_entries - disamb_count
        logger.info(
            f"CoOccurrenceBuilder initialized with {self.n_entries} vocab entries "
            f"({disamb_count} disambiguated, {non_disamb_count} non-disambiguated)"
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
        n = self.n_entries
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

            for sid, entries in store:
                # Filter out None values (tokens not in vocabulary)
                # entries are already integers (entry IDs = matrix indices)
                entry_ids = [e for e in entries if e is not None]

                # Count co-occurrences (all pairs)
                seen_pairs: Set[Tuple[int, int]] = set()
                for i, id_i in enumerate(entry_ids):
                    for j in range(i + 1, len(entry_ids)):
                        id_j = entry_ids[j]

                        # Consistent ordering for symmetric matrix
                        a, b = (id_i, id_j) if id_i < id_j else (id_j, id_i)

                        if (a, b) not in seen_pairs:
                            matrix[a, b] += 1
                            matrix[b, a] += 1  # Symmetric
                            seen_pairs.add((a, b))
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
