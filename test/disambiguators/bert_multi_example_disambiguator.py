"""Multi-example BERT disambiguation with averaged embeddings."""

from typing import Dict, List

import torch

from disambiguators.bert_word_embedding_disambiguator import (
    BertWordEmbeddingDisambiguator,
)
from disambiguators.base_disambiguator import DisambiguationResult


class BertMultiExampleDisambiguator(BertWordEmbeddingDisambiguator):
    """WSD using multiple example sentences per meaning, averaging their embeddings."""

    def _get_meaning_sentence(self, meaning: Dict[str, str]) -> str:
        """Not used - we override disambiguate() to handle multiple examples."""
        return meaning.get("examples", [""])[0]

    def disambiguate(
        self, target_sentence: str, meanings: List[Dict]
    ) -> DisambiguationResult:
        """
        Disambiguate using averaged embeddings from multiple examples per meaning.

        Args:
            target_sentence: Sentence with [target_word] in brackets
            meanings: List of dicts with 'examples' key (list of example sentences)

        Returns:
            DisambiguationResult with predicted index and confidence
        """
        target_word, _, _, _ = self._extract_bracketed_word(target_sentence)
        target_emb = self._get_contextual_embedding(target_sentence)

        scores = []
        for meaning in meanings:
            examples = meaning.get("examples", [])
            if not examples:
                scores.append(0.0)
                continue

            # Compute embedding for each example and average them
            example_embeddings = []
            for example in examples:
                emb = self._get_contextual_embedding(example)
                example_embeddings.append(emb)

            # Average all example embeddings
            averaged_emb = torch.stack(example_embeddings).mean(dim=0)

            similarity = self._cosine_similarity(target_emb, averaged_emb)
            scores.append(similarity)

        predicted_idx = scores.index(max(scores))
        return DisambiguationResult(
            predicted_index=predicted_idx,
            confidence=scores[predicted_idx],
            all_scores=scores,
            target_word=target_word,
            method_name=f"BERT-MultiEx-{self.model_name.split('/')[-1]}",
        )
