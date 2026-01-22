"""Multi-example BERT disambiguation with averaged embeddings."""

from typing import Dict, List

import torch

from disambiguators.bert_word_embedding_disambiguator import (
    BertWordEmbeddingDisambiguator,
)
from disambiguators.base_disambiguator import DisambiguationResult


class BertMultiExampleDisambiguator(BertWordEmbeddingDisambiguator):
    """WSD using multiple example sentences per meaning, averaging their embeddings."""

    def __init__(self, model_name: str = "bert-base-uncased", min_score_margin: float = 0.0):
        """
        Args:
            model_name: BERT model name
            min_score_margin: Minimum difference between best and second-best score.
                              If margin is smaller, returns predicted_index=-1 (ambiguous).
        """
        super().__init__(model_name)
        self.min_score_margin = min_score_margin

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

        # Find best and second-best scores
        sorted_scores = sorted(scores, reverse=True)
        best_score = sorted_scores[0]
        second_best_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        margin = best_score - second_best_score

        # Check if margin is sufficient
        if margin < self.min_score_margin:
            predicted_idx = -1  # Ambiguous, no confident prediction
        else:
            predicted_idx = scores.index(best_score)

        return DisambiguationResult(
            predicted_index=predicted_idx,
            confidence=best_score,
            all_scores=scores,
            target_word=target_word,
            method_name=f"BERT-MultiEx-{self.model_name.split('/')[-1]}",
        )
