"""Example-based BERT disambiguation."""

from typing import Dict
from .bert_word_embedding_disambiguator import BertWordEmbeddingDisambiguator


class BertExampleDisambiguator(BertWordEmbeddingDisambiguator):
    """WSD using example sentences with bracketed words."""

    def _get_meaning_sentence(self, meaning: Dict[str, str]) -> str:
        """Return example field from meaning dict."""
        return meaning["example"]
