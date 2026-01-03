"""Definition-based BERT disambiguation."""

from typing import Dict
from .bert_word_embedding_disambiguator import BertWordEmbeddingDisambiguator


class BertDefinitionDisambiguator(BertWordEmbeddingDisambiguator):
    """WSD using definitions with bracketed words."""

    def _get_meaning_sentence(self, meaning: Dict[str, str]) -> str:
        """Return definition field from meaning dict."""
        return meaning["definition"]
