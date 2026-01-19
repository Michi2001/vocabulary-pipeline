"""Word sense disambiguation implementations."""

from .base_disambiguator import BaseDisambiguator, DisambiguationResult
from .bert_word_embedding_disambiguator import BertWordEmbeddingDisambiguator
from .bert_definition_disambiguator import BertDefinitionDisambiguator
from .bert_example_disambiguator import BertExampleDisambiguator
from .bert_multi_example_disambiguator import BertMultiExampleDisambiguator

__all__ = [
    "BaseDisambiguator",
    "DisambiguationResult",
    "BertWordEmbeddingDisambiguator",
    "BertDefinitionDisambiguator",
    "BertExampleDisambiguator",
    "BertMultiExampleDisambiguator",
]
