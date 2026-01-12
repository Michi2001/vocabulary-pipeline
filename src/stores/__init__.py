"""Store classes for index-aligned sentence data."""

from src.stores.base_store import BaseStore, BaseStoreWriter
from src.stores.sentence_store import SentenceStore
from src.stores.lemmatized_store import LemmatizedStore
from src.stores.disambiguated_store import DisambiguatedStore

__all__ = [
    "BaseStore",
    "BaseStoreWriter",
    "SentenceStore",
    "LemmatizedStore",
    "DisambiguatedStore",
]
