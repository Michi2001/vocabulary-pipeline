"""Corpus processor with separate stores for sentences, lemmas, and frequencies."""

import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

from src.corpus_readers.base_reader import BaseCorpusReader
from src.frequency_counter import FrequencyCounter
from src.lemmatizer import Lemmatizer
from src.stores.lemmatized_store import LemmatizedStore
from src.stores.sentence_store import SentenceStore
from src.vocab import LemmaToken

logger = logging.getLogger(__name__)


class CorpusProcessor:
    """Processes a single corpus and returns stores and frequency data."""

    def __init__(
        self,
        corpus_config: Dict[str, Any],
        reader: BaseCorpusReader,
        lemmatizer: Lemmatizer,
        processing_config: Dict[str, Any],
        cache_config: Dict[str, Any] = None,
    ):
        self.corpus_config = corpus_config
        self.reader = reader
        self.lemmatizer = lemmatizer
        self.processing_config = processing_config
        self.cache_config = cache_config or {"enabled": False}

        self.corpus_name = corpus_config.get(
            "name", Path(corpus_config["file_path"]).stem
        )

    def _generate_cache_key(self) -> str:
        """
        Generate cache key from configuration (excludes filtering, batch sizes).
        The key is based on:
        - Corpus filename
        - Corpus file modification time
        - Lemmatizer language model
        - Max sentences to process
        """
        corpus_path = self.corpus_config["file_path"]
        corpus_filename = Path(corpus_path).name
        file_mtime = os.path.getmtime(corpus_path) if os.path.exists(corpus_path) else 0
        max_sentences = self.corpus_config.get("max_sentences", None)

        key_components = [
            corpus_filename,
            str(file_mtime),
            self.lemmatizer.language_model,
            str(max_sentences),
        ]

        key_string = "|".join(key_components)
        return hashlib.sha256(key_string.encode()).hexdigest()[:12]

    def get_cache_key(self) -> str:
        """Get the cache key for this corpus."""
        return self._generate_cache_key()

    def get_cache_dir(self) -> Path:
        """Get cache directory for this corpus and cache key."""
        cache_base = Path(self.cache_config.get("directory", "cache"))
        return cache_base / self.corpus_name / self._generate_cache_key()

    def _get_paths(self) -> Tuple[Path, Path, Path]:
        """Get paths for all three cached artifacts."""
        cache_dir = self.get_cache_dir()
        return (
            cache_dir / "SentenceStore.jsonl.gz",
            cache_dir / "LemmatizedStore.jsonl.gz",
            cache_dir / "FrequencyCounter.pkl",
        )

    def _all_cached(self) -> bool:
        """Check if all three artifacts exist in cache."""
        sent_path, lemma_path, freq_path = self._get_paths()
        return sent_path.exists() and lemma_path.exists() and freq_path.exists()

    def _load_from_cache(
        self,
    ) -> Tuple[SentenceStore, LemmatizedStore, FrequencyCounter]:
        """Load all three artifacts from cache."""
        sent_path, lemma_path, freq_path = self._get_paths()

        sentence_store = SentenceStore(sent_path)
        lemmatized_store = LemmatizedStore(lemma_path)

        with open(freq_path, "rb") as f:
            freq_data = pickle.load(f)
        counter = FrequencyCounter.from_dict(freq_data)

        cache_key = self._generate_cache_key()
        logger.info(f"Loaded from cache: {self.corpus_name} (key: {cache_key})")
        return sentence_store, lemmatized_store, counter

    def _process_and_cache(
        self,
    ) -> Tuple[SentenceStore, LemmatizedStore, FrequencyCounter]:
        """Process corpus and save to cache."""
        logger.info(f"Processing corpus: {self.corpus_name}")

        # Get and log metadata
        metadata = self.reader.get_metadata()
        total_sentences = metadata.get("total_sentences", "unknown")
        max_sentences = self.corpus_config.get("max_sentences", None)

        processed_sentences = (
            min(total_sentences, max_sentences)
            if max_sentences is not None and isinstance(total_sentences, int)
            else total_sentences
        )
        logger.info(
            f"Processing {processed_sentences:,}/{total_sentences:,} sentences from {self.corpus_name}"
        )

        # Initialize stores and counter
        sent_path, lemma_path, freq_path = self._get_paths()
        sentence_store = SentenceStore(sent_path)
        lemmatized_store = LemmatizedStore(lemma_path)
        counter = FrequencyCounter()

        # Read sentences in batches
        sentence_batches: Generator[List[str], None, None] = self.reader.read_sentences(
            batch_size=self.processing_config.get("sentence_batch_size", 10000)
        )

        # Track sentence IDs
        sid = 0

        def on_sentence(original_tokens: List[str], lemma_tokens: List[LemmaToken]):
            nonlocal sid
            sent_writer.write(sid, original_tokens)
            lemma_writer.write(sid, lemma_tokens)
            sid += 1

        # Process with index-aligned storage
        with sentence_store.open_writer() as sent_writer, lemmatized_store.open_writer() as lemma_writer:
            lemma_pos_stream = self.lemmatizer.process_corpus_batches_aligned(
                sentence_batches,
                on_sentence=on_sentence,
            )
            counter.update(lemma_pos_stream)

        # Save frequency counter
        freq_path.parent.mkdir(parents=True, exist_ok=True)
        with open(freq_path, "wb") as f:
            pickle.dump(counter.to_dict(), f)

        stats = counter.get_statistics()
        cache_key = self._generate_cache_key()
        logger.info(
            f"Completed {self.corpus_name}: {stats['total_tokens']:,} tokens, "
            f"{stats['unique_lemmas']:,} unique lemmas (key: {cache_key})"
        )

        return sentence_store, lemmatized_store, counter

    def process(self) -> Tuple[SentenceStore, LemmatizedStore, FrequencyCounter]:
        """
        Process corpus with caching.

        Returns:
            Tuple of (SentenceStore, LemmatizedStore, FrequencyCounter).
            Note: FrequencyCounter is raw (unfiltered). Apply post-processing separately.
        """
        if self._all_cached():
            return self._load_from_cache()

        return self._process_and_cache()
