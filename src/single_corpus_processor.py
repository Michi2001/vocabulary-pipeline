import os
import hashlib
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Generator, List, Tuple

from src.corpus_readers.base_reader import BaseCorpusReader
from src.lemmatizer import Lemmatizer
from src.frequency_counter import FrequencyCounter
from src.post_processor import FrequencyPostProcessor
from src.sentence_token_store import SentenceTokenStore

logger = logging.getLogger(__name__)


class SingleCorpusProcessor:
    """
    Processes a single corpus and returns frequency data.

    Includes integrated caching functionality using Pickle.
    """

    def __init__(
        self,
        corpus_config: Dict[str, Any],
        reader: BaseCorpusReader,
        lemmatizer: Lemmatizer,
        processing_config: Dict[str, Any],
        cache_config: Dict[str, Any] = None,
        filter_config: Dict[str, Any] = None,
        sentence_store_config: Dict[str, Any] = None,
    ):
        """
        Initialize the single corpus processor.

        Args:
            corpus_config: Configuration for this corpus (file_path, type, encoding)
            reader: CorpusReader instance to use
            lemmatizer: Lemmatizer instance to use
            processing_config: Processing configuration (batch sizes, etc.)
            cache_config: Optional cache configuration (enabled, directory)
            filter_config: Optional filter configuration (accepted_pos, pos_mappings, etc.)
            sentence_store_config: Optional config for sentence-token storage (enabled)
        """
        self.corpus_config = corpus_config
        self.reader = reader
        self.lemmatizer = lemmatizer
        self.processing_config = processing_config
        self.cache_config = cache_config or {"enabled": False}
        self.filter_config = filter_config or {}
        self.sentence_store_config = sentence_store_config or {"enabled": False}

        self.corpus_name = corpus_config.get(
            "name", Path(corpus_config["file_path"]).stem
        )

        self._sentence_store: SentenceTokenStore = None

    def _generate_cache_key(self) -> str:
        """
        Generate cache key from configuration.

        Includes: corpus filename, file modification time, spaCy model, max_sentences.

        Does NOT include filtering configuration (accepted_pos, pos_mappings, etc.)
        as those are applied during post-processing.

        Batch sizes are NOT included as they only affect performance, not results.

        Returns:
            SHA256 hash string
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
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """
        Get path for cache file.

        Args:
            cache_key: Cache key hash

        Returns:
            Path to cache file
        """
        cache_base_dir = Path(self.cache_config.get("directory", "cache"))
        cache_corpus_dir = cache_base_dir / self.corpus_name
        cache_corpus_dir.mkdir(parents=True, exist_ok=True)
        return cache_corpus_dir / f"{cache_key}.pkl"

    def _get_sentence_store_path(self, cache_key: str) -> Path:
        """Get path for sentence-token store file."""
        cache_base_dir = Path(self.cache_config.get("directory", "cache"))
        cache_corpus_dir = cache_base_dir / self.corpus_name
        cache_corpus_dir.mkdir(parents=True, exist_ok=True)
        return cache_corpus_dir / f"{cache_key}_sentences.jsonl.gz"

    def get_sentence_store(self) -> SentenceTokenStore:
        """Get the sentence store (available after processing with sentence_store enabled)."""
        if self._sentence_store is None:
            if self.sentence_store_config.get("enabled", False):
                cache_key = self._generate_cache_key()
                store_path = self._get_sentence_store_path(cache_key)
                if store_path.exists():
                    self._sentence_store = SentenceTokenStore(store_path)
        return self._sentence_store

    def _load_from_cache(self, cache_key: str) -> FrequencyCounter:
        """
        Load frequency counter from cache.

        Args:
            cache_key: Cache key hash

        Returns:
            FrequencyCounter if cache exists and is valid, None otherwise
        """
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)

            logger.info(
                f"Loaded from cache: {self.corpus_name} (key: {cache_key[:8]}...)"
            )
            return FrequencyCounter.from_dict(data)

        except Exception as e:
            logger.warning(f"Failed to load cache for {self.corpus_name}: {e}")
            return None

    def _save_to_cache(self, cache_key: str, counter: FrequencyCounter) -> None:
        """
        Save frequency counter to cache.

        Args:
            cache_key: Cache key hash
            counter: FrequencyCounter to save
        """
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(counter.to_dict(), f)

            logger.info(f"Saved to cache: {self.corpus_name} (key: {cache_key[:8]}...)")

        except Exception as e:
            logger.warning(f"Failed to save cache for {self.corpus_name}: {e}")

    def _process_corpus(self, cache_key: str = None) -> FrequencyCounter:
        """
        Process corpus without cache.

        Args:
            cache_key: Cache key for sentence store path (optional)

        Returns:
            FrequencyCounter with frequency data
        """
        logger.info(f"Processing corpus: {self.corpus_name}")

        # Get and log metadata
        metadata = self.reader.get_metadata()
        total_sentences = metadata.get("total_sentences", "unknown")
        max_sentences = self.corpus_config.get("max_sentences", None)

        processed_sentences = (
            min(total_sentences, max_sentences)
            if max_sentences is not None
            else total_sentences
        )
        logger.info(
            f"Processing {processed_sentences:,}/{total_sentences:,} sentences from {self.corpus_name}"
        )

        # Initialize counter
        counter = FrequencyCounter()

        # Read sentences in batches
        sentence_batches: Generator[List[str], None, None] = self.reader.read_sentences(
            batch_size=self.processing_config.get("sentence_batch_size", 10000)
        )

        # Check if we should store sentence-token mappings
        store_sentences = self.sentence_store_config.get("enabled", False)

        if store_sentences and cache_key:
            # Process with sentence-token storage
            store_path = self._get_sentence_store_path(cache_key)
            self._sentence_store = SentenceTokenStore(store_path)
            self._sentence_store.open_write()

            try:
                lemma_pos_stream = self.lemmatizer.process_corpus_batches_with_offsets(
                    sentence_batches,
                    on_sentence=lambda tokens: self._sentence_store.write(tokens),
                )
                counter.update(lemma_pos_stream)
            finally:
                self._sentence_store.close()

            logger.info(f"Sentence-token store saved: {store_path}")
        else:
            # Process without sentence storage (original behavior)
            lemma_pos_stream: Generator[Tuple[str, str], None, None] = (
                self.lemmatizer.process_corpus_batches(sentence_batches)
            )
            counter.update(lemma_pos_stream)

        stats = counter.get_statistics()
        logger.info(
            f"Completed {self.corpus_name}: {stats['total_tokens']:,} tokens, "
            f"{stats['unique_lemmas']:,} unique lemmas"
        )

        return counter

    def process(self) -> FrequencyCounter:
        """
        Process corpus with optional caching and post-processing.

        Returns:
            FrequencyCounter with filtered/transformed frequency data
        """
        cache_key = self._generate_cache_key()

        # Check cache if enabled
        if self.cache_config.get("enabled", False):
            cached_counter = self._load_from_cache(cache_key)

            if cached_counter is not None:
                raw_counter = cached_counter
                # Check if sentence store exists when cache hit
                if self.sentence_store_config.get("enabled", False):
                    store_path = self._get_sentence_store_path(cache_key)
                    if store_path.exists():
                        self._sentence_store = SentenceTokenStore(store_path)
                        logger.info(f"Loaded sentence store from cache: {store_path}")
                    else:
                        # Cache hit but no sentence store - need to reprocess
                        logger.info(
                            "Sentence store missing, reprocessing corpus..."
                        )
                        raw_counter = self._process_corpus(cache_key)
                        self._save_to_cache(cache_key, raw_counter)
            else:
                # Process corpus and cache raw results
                raw_counter = self._process_corpus(cache_key)
                self._save_to_cache(cache_key, raw_counter)
        else:
            # Process corpus without cache
            raw_counter = self._process_corpus(cache_key)

        # Apply post-processing filters
        if self.filter_config:
            logger.info(f"Applying post-processing filters to {self.corpus_name}...")
            processor = FrequencyPostProcessor(self.filter_config)
            filtered_counter = processor.process(raw_counter)
            return filtered_counter
        else:
            # No filtering - return raw data
            logger.info(
                f"No filtering configured, returning raw data for {self.corpus_name}"
            )
            return raw_counter
