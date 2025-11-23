import os
import hashlib
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Generator, List, Tuple

from src.corpus_readers.base_reader import BaseCorpusReader
from src.lemmatizer import Lemmatizer
from src.frequency_counter import FrequencyCounter

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
    ):
        """
        Initialize the single corpus processor.

        Args:
            corpus_config: Configuration for this corpus (file_path, type, encoding)
            reader: CorpusReader instance to use
            lemmatizer: Lemmatizer instance to use
            processing_config: Processing configuration (batch sizes, etc.)
            cache_config: Optional cache configuration (enabled, directory)
        """
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
        Generate cache key from configuration.

        Includes: corpus filename (not full path), file modification time, spaCy model,
        and MAIN_POS_CATEGORIES.

        Batch sizes are NOT included as they only affect performance, not results.

        Returns:
            SHA256 hash string
        """
        corpus_path = self.corpus_config["file_path"]
        corpus_filename = Path(corpus_path).name
        file_mtime = os.path.getmtime(corpus_path) if os.path.exists(corpus_path) else 0

        key_components = [
            corpus_filename,
            str(file_mtime),
            self.lemmatizer.language_model,
            str(sorted(FrequencyCounter.MAIN_POS_CATEGORIES)),
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

    def _process_corpus(self) -> FrequencyCounter:
        """
        Process corpus without cache.

        Returns:
            FrequencyCounter with frequency data
        """
        logger.info(f"Processing corpus: {self.corpus_name}")

        # Initialize counter
        counter = FrequencyCounter()

        # Read sentences in batches
        sentence_batches: Generator[List[str], None, None] = self.reader.read_sentences(
            batch_size=self.processing_config.get("sentence_batch_size", 10000)
        )

        # Process with lemmatizer
        lemma_pos_stream: Generator[Tuple[str, str, bool], None, None] = (
            self.lemmatizer.process_corpus_batches(sentence_batches)
        )

        # Update counter
        counter.update(lemma_pos_stream)

        stats = counter.get_statistics()
        logger.info(
            f"Completed {self.corpus_name}: {stats['total_tokens']:,} tokens, "
            f"{stats['unique_lemmas']:,} unique lemmas"
        )

        return counter

    def process(self) -> FrequencyCounter:
        """
        Process corpus with optional caching.

        Returns:
            FrequencyCounter with frequency data
        """
        # Check cache if enabled
        if self.cache_config.get("enabled", False):
            cache_key = self._generate_cache_key()
            cached_counter = self._load_from_cache(cache_key)

            if cached_counter is not None:
                return cached_counter

        # Process corpus
        counter = self._process_corpus()

        # Save to cache if enabled
        if self.cache_config.get("enabled", False):
            cache_key = self._generate_cache_key()
            self._save_to_cache(cache_key, counter)

        return counter
