import os
import random
from typing import Generator, Dict, Any, Optional
from .base_reader import BaseCorpusReader


class LeipzigCorpusReader(BaseCorpusReader):
    """
    Reader for Leipzig Corpora Collection format.

    Expected format: *_sentences.txt
    Tab-separated: Sentence_ID <TAB> Sentence
    """

    def __init__(
        self,
        file_path: str,
        encoding: str = "utf-8",
        max_sentences: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize Leipzig corpus reader.

        Args:
            file_path: Path to *_sentences.txt file
            encoding: File encoding (default: utf-8)
            max_sentences: Maximum number of sentences to process (None = all sentences)
            **kwargs: Additional parameters
        """
        super().__init__(file_path, max_sentences=max_sentences, **kwargs)
        self.encoding = encoding

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Corpus file not found: {file_path}")

    def read_sentences(
        self, batch_size: int = 10000
    ) -> Generator[list[str], None, None]:
        """
        Read sentences from Leipzig corpus in batches.

        Yields batches of sentences (without IDs) for memory-efficient processing.
        If max_sentences is set, a random selection of sentences will be returned using
        reservoir sampling (avoids loading entire file into memory).

        Args:
            batch_size: Number of sentences per batch

        Yields:
            List of sentences (strings) for each batch
        """
        if self.max_sentences is None:
            # Stream sentences without keeping all in memory
            batch = []
            with open(self.file_path, "r", encoding=self.encoding) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split("\t", 1)
                    if len(parts) == 2:
                        _, sentence = parts
                        batch.append(sentence)

                        if len(batch) >= batch_size:
                            yield batch
                            batch = []

            if batch:
                yield batch
        else:
            # Use reservoir sampling to avoid loading entire file into memory
            reservoir_size = self.max_sentences
            reservoir = []
            n = 0

            with open(self.file_path, "r", encoding=self.encoding) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split("\t", 1)
                    if len(parts) == 2:
                        _, sentence = parts
                        n += 1

                        if len(reservoir) < reservoir_size:
                            reservoir.append(sentence)
                        else:
                            # Reservoir sampling: randomly replace element
                            j = random.randint(0, n - 1)
                            if j < reservoir_size:
                                reservoir[j] = sentence

            # Yield reservoir in batches
            for i in range(0, len(reservoir), batch_size):
                yield reservoir[i:i + batch_size]

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the Leipzig corpus.

        Returns:
            Dictionary with file size, estimated sentence count, and file path
        """
        file_size = os.path.getsize(self.file_path)

        sentence_count = 0
        with open(self.file_path, "r", encoding=self.encoding) as f:
            for line in f:
                if line.strip():
                    sentence_count += 1

        return {
            "file_path": self.file_path,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "total_sentences": sentence_count,
            "format": "Leipzig Corpora Collection",
            "encoding": self.encoding,
        }
