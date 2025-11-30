import os
import random
from typing import Generator, Dict, Any, Optional
from .base_reader import BaseCorpusReader


class OpusCorpusReader(BaseCorpusReader):
    """
    Reader for OPUS corpus format.

    Expected format: .txt file with one sentence per line
    """

    def __init__(
        self,
        file_path: str,
        encoding: str = "utf-8",
        max_sentences: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize OPUS corpus reader.

        Args:
            file_path: Path to .txt file
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
        Read sentences from OPUS corpus in batches.

        Yields batches of sentences for memory-efficient processing.
        If max_sentences is set, a random selection of sentences will be returned.

        Args:
            batch_size: Number of sentences per batch

        Yields:
            List of sentences (strings) for each batch
        """
        all_sentences = []
        with open(self.file_path, "r", encoding=self.encoding) as f:
            for line in f:
                sentence = line.strip()
                if not sentence:
                    continue

                all_sentences.append(sentence)

                if self.max_sentences is None and len(all_sentences) >= batch_size:
                    yield all_sentences
                    all_sentences = []

        if self.max_sentences is not None and len(all_sentences) > self.max_sentences:
            all_sentences = random.sample(all_sentences, self.max_sentences)

        for i in range(0, len(all_sentences), batch_size):
            yield all_sentences[i : i + batch_size]

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the OPUS corpus.

        Returns:
            Dictionary with file size, total sentence count, and file path
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
            "format": "OPUS",
            "encoding": self.encoding,
        }
