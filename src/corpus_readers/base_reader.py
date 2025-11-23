from abc import ABC, abstractmethod
from typing import Generator, Dict, Any


class BaseCorpusReader(ABC):
    """
    Abstract base class for corpus readers.

    This allows for easy extension to support different corpus formats
    (e.g., Leipzig, XML, JSON, plain text) without changing the processing pipeline.
    """

    def __init__(self, file_path: str, **kwargs):
        """
        Initialize the corpus reader.

        Args:
            file_path: Path to the corpus file
            **kwargs: Additional format-specific parameters
        """
        self.file_path = file_path
        self.kwargs = kwargs

    @abstractmethod
    def read_sentences(self, batch_size: int = 10000) -> Generator[list[str], None, None]:
        """
        Read sentences from the corpus in batches.

        This method should yield batches of sentences as lists of strings.
        Using a generator allows for memory-efficient processing of large corpora.

        Args:
            batch_size: Number of sentences to yield per batch

        Yields:
            List of sentences (strings) for each batch
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the corpus.

        Returns:
            Dictionary with metadata (e.g., total sentences, file size, language)
        """
        pass
