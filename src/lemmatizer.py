import spacy
from typing import List, Tuple, Generator
import logging
import re

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Lemmatizer:
    """
    Lemmatizer using spaCy for contextual lemmatization.

    Yields all tokens with original POS tags and lemmas without filtering.
    Processes sentences in batches for efficient memory usage and performance.
    """

    def __init__(self, language_model: str = "es_core_news_sm", batch_size: int = 1000):
        """
        Initialize the lemmatizer.

        Args:
            language_model: spaCy model name (default: Spanish small model)
            batch_size: Number of sentences to process in one spaCy batch
        """
        self.language_model = language_model
        self.batch_size = batch_size

        self.nlp = spacy.load(language_model)
        self.nlp.max_length = 2000000

    def _process_sentences(
        self, sentences: List[str]
    ) -> Generator[Tuple[str, str, bool], None, None]:
        """
        Process a batch of sentences and yield (lemma, pos, is_oov) tuples.
        Yields ALL tokens with original POS tags and original lemmas.

        Args:
            sentences: List of sentences to process

        Yields:
            Tuples of (lemma, pos_tag, is_oov) for each token
        """
        for doc in self.nlp.pipe(sentences, batch_size=self.batch_size):
            for token in doc:
                # Skip whitespace and punctuation tokens only
                if not token.is_space and not token.is_punct:
                    lemma = token.lemma_.lower()  # Keep original lemma
                    pos = token.pos_                # Keep original POS tag
                    is_oov = token.is_oov
                    yield (lemma, pos, is_oov)

    def process_corpus_batches(
        self, sentence_batches: Generator[List[str], None, None]
    ) -> Generator[Tuple[str, str, bool], None, None]:
        """
        Process batches of sentences from a corpus reader.

        Args:
            sentence_batches: Generator yielding batches of sentences

        Yields:
            Tuples of (lemma, pos_tag, is_oov) for each token in the corpus
        """
        total_sentences = 0
        batch_count = 0

        for batch in sentence_batches:
            batch_count += 1
            total_sentences += len(batch)
            yield from self._process_sentences(batch)

        logger.info(
            f"Completed processing {batch_count} batches, {total_sentences:,} total sentences"
        )
