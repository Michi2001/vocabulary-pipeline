import spacy
from typing import List, Tuple, Generator, Optional, Callable
import logging
import re

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Type alias for token with offset: (lemma, pos, char_start, char_end)
TokenWithOffset = Tuple[str, str, int, int]


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
    ) -> Generator[Tuple[str, str], None, None]:
        """
        Process a batch of sentences and yield (lemma, pos) tuples.
        Yields ALL tokens with original POS tags and original lemmas.

        Args:
            sentences: List of sentences to process

        Yields:
            Tuples of (lemma, pos_tag) for each token
        """
        for doc in self.nlp.pipe(sentences, batch_size=self.batch_size):
            for token in doc:
                # Skip whitespace and punctuation tokens only
                if not token.is_space and not token.is_punct:
                    lemma = token.lemma_.lower()  # Keep original lemma
                    pos = token.pos_  # Keep original POS tag
                    yield (lemma, pos)

    def process_corpus_batches(
        self, sentence_batches: Generator[List[str], None, None]
    ) -> Generator[Tuple[str, str], None, None]:
        """
        Process batches of sentences from a corpus reader.

        Args:
            sentence_batches: Generator yielding batches of sentences

        Yields:
            Tuples of (lemma, pos_tag) for each token in the corpus
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

    def _process_sentences_with_offsets(
        self, sentences: List[str]
    ) -> Generator[
        Tuple[List[TokenWithOffset], Generator[Tuple[str, str], None, None]], None, None
    ]:
        """
        Process sentences and yield tokens with character offsets.

        For each sentence, yields a tuple of:
        - List of all tokens with offsets for storage
        - Generator of (lemma, pos) for frequency counting

        Args:
            sentences: List of sentences to process

        Yields:
            Tuple of (tokens_with_offsets, lemma_pos_generator)
                tokens_with_offset: e.g.  [("el", "DET", 0, 2), ("gato", "NOUN", 3, 7), ("comer", "VERB", 8, 12)]
        """
        for doc in self.nlp.pipe(sentences, batch_size=self.batch_size):
            tokens_with_offsets: List[TokenWithOffset] = []
            lemma_pos_pairs: List[Tuple[str, str]] = []

            for token in doc:
                if not token.is_space and not token.is_punct:
                    lemma = token.lemma_.lower()
                    pos = token.pos_
                    tokens_with_offsets.append(
                        (lemma, pos, token.idx, token.idx + len(token.text))
                    )
                    lemma_pos_pairs.append((lemma, pos))

            yield tokens_with_offsets, (pair for pair in lemma_pos_pairs)

    def process_corpus_batches_with_offsets(
        self,
        sentence_batches: Generator[List[str], None, None],
        on_sentence: Optional[Callable[[List[TokenWithOffset]], None]] = None,
    ) -> Generator[Tuple[str, str], None, None]:
        """
        Process batches with token offsets, calling callback for each sentence.

        This enables parallel frequency counting and sentence-token storage.

        Args:
            sentence_batches: Generator yielding batches of sentences
            on_sentence: Callback receiving tokens with offsets for each sentence

        Yields:
            Tuples of (lemma, pos_tag) for frequency counting
        """
        total_sentences = 0
        batch_count = 0

        for batch in sentence_batches:
            batch_count += 1
            for (
                tokens_with_offsets,
                lemma_pos_gen,
            ) in self._process_sentences_with_offsets(batch):
                total_sentences += 1
                if on_sentence:
                    on_sentence(tokens_with_offsets)
                yield from lemma_pos_gen

        logger.info(
            f"Completed processing {batch_count} batches, {total_sentences:,} total sentences (with offsets)"
        )
