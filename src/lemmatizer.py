import spacy
from typing import List, Tuple, Generator, Optional, Callable
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from src.vocab import LemmaToken

# Type alias for token with offset: (lemma, pos, char_start, char_end) - deprecated
TokenWithOffset = Tuple[str, str, int, int]

# New type: aligned tokens (original_tokens, lemma_tokens)
AlignedTokens = Tuple[List[str], List[LemmaToken]]


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

        self.nlp = spacy.load(language_model, disable=["parser", "ner"])
        self.nlp.max_length = 2000000

    def _process_sentences_aligned(
        self, sentences: List[str]
    ) -> Generator[AlignedTokens, None, None]:
        """
        Process sentences and yield index-aligned token lists.

        For each sentence, yields a tuple of:
        - List of original tokens (same order/length as lemma_tokens)
        - List of lemma tokens [(lemma, pos), ...]

        Both lists have the same length and are index-aligned.
        """
        for doc in self.nlp.pipe(sentences, batch_size=self.batch_size):
            original_tokens: List[str] = []
            lemma_tokens: List[LemmaToken] = []

            for token in doc:
                if not token.is_space and not token.is_punct:
                    original_tokens.append(token.text)
                    lemma_tokens.append((token.lemma_.lower(), token.pos_))

            yield original_tokens, lemma_tokens

    def process_corpus_batches_aligned(
        self,
        sentence_batches: Generator[List[str], None, None],
        on_sentence: Optional[Callable[[List[str], List[LemmaToken]], None]] = None,
    ) -> Generator[LemmaToken, None, None]:
        """
        Process batches with aligned original and lemma tokens.

        Yields (lemma, pos) tuples for frequency counting while calling
        on_sentence callback with aligned token lists for storage.

        Args:
            sentence_batches: Generator yielding batches of sentences
            on_sentence: Callback receiving (original_tokens, lemma_tokens)

        Yields:
            LemmaToken tuples for frequency counting
        """
        total_sentences = 0
        batch_count = 0

        for batch in sentence_batches:
            batch_count += 1
            for original_tokens, lemma_tokens in self._process_sentences_aligned(batch):
                total_sentences += 1
                if on_sentence:
                    on_sentence(original_tokens, lemma_tokens)
                yield from lemma_tokens

        logger.info(
            f"Completed processing {batch_count} batches, {total_sentences:,} total sentences (aligned)"
        )
