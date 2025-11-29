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

        # POS tags to accept and count
        self.accepted_pos = {
            "NOUN",  # Nouns: casa, perro, ciudad
            "VERB",  # Verbs: comer, hablar, vivir
            "ADJ",  # Adjectives: bonito, grande, rápido
            "ADV",  # Adverbs (modify verbs, adjectives, or other adverbs): muy, bien, siempre, ahora
            # either standalone (e.g. rápido) or derived from adjectives (e.g. rápidamente)
            "ADP",  # Prepositions: de, en, para, con
            "CCONJ",  # Coordinating conjunctions: y, o, pero
            "SCONJ",  # Subordinating conjunctions: que, si, porque
            "AUX",  # Auxiliary verbs: ser, estar, haber
        }

        # POS tags to ignore (not count, no warning)
        self.ignored_pos = {
            "SYM",  # Symbols: %, $, @
            "X",  # Other/unclassified: foreign words, URLs
            "SPACE",  # Whitespace
            "DET",  # Determiners/articles: el, la, un, este
            "PROPN",  # Proper nouns: Madrid, García, España
            "NUM",  # Numbers: uno, dos, tres
            "INTJ",  # Interjections: oh, ah, hola
            "PRON",  # Pronouns: yo, tú, él, ella
            "PUNCT",  # Punctuation: . , ; :
            "PART",  # Particles: Seems to only contain 'no' in Spanish
        }

        self.nlp = spacy.load(language_model)
        self.nlp.max_length = 2000000

    def _clean_lemma(self, lemma: str) -> str:
        """
        Clean lemma by removing multi-word artifacts.

        Args:
            lemma: Raw lemma from spaCy

        Returns:
            Cleaned single-word lemma
        """
        # Only clean if it's a multi-word lemma
        if " " in lemma:
            # Remove enclitic pronouns that spaCy sometimes adds (e.g., "hacer él" -> "hacer")
            lemma = re.sub(
                r"\s+(yo|tú|él|ella|nosotros|vosotros|ellos|ellas|lo|la|los|las|le|les|me|te|se)$",
                "",
                lemma,
            )

        return lemma

    def _process_sentences(
        self, sentences: List[str]
    ) -> Generator[Tuple[str, str, bool], None, None]:
        """
        Process a batch of sentences and yield (lemma, pos, is_oov) tuples.
        Only yields tokens with accepted POS tags.

        Args:
            sentences: List of sentences to process

        Yields:
            Tuples of (lemma, pos_tag, is_oov) for each token with accepted POS
        """
        for doc in self.nlp.pipe(sentences, batch_size=self.batch_size):
            for token in doc:
                if not token.is_space and not token.is_punct:
                    pos = token.pos_

                    # Check if POS should be accepted
                    if pos in self.accepted_pos:
                        lemma = token.lemma_.lower()
                        lemma = self._clean_lemma(lemma)
                        is_oov = token.is_oov
                        yield (lemma, pos, is_oov)

                    elif pos in self.ignored_pos:
                        # Silently ignore
                        continue

                    else:
                        # Unknown POS tag - warn once per tag
                        logger.warning(
                            f"Unknown POS tag encountered: '{pos}' "
                            f"(token: '{token.text}', lemma: '{token.lemma_}'). "
                            f"This tag is neither accepted nor ignored."
                        )

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
