"""
Base classes for word sense disambiguation.

Provides minimal interface to support completely different disambiguation approaches:
- BERT-based (word embeddings in context)
- Sentence embeddings (compare full definitions)
- WordNet synset-based
- Classical TF-IDF
- Hybrid methods
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Tuple
import re


@dataclass
class DisambiguationResult:
    """Result of word sense disambiguation."""

    predicted_index: int
    confidence: float
    all_scores: List[float]
    target_word: str
    method_name: str


class BaseDisambiguator(ABC):
    """Abstract base class for word sense disambiguation."""

    @abstractmethod
    def disambiguate(
        self, target_sentence: str, meanings: List[Dict[str, str]]
    ) -> DisambiguationResult:
        """
        Disambiguate target word using provided meanings.

        Args:
            target_sentence: Sentence with [target_word] in brackets
            meanings: List of dicts with 'definition' and 'example' keys

        Returns:
            DisambiguationResult with predicted index and confidence
        """
        pass

    def _extract_bracketed_word(self, sentence: str) -> Tuple[str, str, int, int]:
        """
        Extract word from brackets and return cleaned sentence with positions.

        Returns:
            (target_word, sentence_without_brackets, char_start, char_end)
            char_start/char_end are positions in the cleaned sentence
        """
        match = re.search(r"\[([^\]]+)\]", sentence)
        if not match:
            raise ValueError(f"No bracketed word found in: {sentence}")

        target_word = match.group(1)
        bracket_start = match.start()  # Position of '['

        # Position in cleaned sentence (without brackets)
        char_start = bracket_start
        char_end = char_start + len(target_word)

        cleaned_sentence = sentence.replace(f"[{target_word}]", target_word)
        return target_word, cleaned_sentence, char_start, char_end
