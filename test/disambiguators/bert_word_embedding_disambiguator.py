"""
BERT-based word embedding disambiguator.

Uses contextualized BERT embeddings (second-to-last layer) of target words
to compare meanings via cosine similarity.
"""

from abc import abstractmethod
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModel

from .base_disambiguator import BaseDisambiguator, DisambiguationResult


class BertWordEmbeddingDisambiguator(BaseDisambiguator):
    """
    Base class for BERT word-embedding disambiguation approaches.

    Uses Template Method pattern: subclasses choose whether to use
    definition or example sentences via _get_meaning_sentence().
    """

    def __init__(self, model_name: str = "bert-base-multilingual-cased"):
        """
        Initialize with specified BERT model.

        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load BERT model and tokenizer."""
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()

    def _get_contextual_embedding(self, sentence_with_brackets: str) -> torch.Tensor:
        """
        Get BERT contextual embedding for bracketed word (all layers averaged).

        Args:
            sentence_with_brackets: Sentence with [target_word] in brackets

        Returns:
            Embedding tensor
        """
        self._load_model()

        # Extract bracketed word and its position
        target_word, clean_sentence, char_start, char_end = (
            self._extract_bracketed_word(sentence_with_brackets)
        )

        # Tokenize clean sentence
        inputs = self._tokenizer(clean_sentence, return_tensors="pt")

        # Find token positions for target word using character-to-token mapping
        target_positions = []
        for char_pos in range(char_start, char_end):
            token_idx = inputs.char_to_token(0, char_pos)
            if token_idx is not None and token_idx not in target_positions:
                target_positions.append(token_idx)

        if not target_positions:
            raise ValueError(
                f"Target word '{target_word}' at position {char_start}-{char_end} "
                f"not found in tokenized sentence"
            )

        # Get BERT embeddings from second-to-last hidden layer
        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
            hidden_states = (
                outputs.hidden_states
            )  # tuple of (num_layers, batch_size, seq_len, hidden_dim)

            # Use second-to-last layer (index -2)
            embeddings = hidden_states[-2][0]  # [seq_len, hidden_dim]

        # Average embeddings of target word tokens
        word_embedding = embeddings[target_positions].mean(dim=0)

        return word_embedding

    def _cosine_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """Calculate cosine similarity between two vectors."""
        return torch.nn.functional.cosine_similarity(
            vec1.unsqueeze(0), vec2.unsqueeze(0)
        ).item()

    @abstractmethod
    def _get_meaning_sentence(self, meaning: Dict[str, str]) -> str:
        """
        Get sentence to use for this meaning.

        Subclasses decide: definition or example.

        Args:
            meaning: Dict with 'definition' and 'example' keys

        Returns:
            Sentence with bracketed word
        """
        pass

    def disambiguate(
        self, target_sentence: str, meanings: List[Dict[str, str]]
    ) -> DisambiguationResult:
        """
        Disambiguate using BERT word embeddings.

        Args:
            target_sentence: Sentence with [target_word] in brackets
            meanings: List of dicts with 'definition' and 'example' keys

        Returns:
            DisambiguationResult with predicted index and confidence
        """
        # Extract target word and get its embedding
        target_word, _, _, _ = self._extract_bracketed_word(target_sentence)
        target_emb = self._get_contextual_embedding(target_sentence)

        # Compare with each meaning
        scores = []
        for meaning in meanings:
            meaning_sentence = self._get_meaning_sentence(meaning)
            meaning_emb = self._get_contextual_embedding(meaning_sentence)
            similarity = self._cosine_similarity(target_emb, meaning_emb)
            scores.append(similarity)

        # Return result
        predicted_idx = scores.index(max(scores))
        return DisambiguationResult(
            predicted_index=predicted_idx,
            confidence=scores[predicted_idx],
            all_scores=scores,
            target_word=target_word,
            method_name=f"BERT-Word-{self.__class__.__name__.replace('Bert', '').replace('Disambiguator', '')}",
        )
