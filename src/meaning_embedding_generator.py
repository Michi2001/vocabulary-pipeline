"""
Meaning Embedding Generator for Word Sense Disambiguation.

Pre-computes BERT embeddings for target words in example sentences.
Uses metadata (_target_start, _target_end) extracted by LemmaTranslator.
These embeddings are used later by the BatchDisambiguator for WSD.
"""

import logging
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from src.vocab import Translation

logger = logging.getLogger(__name__)


class EmbeddingTask(NamedTuple):
    """Tracks which meaning an example sentence belongs to."""

    translation_idx: int
    meaning_idx: int
    sentence: str
    char_start: int
    char_end: int


def _get_torch_device() -> torch.device:
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MeaningEmbeddingGenerator:
    """
    Generates BERT embeddings for target words in example sentences.

    Uses the same BERT model and embedding extraction strategy as the disambiguator
    to ensure consistency between pre-computed meaning embeddings and runtime embeddings.
    """

    def __init__(
        self,
        model_name: str = "dccuchile/bert-base-spanish-wwm-cased",
        bert_batch_tokens: int = 4096,
    ):
        self.model_name = model_name
        self.bert_batch_tokens = bert_batch_tokens
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load_model(self):
        """Lazy load BERT model and tokenizer."""
        if self._model is None:
            self._device = _get_torch_device()
            logger.info(
                f"Loading BERT model: {self.model_name} (device: {self._device})"
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self._device)
            self._model.eval()

    def _collect_tasks(
        self, translations: List[Translation]
    ) -> Tuple[List[EmbeddingTask], int, int]:
        """Collect all valid example sentences as embedding tasks."""
        tasks = []
        total_meanings = 0
        missing_metadata = 0

        for t_idx, translation in enumerate(translations):
            if not translation.meanings:
                continue

            for m_idx, meaning in enumerate(translation.meanings):
                total_meanings += 1

                if not meaning.examples_es:
                    continue

                if meaning._target_starts is None or meaning._target_ends is None:
                    missing_metadata += 1
                    continue

                for e_idx, example in enumerate(meaning.examples_es):
                    if e_idx >= len(meaning._target_starts) or e_idx >= len(
                        meaning._target_ends
                    ):
                        break

                    char_start = meaning._target_starts[e_idx]
                    char_end = meaning._target_ends[e_idx]

                    if char_start is None or char_end is None:
                        continue

                    tasks.append(
                        EmbeddingTask(
                            translation_idx=t_idx,
                            meaning_idx=m_idx,
                            sentence=example,
                            char_start=char_start,
                            char_end=char_end,
                        )
                    )

        return tasks, total_meanings, missing_metadata

    def _process_batch(self, batch: List[EmbeddingTask]) -> Dict[int, List[float]]:
        """Run BERT on a batch and return {batch_index: embedding}."""
        texts = [t.sentence for t in batch]
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        # Resolve char_to_token BEFORE moving tensors to device
        token_positions: Dict[int, List[int]] = {}
        for idx, task in enumerate(batch):
            positions = []
            for char_pos in range(task.char_start, task.char_end):
                bert_idx = inputs.char_to_token(idx, char_pos)
                if bert_idx is not None and bert_idx not in positions:
                    positions.append(bert_idx)
            if positions:
                token_positions[idx] = positions

        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-2]  # (batch, seq, hidden)

        result: Dict[int, List[float]] = {}
        for idx, positions in token_positions.items():
            emb = hidden_states[idx, positions].mean(dim=0)
            result[idx] = emb.cpu().tolist()

        return result

    def _flush_batch(
        self,
        batch: List[EmbeddingTask],
        embeddings: Dict[Tuple[int, int], List[List[float]]],
    ) -> None:
        """Process a batch and accumulate results into embeddings dict."""
        results = self._process_batch(batch)
        for idx, emb in results.items():
            task = batch[idx]
            key = (task.translation_idx, task.meaning_idx)
            if key not in embeddings:
                embeddings[key] = []
            embeddings[key].append(emb)

    def generate(self, translations: List[Translation]) -> List[Translation]:
        """Add averaged BERT embedding from multiple examples per meaning."""
        logger.info(f"Generating meaning embeddings for {len(translations)} lemmas...")
        self._load_model()

        # Phase 1: Collect all tasks
        tasks, total_meanings, missing_metadata = self._collect_tasks(translations)
        logger.info(f"Collected {len(tasks)} examples from {total_meanings} meanings")

        # Phase 2: Process in token-budget batches
        embeddings: Dict[Tuple[int, int], List[List[float]]] = (
            {}
        )  # (translation_idx, meaning_idx) -> list of embeddings
        batch: List[EmbeddingTask] = []
        batch_tokens = 0
        total_processed = 0

        for task in tasks:
            batch.append(task)
            batch_tokens += len(task.sentence.split())

            if batch_tokens >= self.bert_batch_tokens:
                self._flush_batch(batch, embeddings)
                total_processed += len(batch)
                batch = []
                batch_tokens = 0

        if batch:
            self._flush_batch(batch, embeddings)
            total_processed += len(batch)

        # Phase 3: Average and assign
        successful = 0
        for (t_idx, m_idx), emb_list in embeddings.items():
            averaged = np.mean(emb_list, axis=0).tolist()
            translations[t_idx].meanings[m_idx]._bert_embedding = averaged
            successful += 1

        logger.info(
            f"Meaning embeddings generated: {successful}/{total_meanings} "
            f"(from {total_processed} examples, missing metadata: {missing_metadata})"
        )

        return translations
