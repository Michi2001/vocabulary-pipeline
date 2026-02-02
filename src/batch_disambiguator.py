"""
Batch Word Sense Disambiguator.

Disambiguates polysemous words using BERT embeddings and pre-computed meaning embeddings.
Uses two-pass processing with token-based batching for efficiency.

Supports selective disambiguation: only specified POS types are disambiguated,
other POS types get their single VocabEntry ID directly without BERT processing.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.stores.disambiguated_store import DisambiguatedStore
from src.stores.lemmatized_store import LemmatizedStore
from src.stores.sentence_store import SentenceStore
from src.vocab import LemmaToken, Vocabulary

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class PolyTokenInfo:
    """Polysemous token identified during sentence classification."""

    token_idx: int
    char_start: int
    char_end: int
    lemma_token: LemmaToken


@dataclass
class BertTask:
    """A sentence queued for BERT processing."""

    sid: int
    text: str
    token_count: int
    entries: List[Optional[int]]  # Pre-filled mono-semantic/non-disambiguated results
    poly_infos: List[PolyTokenInfo] = field(default_factory=list)


# =============================================================================
# Ordered Write Buffer
# =============================================================================


class OrderedWriteBuffer:
    """Maintains sentence order when writing results."""

    def __init__(self, writer):
        self._writer = writer
        self._buffer: Dict[int, List[Optional[int]]] = {}
        self._next_sid = 0
        self.total_disambiguated = 0

    def add(self, sid: int, entries: List[Optional[int]]) -> None:
        """Add result to buffer."""
        self._buffer[sid] = entries
        self._flush()

    def _flush(self) -> None:
        """Write all consecutive sentences from buffer."""
        while self._next_sid in self._buffer:
            entries = self._buffer.pop(self._next_sid)
            self._writer.write(self._next_sid, entries)
            self.total_disambiguated += sum(1 for e in entries if e is not None)
            self._next_sid += 1


# =============================================================================
# BERT Batch Processor
# =============================================================================


class BertBatchProcessor:
    """Handles BERT encoding and embedding extraction."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        logger.info(f"Loading BERT model: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        if torch.cuda.is_available():
            self._model = self._model.cuda()
            logger.info("Using CUDA for disambiguation")

    def extract_embeddings(
        self, tasks: List[BertTask]
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Extract embeddings for all polysemous tokens in tasks.

        Returns:
            Dict mapping (task_idx, token_idx) -> embedding
        """
        if not tasks:
            return {}

        self._load_model()

        texts = [t.text for t in tasks]
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-2]  # (batch, seq, hidden)

        result: Dict[Tuple[int, int], np.ndarray] = {}

        for task_idx, task in enumerate(tasks):
            for pi in task.poly_infos:
                # Map character positions to BERT subword positions
                bert_positions = []
                for char_pos in range(pi.char_start, pi.char_end):
                    bert_idx = inputs.char_to_token(task_idx, char_pos)
                    if bert_idx is not None and bert_idx not in bert_positions:
                        bert_positions.append(bert_idx)

                if bert_positions:
                    emb = hidden_states[task_idx, bert_positions].mean(dim=0)
                    result[(task_idx, pi.token_idx)] = emb.cpu().numpy()

        return result


# =============================================================================
# Main Disambiguator
# =============================================================================


class BatchDisambiguator:
    """Disambiguates polysemous words using BERT embeddings."""

    def __init__(
        self,
        vocabulary: Vocabulary,
        model_name: str = "dccuchile/bert-base-spanish-wwm-cased",
        bert_batch_tokens: int = 4096,
    ):
        self.vocabulary = vocabulary
        self.bert_batch_tokens = bert_batch_tokens
        self._bert = BertBatchProcessor(model_name)

        # For disambiguated POS: mono/poly lookup
        self._mono_entries: Dict[LemmaToken, int] = {}
        self._poly_entries: Dict[LemmaToken, List[Tuple[int, np.ndarray]]] = {}

        # For non-disambiguated POS: direct lookup
        self._non_disambiguated_entries: Dict[LemmaToken, int] = {}

        # All vocab tokens
        self._vocab_tokens: Set[LemmaToken] = set()

        self._build_lookups()

    def _build_lookups(self) -> None:
        """Build lookup tables from vocabulary."""
        disambiguated_pos = self.vocabulary.disambiguated_pos

        for lemma_token, entry_ids in self.vocabulary.entries_by_lemma_token.items():
            self._vocab_tokens.add(lemma_token)
            pos = lemma_token[1]

            if pos not in disambiguated_pos:
                # Non-disambiguated POS: single entry, no BERT needed
                # Should only have one entry per lemma_token
                self._non_disambiguated_entries[lemma_token] = entry_ids[0]
            else:
                # Disambiguated POS: check for mono/poly
                entries_with_emb = []
                first_id = None

                for eid in entry_ids:
                    if first_id is None:
                        first_id = eid
                    entry = self.vocabulary.vocab_entries[eid]
                    if entry.bert_embedding is not None:
                        entries_with_emb.append((eid, entry.bert_embedding))

                if len(entries_with_emb) == 1:
                    self._mono_entries[lemma_token] = entries_with_emb[0][0]
                elif len(entries_with_emb) > 1:
                    self._poly_entries[lemma_token] = entries_with_emb
                elif first_id is not None:
                    # No embeddings, use first entry
                    self._mono_entries[lemma_token] = first_id

        logger.info(
            f"Lookups: {len(self._poly_entries)} polysemous, "
            f"{len(self._mono_entries)} mono-semantic, "
            f"{len(self._non_disambiguated_entries)} non-disambiguated"
        )

    def _classify_token(self, lemma_token: LemmaToken) -> Tuple[Optional[int], bool]:
        """
        Classify a single token.

        Returns:
            (entry_id, is_polysemous)
            - entry_id is set for mono-semantic and non-disambiguated tokens
            - entry_id is None for polysemous tokens (need BERT)
            - is_polysemous is True only for polysemous tokens
        """
        if lemma_token[0] == "_DELETED_":
            return None, False
        if lemma_token not in self._vocab_tokens:
            return None, False

        # Check non-disambiguated first (no BERT needed)
        if lemma_token in self._non_disambiguated_entries:
            return self._non_disambiguated_entries[lemma_token], False

        # Check mono-semantic (no BERT needed)
        if lemma_token in self._mono_entries:
            return self._mono_entries[lemma_token], False

        # Check polysemous (BERT needed)
        if lemma_token in self._poly_entries:
            return None, True

        return None, False

    def _classify_sentence(
        self, original_tokens: List[str], lemma_tokens: List[LemmaToken]
    ) -> Tuple[List[Optional[int]], List[PolyTokenInfo]]:
        """Classify all tokens, returning entries and polysemous token info."""
        entries: List[Optional[int]] = []
        poly_infos: List[PolyTokenInfo] = []

        char_pos = 0
        for idx, (orig, lemma) in enumerate(zip(original_tokens, lemma_tokens)):
            entry_id, is_poly = self._classify_token(lemma)
            entries.append(entry_id)

            if is_poly:
                poly_infos.append(
                    PolyTokenInfo(
                        token_idx=idx,
                        char_start=char_pos,
                        char_end=char_pos + len(orig),
                        lemma_token=lemma,
                    )
                )

            char_pos += len(orig) + 1  # +1 for space

        return entries, poly_infos

    def _find_best_entry(self, lemma_token: LemmaToken, embedding: np.ndarray) -> int:
        """Find best matching entry using cosine similarity."""
        candidates = self._poly_entries[lemma_token]
        best_id = candidates[0][0]
        best_score = -1.0

        emb_norm = np.linalg.norm(embedding)
        if emb_norm == 0:
            return best_id

        for entry_id, meaning_emb in candidates:
            meaning_norm = np.linalg.norm(meaning_emb)
            if meaning_norm > 0:
                score = np.dot(embedding, meaning_emb) / (emb_norm * meaning_norm)
                if score > best_score:
                    best_score = score
                    best_id = entry_id

        return best_id

    def _process_bert_tasks(self, tasks: List[BertTask]) -> None:
        """Run BERT on tasks and fill in polysemous entry IDs."""
        embeddings = self._bert.extract_embeddings(tasks)

        for (task_idx, token_idx), emb in embeddings.items():
            task = tasks[task_idx]
            lemma_token = next(
                pi.lemma_token for pi in task.poly_infos if pi.token_idx == token_idx
            )
            task.entries[token_idx] = self._find_best_entry(lemma_token, emb)

    def process(
        self,
        sentence_store: SentenceStore,
        lemmatized_store: LemmatizedStore,
        output_path: Path,
        use_corrections: bool = True,
    ) -> DisambiguatedStore:
        """Process all sentences and write disambiguation results."""
        logger.info(
            f"Starting disambiguation (bert_batch_tokens={self.bert_batch_tokens})"
        )

        # Select lemma source (both apply POS mapping if set)
        if use_corrections and lemmatized_store.has_corrections():
            logger.info("Using corrected lemmas")
            sentences = lemmatized_store.iter_corrected_with_sentences(sentence_store)
        else:
            logger.info("Using original lemmas")
            sentences = lemmatized_store.iter_with_sentences(sentence_store)

        disambiguated_store = DisambiguatedStore(output_path)

        with disambiguated_store.open_writer() as writer:
            stats = self._process_sentences(sentences, writer)

        logger.info(
            f"Done: {stats['total']:,} sentences, "
            f"{stats['poly']:,} with polysemous ({stats['poly']*100/max(1,stats['total']):.1f}%), "
            f"{stats['disambiguated']:,} tokens disambiguated"
        )

        return disambiguated_store

    def _process_sentences(
        self,
        sentences: Iterator[Tuple[int, List[str], List[LemmaToken]]],
        writer,
    ) -> Dict[str, int]:
        """Main processing loop with batching logic."""
        buffer = OrderedWriteBuffer(writer)
        bert_queue: List[BertTask] = []
        queue_tokens = 0

        stats = {"total": 0, "poly": 0, "disambiguated": 0}

        for sid, original_tokens, lemma_tokens in sentences:
            entries, poly_infos = self._classify_sentence(
                original_tokens, lemma_tokens
            )

            if not poly_infos:
                # No polysemous tokens - write directly
                buffer.add(sid, entries)
            else:
                # Queue for BERT processing
                bert_queue.append(
                    BertTask(
                        sid=sid,
                        text=" ".join(original_tokens),
                        token_count=len(original_tokens),
                        entries=entries,
                        poly_infos=poly_infos,
                    )
                )
                queue_tokens += len(original_tokens)
                stats["poly"] += 1

                # Flush if batch is full
                if queue_tokens >= self.bert_batch_tokens:
                    self._flush_bert_queue(bert_queue, buffer)
                    bert_queue = []
                    queue_tokens = 0

            stats["total"] += 1
            if stats["total"] % 50000 == 0:
                logger.info(
                    f"Processed {stats['total']:,} sentences, "
                    f"{stats['poly']:,} with polysemous"
                )

        # Final flush
        if bert_queue:
            self._flush_bert_queue(bert_queue, buffer)

        stats["disambiguated"] = buffer.total_disambiguated
        return stats

    def _flush_bert_queue(
        self, tasks: List[BertTask], buffer: OrderedWriteBuffer
    ) -> None:
        """Process BERT queue and add results to write buffer."""
        self._process_bert_tasks(tasks)
        for task in tasks:
            buffer.add(task.sid, task.entries)
