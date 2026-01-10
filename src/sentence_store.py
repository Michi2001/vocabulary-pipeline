"""
Index-aligned Sentence Store.

Stores sentence data in separate compressed JSONL files.
All token lists per sentence have the same length and order (index-aligned).
"""

import gzip
import json
import logging
import pickle
from pathlib import Path
from typing import Generator, List, Tuple, Optional, Iterator, Dict, Set

from src.vocab import LemmaToken

logger = logging.getLogger(__name__)

# Type alias for correction mapping
CorrectionMapping = Dict[LemmaToken, LemmaToken]


class _BaseWriter:
    """Base class for streaming writers."""

    def __init__(self, path: Path):
        self.path = path
        self._file = None
        self._count = 0

    def _open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = gzip.open(self.path, "wt", encoding="utf-8")
        self._count = 0

    def _write_line(self, data: dict):
        line = json.dumps(data, ensure_ascii=False)
        self._file.write(line + "\n")
        self._count += 1
        if self._count % 100000 == 0:
            logger.info(f"{self.__class__.__name__}: Written {self._count:,} entries")

    def close(self):
        if self._file:
            self._file.close()
            self._file = None
            logger.info(
                f"{self.__class__.__name__}: Finalized with {self._count:,} entries"
            )

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class SentencesWriter(_BaseWriter):
    """Writer for original sentence tokens."""

    def write(self, sid: int, tokens: List[str]):
        """Write original tokens for a sentence."""
        self._write_line({"sid": sid, "tokens": tokens})


class LemmatizedWriter(_BaseWriter):
    """Writer for lemmatized tokens."""

    def write(self, sid: int, tokens: List[LemmaToken]):
        """Write lemmatized tokens (lemma, pos) for a sentence."""
        # Store as [[lemma, pos], ...] for compact JSON
        token_list = [[t[0], t[1]] for t in tokens]
        self._write_line({"sid": sid, "tokens": token_list})


class DisambiguatedWriter(_BaseWriter):
    """Writer for disambiguated variant IDs."""

    def write(self, sid: int, variants: List[Optional[str]]):
        """Write variant IDs for a sentence. None for tokens not in vocabulary."""
        self._write_line({"sid": sid, "variants": variants})


class SentenceStore:
    """
    Manages sentence data in separate compressed JSONL files.
    All token lists are index-aligned (same length/order).

    Files:
    - sentences.jsonl.gz:     {sid, tokens: ["El", "gato", ...]}
    - lemmatized.jsonl.gz:    {sid, tokens: [["el","DET"], ["gato","NOUN"], ...]}
    - correction_mapping.pkl: Correction mapping and deleted tokens
    - disambiguated.jsonl.gz: {sid, variants: ["000123", null, ...]}
    """

    def __init__(self, base_path: Path):
        """
        Initialize SentenceStore.

        Args:
            base_path: Directory path for store files
        """
        self.base_path = Path(base_path)
        self._sentences_path = self.base_path / "sentences.jsonl.gz"
        self._lemmatized_path = self.base_path / "lemmatized.jsonl.gz"
        self._correction_mapping_path = self.base_path / "correction_mapping.pkl"
        self._disambiguated_path = self.base_path / "disambiguated.jsonl.gz"

        # Lazy-loaded correction data
        self._correction_mapping: Optional[CorrectionMapping] = None
        self._deleted_tokens: Optional[Set[LemmaToken]] = None
        self._corrections_loaded = False

    # Writer methods
    def open_sentences_writer(self) -> SentencesWriter:
        """Open writer for original sentence tokens."""
        return SentencesWriter(self._sentences_path)

    def open_lemmatized_writer(self) -> LemmatizedWriter:
        """Open writer for lemmatized tokens."""
        return LemmatizedWriter(self._lemmatized_path)

    def open_disambiguated_writer(self) -> DisambiguatedWriter:
        """Open writer for disambiguation results."""
        return DisambiguatedWriter(self._disambiguated_path)

    # Correction mapping methods
    def save_correction_mapping(
        self,
        correction_mapping: CorrectionMapping,
        deleted_tokens: Set[LemmaToken],
    ) -> None:
        """
        Save correction mapping and deleted tokens to disk.

        Args:
            correction_mapping: Mapping from original to corrected tokens
            deleted_tokens: Set of tokens marked for deletion
        """
        self.base_path.mkdir(parents=True, exist_ok=True)
        data = {
            "correction_mapping": correction_mapping,
            "deleted_tokens": deleted_tokens,
        }
        with open(self._correction_mapping_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(
            f"Saved correction mapping: {len(correction_mapping)} mappings, "
            f"{len(deleted_tokens)} deleted tokens"
        )

        # Update in-memory cache
        self._correction_mapping = correction_mapping
        self._deleted_tokens = deleted_tokens
        self._corrections_loaded = True

    def _load_correction_mapping(self) -> None:
        """Lazy load correction mapping from disk."""
        if self._corrections_loaded:
            return

        if not self._correction_mapping_path.exists():
            raise FileNotFoundError(
                f"Correction mapping not found: {self._correction_mapping_path}. "
                "Run LLM correction first or use iter_lemmatized() instead."
            )

        with open(self._correction_mapping_path, "rb") as f:
            data = pickle.load(f)
        self._correction_mapping = data.get("correction_mapping", {})
        self._deleted_tokens = data.get("deleted_tokens", set())
        logger.info(
            f"Loaded correction mapping: {len(self._correction_mapping)} mappings, "
            f"{len(self._deleted_tokens)} deleted tokens"
        )

        self._corrections_loaded = True

    # Read methods (streaming)
    def iter_sentences(self) -> Generator[Tuple[int, List[str]], None, None]:
        """Iterate over original sentence tokens."""
        if not self._sentences_path.exists():
            raise FileNotFoundError(f"Sentences file not found: {self._sentences_path}")

        with gzip.open(self._sentences_path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                yield data["sid"], data["tokens"]

    def iter_lemmatized(self) -> Generator[Tuple[int, List[LemmaToken]], None, None]:
        """Iterate over lemmatized tokens."""
        if not self._lemmatized_path.exists():
            raise FileNotFoundError(
                f"Lemmatized file not found: {self._lemmatized_path}"
            )

        with gzip.open(self._lemmatized_path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                # Convert [[lemma, pos], ...] back to [(lemma, pos), ...]
                tokens = [(t[0], t[1]) for t in data["tokens"]]
                yield data["sid"], tokens

    def iter_corrected(self) -> Generator[Tuple[int, List[LemmaToken]], None, None]:
        """
        Iterate over corrected tokens by applying correction mapping to lemmatized tokens.

        This method applies the correction mapping dynamically, avoiding the need to
        store a separate corrected.jsonl.gz file.
        """
        # Load correction mapping if not already loaded
        self._load_correction_mapping()

        # Apply corrections dynamically
        for sid, tokens in self.iter_lemmatized():
            corrected_tokens = []
            for token in tokens:
                if token in self._deleted_tokens:
                    # Mark deleted tokens with special value
                    corrected_tokens.append(("_DELETED_", ""))
                elif token in self._correction_mapping:
                    # Apply merge correction
                    corrected_tokens.append(self._correction_mapping[token])
                else:
                    # Keep unchanged
                    corrected_tokens.append(token)
            yield sid, corrected_tokens

    def iter_disambiguated(
        self,
    ) -> Generator[Tuple[int, List[Optional[str]]], None, None]:
        """Iterate over disambiguation results."""
        if not self._disambiguated_path.exists():
            raise FileNotFoundError(
                f"Disambiguated file not found: {self._disambiguated_path}"
            )

        with gzip.open(self._disambiguated_path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                yield data["sid"], data["variants"]

    def iter_sentences_with_lemmatized(
        self,
    ) -> Generator[Tuple[int, List[str], List[LemmaToken]], None, None]:
        """Iterate over sentences with their lemmatized tokens (parallel iteration)."""
        sentences_iter = self.iter_sentences()
        lemmatized_iter = self.iter_lemmatized()

        for (sid1, tokens), (sid2, lemmas) in zip(sentences_iter, lemmatized_iter):
            if sid1 != sid2:
                raise ValueError(
                    f"Sentence ID mismatch: sentences has {sid1}, lemmatized has {sid2}"
                )
            yield sid1, tokens, lemmas

    def iter_sentences_with_corrected(
        self,
    ) -> Generator[Tuple[int, List[str], List[LemmaToken]], None, None]:
        """Iterate over sentences with their corrected tokens (parallel iteration)."""
        sentences_iter = self.iter_sentences()
        corrected_iter = self.iter_corrected()

        for (sid1, tokens), (sid2, lemmas) in zip(sentences_iter, corrected_iter):
            if sid1 != sid2:
                raise ValueError(
                    f"Sentence ID mismatch: sentences has {sid1}, corrected has {sid2}"
                )
            yield sid1, tokens, lemmas

    # Utility methods
    def has_sentences(self) -> bool:
        """Check if sentences file exists."""
        return self._sentences_path.exists()

    def has_lemmatized(self) -> bool:
        """Check if lemmatized file exists."""
        return self._lemmatized_path.exists()

    def has_corrected(self) -> bool:
        """Check if correction mapping exists."""
        return self._correction_mapping_path.exists()

    def has_disambiguated(self) -> bool:
        """Check if disambiguated file exists."""
        return self._disambiguated_path.exists()

    def get_sentence_count(self) -> int:
        """Get total number of sentences (requires full scan)."""
        if not self._sentences_path.exists():
            return 0

        count = 0
        with gzip.open(self._sentences_path, "rt", encoding="utf-8") as f:
            for _ in f:
                count += 1
        return count

    def __repr__(self) -> str:
        parts = []
        if self.has_sentences():
            parts.append("sentences")
        if self.has_lemmatized():
            parts.append("lemmatized")
        if self.has_corrected():
            parts.append("corrections")
        if self.has_disambiguated():
            parts.append("disambiguated")
        return f"SentenceStore({self.base_path}, files=[{', '.join(parts)}])"
