"""
Index-aligned Sentence Store.

Stores sentence data in separate compressed JSONL files.
All token lists per sentence have the same length and order (index-aligned).
"""

import gzip
import json
import logging
from pathlib import Path
from typing import Generator, List, Tuple, Optional, Iterator

from src.vocab import LemmaToken

logger = logging.getLogger(__name__)


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


class CorrectedWriter(_BaseWriter):
    """Writer for corrected lemmatized tokens."""

    def write(self, sid: int, tokens: List[LemmaToken]):
        """Write corrected tokens (lemma, pos) for a sentence."""
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
    - corrected.jsonl.gz:     {sid, tokens: [["el","DET"], ["gato","NOUN"], ...]}
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
        self._corrected_path = self.base_path / "corrected.jsonl.gz"
        self._disambiguated_path = self.base_path / "disambiguated.jsonl.gz"

    # Writer methods
    def open_sentences_writer(self) -> SentencesWriter:
        """Open writer for original sentence tokens."""
        return SentencesWriter(self._sentences_path)

    def open_lemmatized_writer(self) -> LemmatizedWriter:
        """Open writer for lemmatized tokens."""
        return LemmatizedWriter(self._lemmatized_path)

    def open_corrected_writer(self) -> CorrectedWriter:
        """Open writer for corrected tokens."""
        return CorrectedWriter(self._corrected_path)

    def open_disambiguated_writer(self) -> DisambiguatedWriter:
        """Open writer for disambiguation results."""
        return DisambiguatedWriter(self._disambiguated_path)

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
        """Iterate over corrected tokens."""
        if not self._corrected_path.exists():
            raise FileNotFoundError(f"Corrected file not found: {self._corrected_path}")

        with gzip.open(self._corrected_path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                tokens = [(t[0], t[1]) for t in data["tokens"]]
                yield data["sid"], tokens

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
        """Check if corrected file exists."""
        return self._corrected_path.exists()

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
            parts.append("corrected")
        if self.has_disambiguated():
            parts.append("disambiguated")
        return f"SentenceStore({self.base_path}, files=[{', '.join(parts)}])"
