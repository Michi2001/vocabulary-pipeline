"""
Sentence-Token Store for Word Sense Disambiguation.

Stores sentence→token mappings in compressed JSONL format for later disambiguation.
Designed for memory-efficient streaming with 100M+ sentences.
"""

import gzip
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Tuple, Optional

logger = logging.getLogger(__name__)


@dataclass
class SentenceTokens:
    """Tokens for a single sentence."""

    sid: int
    tokens: List[Tuple[str, str, int, int]]  # [(lemma, pos, char_start, char_end), ...]

    def get_lemmas_with_pos(self) -> List[Tuple[str, str]]:
        """Return list of (lemma, pos) tuples."""
        return [(t[0], t[1]) for t in self.tokens]

    def get_token_at_position(self, char_pos: int) -> Optional[Tuple[str, str, int, int]]:
        """Find token containing the given character position."""
        for token in self.tokens:
            if token[2] <= char_pos < token[3]:
                return token
        return None


class SentenceTokenStore:
    """
    Memory-efficient storage for sentence-token mappings.

    Uses compressed JSONL format (.jsonl.gz) for streaming read/write.
    Format per line: {"sid": int, "tokens": [[lemma, pos, start, end], ...]}
    """

    def __init__(self, path: Path):
        """Initialize store with file path."""
        self.path = Path(path)
        self._file = None
        self._mode = None
        self._sentence_count = 0

    def open_write(self) -> "SentenceTokenStore":
        """Open store for writing (streaming append)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = gzip.open(self.path, "wt", encoding="utf-8")
        self._mode = "write"
        self._sentence_count = 0
        return self

    def write(self, tokens: List[Tuple[str, str, int, int]]) -> int:
        """
        Write tokens for one sentence.

        Args:
            tokens: List of (lemma, pos, char_start, char_end) tuples

        Returns:
            Sentence ID assigned to this sentence
        """
        if self._mode != "write":
            raise RuntimeError("Store not opened for writing")

        sid = self._sentence_count
        # Compact format: list of lists instead of dicts
        token_list = [[t[0], t[1], t[2], t[3]] for t in tokens]
        line = json.dumps({"sid": sid, "tokens": token_list}, ensure_ascii=False)
        self._file.write(line + "\n")
        self._sentence_count += 1

        if self._sentence_count % 100000 == 0:
            logger.info(f"SentenceTokenStore: Written {self._sentence_count:,} sentences")

        return sid

    def close(self):
        """Close the store file."""
        if self._file:
            self._file.close()
            self._file = None
            if self._mode == "write":
                logger.info(
                    f"SentenceTokenStore: Finalized with {self._sentence_count:,} sentences"
                )
            self._mode = None

    def iter_sentences(self) -> Generator[SentenceTokens, None, None]:
        """
        Iterate over all stored sentences (streaming read).

        Yields:
            SentenceTokens objects
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Store file not found: {self.path}")

        with gzip.open(self.path, "rt", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    tokens = [
                        (t[0], t[1], t[2], t[3]) for t in data["tokens"]
                    ]
                    yield SentenceTokens(sid=data["sid"], tokens=tokens)
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    logger.warning(f"Skipping malformed line {line_num}: {e}")

    def get_sentence_count(self) -> int:
        """Get total number of sentences (requires full scan if not in write mode)."""
        if self._mode == "write":
            return self._sentence_count

        if not self.path.exists():
            return 0

        count = 0
        with gzip.open(self.path, "rt", encoding="utf-8") as f:
            for _ in f:
                count += 1
        return count

    def exists(self) -> bool:
        """Check if store file exists."""
        return self.path.exists()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
