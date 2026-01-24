"""Store for lemmatized tokens with correction mapping support."""

import pickle
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set, Tuple

from src.stores.base_store import BaseStore
from src.stores.sentence_store import SentenceStore
from src.vocab import LemmaToken

CorrectionMapping = Dict[LemmaToken, LemmaToken]
PosMapping = Dict[str, str]  # e.g. {"AUX": "VERB", "CCONJ": "CONJ"}


class LemmatizedStore(BaseStore[List[LemmaToken]]):
    """
    Stores lemmatized tokens with optional correction and POS mapping.
    Format: {"sid": int, "tokens": [["el", "DET"], ["gato", "NOUN"], ...]}
    """

    def __init__(self, path: Path):
        super().__init__(path)
        self._correction_mapping: Optional[CorrectionMapping] = None
        self._deleted_tokens: Optional[Set[LemmaToken]] = None
        self._correction_mapping_path: Optional[Path] = None
        self._corrections_loaded = False
        self._pos_mapping: Optional[PosMapping] = None

    def _parse_data(self, data: dict) -> List[LemmaToken]:
        return [(t[0], t[1]) for t in data["tokens"]]

    def _serialize_data(self, sid: int, data: List[LemmaToken]) -> dict:
        return {"sid": sid, "tokens": [[t[0], t[1]] for t in data]}

    def set_correction_mapping(
        self,
        mapping: CorrectionMapping,
        deleted_tokens: Set[LemmaToken],
        global_mapping_path: Path,
    ) -> None:
        """Set correction mapping (reference to global file)."""
        self._correction_mapping = mapping
        self._deleted_tokens = deleted_tokens
        self._correction_mapping_path = global_mapping_path
        self._corrections_loaded = True

    def load_correction_mapping(self, global_mapping_path: Path) -> None:
        """Load correction mapping from global file."""
        if self._corrections_loaded:
            return
        if not global_mapping_path.exists():
            raise FileNotFoundError(f"Correction mapping not found: {global_mapping_path}")

        with open(global_mapping_path, "rb") as f:
            data = pickle.load(f)

        self._correction_mapping = data.get("correction_mapping", {})
        self._deleted_tokens = data.get("deleted_tokens", set())
        self._correction_mapping_path = global_mapping_path
        self._corrections_loaded = True

    def has_corrections(self) -> bool:
        """Check if correction mapping is available."""
        return self._corrections_loaded and self._correction_mapping is not None

    def set_pos_mapping(self, pos_mapping: PosMapping) -> None:
        """Set POS mapping (e.g. AUX→VERB, CCONJ→CONJ)."""
        self._pos_mapping = pos_mapping

    def _apply_pos_mapping(self, token: LemmaToken) -> LemmaToken:
        """Apply POS mapping to a single token."""
        if self._pos_mapping and token[1] in self._pos_mapping:
            return (token[0], self._pos_mapping[token[1]])
        return token

    def iter_corrected(self) -> Generator[Tuple[int, List[LemmaToken]], None, None]:
        """Iterate with POS mapping and corrections applied dynamically.

        Order: POS mapping first, then correction lookup.
        This is because correction mapping uses mapped POS tags (from PostProcessor).
        """
        if not self._corrections_loaded:
            raise ValueError("Correction mapping not loaded. Call load_correction_mapping() first.")

        for sid, tokens in self:
            corrected = []
            for token in tokens:
                # First apply POS mapping (correction keys use mapped POS)
                mapped_token = self._apply_pos_mapping(token)

                if mapped_token in self._deleted_tokens:
                    corrected.append(("_DELETED_", ""))
                elif mapped_token in self._correction_mapping:
                    corrected.append(self._correction_mapping[mapped_token])
                else:
                    corrected.append(mapped_token)
            yield sid, corrected

    def iter_with_sentences(
        self, sentence_store: SentenceStore
    ) -> Generator[Tuple[int, List[str], List[LemmaToken]], None, None]:
        """Parallel iteration with sentence store. Applies POS mapping if set."""
        for (sid1, tokens), (sid2, lemmas) in zip(sentence_store, self):
            if sid1 != sid2:
                raise ValueError(f"SID mismatch: {sid1} vs {sid2}")
            mapped_lemmas = [self._apply_pos_mapping(t) for t in lemmas]
            yield sid1, tokens, mapped_lemmas

    def iter_corrected_with_sentences(
        self, sentence_store: SentenceStore
    ) -> Generator[Tuple[int, List[str], List[LemmaToken]], None, None]:
        """Parallel iteration with corrections and POS mapping applied."""
        for (sid1, tokens), (sid2, lemmas) in zip(sentence_store, self.iter_corrected()):
            if sid1 != sid2:
                raise ValueError(f"SID mismatch: {sid1} vs {sid2}")
            yield sid1, tokens, lemmas
