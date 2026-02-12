"""Vocabulary data structures for translations."""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)

# Type alias for lemma tokens - hashable, usable as dict key
LemmaToken = Tuple[str, str]  # (lemma, pos)

# Default POS types that are disambiguated (have multiple VocabEntries per lemma)
DEFAULT_DISAMBIGUATED_POS = {"NOUN", "VERB", "ADJ", "ADV"}


@dataclass
class Meaning:
    """A single meaning/sense of a lemma."""

    definition_es: str
    translations: List[Dict[str, Any]]  # text, hints?: list[str], gender_target?, flags?
    estimated_share: Optional[float] = None  # Set by ShareUpdater after disambiguation
    structure: Optional[str] = None
    examples_es: Optional[List[str]] = None
    flags: Optional[List[str]] = None
    gender: Optional[str] = None  # Spanish grammatical gender ("m" or "f")
    _bert_embedding: Optional[List[float]] = None
    _target_starts: Optional[List[int]] = None
    _target_ends: Optional[List[int]] = None

    @property
    def primary_translation(self) -> str:
        """Returns the primary (first) translation text."""
        if self.translations:
            primary = self.translations[0]
            hints = primary.get("hints", [])
            hint = f" ({' / '.join(hints)})" if hints else ""
            return primary.get("text", "") + hint
        return ""

    @property
    def all_translations(self) -> str:
        """Returns all translations concatenated with ' / '."""
        if not self.translations:
            return ""
        parts = []
        for t in self.translations:
            text = t.get("text", "")
            hints = t.get("hints", [])
            hint = f" ({' / '.join(hints)})" if hints else ""
            parts.append(text + hint)
        return " / ".join(parts)

    @property
    def bert_embedding(self) -> Optional[np.ndarray]:
        """Returns BERT embedding as numpy array."""
        if self._bert_embedding:
            return np.array(self._bert_embedding)
        return None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Meaning":
        """Create Meaning from dictionary."""
        return cls(
            definition_es=data.get("definition_es", ""),
            translations=data.get("translations", []),
            estimated_share=data.get("estimated_share"),  # None if not present
            structure=data.get("structure"),
            examples_es=data.get("examples_es"),
            flags=data.get("flags", []),
            gender=data.get("gender"),
            _bert_embedding=data.get("meaning_embedding"),
            _target_starts=data.get("_target_starts"),
            _target_ends=data.get("_target_ends"),
        )

    def to_dict(
        self, include_embeddings: bool = True, include_private: bool = True
    ) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "definition_es": self.definition_es,
            "translations": self.translations,
        }
        if self.estimated_share is not None:
            result["estimated_share"] = self.estimated_share
        if self.structure is not None:
            result["structure"] = self.structure
        if self.examples_es is not None:
            result["examples_es"] = self.examples_es
        if self.flags is not None:
            result["flags"] = self.flags
        if self.gender is not None:
            result["gender"] = self.gender
        if include_embeddings and self._bert_embedding is not None:
            result["meaning_embedding"] = self._bert_embedding
        if include_private:
            if self._target_starts is not None:
                result["_target_starts"] = self._target_starts
            if self._target_ends is not None:
                result["_target_ends"] = self._target_ends
        return result


@dataclass
class Translation:
    """A lemma with its meanings."""

    id: str
    lemma: str
    pos: str
    frequency: int
    meanings: List[Meaning]

    @property
    def lemma_token(self) -> LemmaToken:
        """Returns (lemma, pos) tuple."""
        return (self.lemma, self.pos)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Translation":
        """Create Translation from dictionary."""
        meanings = [Meaning.from_dict(m) for m in data.get("meanings", [])]
        return cls(
            id=data.get("id", ""),
            lemma=data.get("lemma", ""),
            pos=data.get("pos", ""),
            frequency=data.get("frequency", 0),
            meanings=meanings,
        )

    def to_dict(
        self, include_embeddings: bool = True, include_private: bool = True
    ) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "lemma": self.lemma,
            "pos": self.pos,
            "frequency": self.frequency,
            "meanings": [
                m.to_dict(include_embeddings, include_private) for m in self.meanings
            ],
        }


@dataclass
class VocabEntry:
    """
    A vocabulary entry for the co-occurrence matrix.

    For disambiguated POS (NOUN, VERB, ADJ, ADV):
        - meaning_idx is set → represents a specific meaning
        - Multiple VocabEntries per Translation

    For non-disambiguated POS (ADP, CONJ, PRON, etc.):
        - meaning_idx is None → represents the entire lemma
        - One VocabEntry per Translation
    """

    id: int  # Index in Vocabulary.vocab_entries list (= matrix index)
    translation: Translation
    meaning_idx: Optional[int]  # None = non-disambiguated (all meanings together)

    @property
    def is_disambiguated(self) -> bool:
        """True if this entry represents a specific meaning."""
        return self.meaning_idx is not None

    @property
    def meaning(self) -> Optional[Meaning]:
        """Returns the Meaning object if disambiguated, else None."""
        if self.meaning_idx is not None:
            return self.translation.meanings[self.meaning_idx]
        return None

    @property
    def lemma(self) -> str:
        return self.translation.lemma

    @property
    def pos(self) -> str:
        return self.translation.pos

    @property
    def frequency(self) -> int:
        """Derived frequency based on translation frequency and meaning's share."""
        if self.meaning_idx is not None and self.meaning is not None:
            if self.meaning.estimated_share is not None:
                return int(self.translation.frequency * self.meaning.estimated_share)
        return self.translation.frequency

    @property
    def source_translation_id(self) -> str:
        """Translation.id for reference."""
        return self.translation.id

    @property
    def string_id(self) -> str:
        """String ID for JSON serialization."""
        if self.meaning_idx is not None:
            return f"{self.source_translation_id}_{self.meaning_idx}"
        return self.source_translation_id

    @property
    def lemma_token(self) -> LemmaToken:
        """Returns (lemma, pos) tuple."""
        return self.translation.lemma_token

    @property
    def primary_translation(self) -> str:
        """Returns the primary translation text."""
        if self.meaning is not None:
            return self.meaning.primary_translation
        # For non-disambiguated: use first meaning
        if self.translation.meanings:
            return self.translation.meanings[0].primary_translation
        return ""

    @property
    def all_translations(self) -> str:
        """Returns all translations concatenated."""
        if self.meaning is not None:
            return self.meaning.all_translations
        # Non-disambiguated: collect texts from all meanings, no hints
        parts = []
        for meaning in self.translation.meanings:
            for t in meaning.translations:
                text = t.get("text", "")
                if text and text not in parts:
                    parts.append(text)
        return " / ".join(parts)

    @property
    def bert_embedding(self) -> Optional[np.ndarray]:
        """Returns BERT embedding as numpy array (only for disambiguated entries)."""
        if self.meaning is not None:
            return self.meaning.bert_embedding
        return None

    @classmethod
    def from_translations(
        cls,
        translations: List[Translation],
        disambiguated_pos: Set[str],
    ) -> List["VocabEntry"]:
        """Create VocabEntries from Translations."""
        entries_unsorted = []

        for translation in translations:
            if translation.pos in disambiguated_pos:
                # Disambiguated: one entry per meaning
                n_meanings = len(translation.meanings)
                for idx, meaning in enumerate(translation.meanings):
                    share = meaning.estimated_share if meaning.estimated_share is not None else 1.0 / n_meanings
                    derived_freq = int(translation.frequency * share)
                    entries_unsorted.append({
                        "translation": translation,
                        "meaning_idx": idx,
                        "derived_freq": derived_freq,
                    })
            else:
                # Non-disambiguated: one entry for entire lemma
                entries_unsorted.append({
                    "translation": translation,
                    "meaning_idx": None,
                    "derived_freq": translation.frequency,
                })

        # Sort by frequency descending, then by translation_id and meaning_idx
        def sort_key(v):
            meaning_idx = v["meaning_idx"] if v["meaning_idx"] is not None else -1
            return (-v["derived_freq"], v["translation"].id, meaning_idx)

        entries_unsorted.sort(key=sort_key)

        return [
            cls(id=idx, translation=v["translation"], meaning_idx=v["meaning_idx"])
            for idx, v in enumerate(entries_unsorted)
        ]


class Vocabulary:
    """Central storage for vocabulary data with optional co-occurrence matrix."""

    def __init__(
        self,
        translations: List[Translation],
        disambiguated_pos: Optional[Set[str]] = None,
    ):
        """Initialize from list of Translation objects."""
        self.translations = translations
        self.disambiguated_pos = disambiguated_pos or DEFAULT_DISAMBIGUATED_POS
        self.translation_by_id = {t.id: t for t in translations}
        self.translation_by_lemma_token: Dict[LemmaToken, Translation] = {
            t.lemma_token: t for t in translations
        }

        # vocab_entries list - index IS the entry ID (= matrix index)
        self.vocab_entries = VocabEntry.from_translations(
            translations, self.disambiguated_pos
        )

        # Lookup: LemmaToken → List of entry IDs (integers)
        self.entries_by_lemma_token: Dict[LemmaToken, List[int]] = {}
        for entry in self.vocab_entries:
            key = entry.lemma_token
            if key not in self.entries_by_lemma_token:
                self.entries_by_lemma_token[key] = []
            self.entries_by_lemma_token[key].append(entry.id)

        # Optional co-occurrence matrix (set after disambiguation)
        self.cooccurrence_matrix: Optional[sparse.csr_matrix] = None

    @classmethod
    def from_translations_list(
        cls,
        translations: List[Translation],
        disambiguated_pos: Optional[Set[str]] = None,
    ) -> "Vocabulary":
        """Create Vocabulary from list of Translation objects."""
        return cls(translations, disambiguated_pos)

    @classmethod
    def from_dicts(
        cls,
        data: List[Dict[str, Any]],
        disambiguated_pos: Optional[Set[str]] = None,
    ) -> "Vocabulary":
        """Create Vocabulary from list of dictionaries."""
        translations = [Translation.from_dict(item) for item in data]
        return cls(translations, disambiguated_pos)

    # File names for directory-based storage
    TRANSLATIONS_FILE = "translations.json"
    COOCCURRENCE_FILE = "cooccurrence_matrix.npz"
    METADATA_FILE = "vocabulary_metadata.json"

    @classmethod
    def from_dir(
        cls,
        dir_path: str,
        disambiguated_pos: Optional[Set[str]] = None,
    ) -> "Vocabulary":
        """Load Vocabulary from directory."""
        dir_path = Path(dir_path)

        # Try to load metadata for disambiguated_pos
        metadata_path = dir_path / cls.METADATA_FILE
        if metadata_path.exists() and disambiguated_pos is None:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                disambiguated_pos = set(metadata.get("disambiguated_pos", []))

        translations_path = dir_path / cls.TRANSLATIONS_FILE
        with open(translations_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        vocab = cls.from_dicts(json_data, disambiguated_pos)

        cooccurrence_path = dir_path / cls.COOCCURRENCE_FILE
        if cooccurrence_path.exists():
            vocab.cooccurrence_matrix = sparse.load_npz(cooccurrence_path)

        return vocab

    def to_dicts(
        self, include_embeddings: bool = True, include_private: bool = True
    ) -> List[Dict[str, Any]]:
        """Convert to list of dictionaries for JSON serialization."""
        return [
            t.to_dict(include_embeddings, include_private) for t in self.translations
        ]

    def save(
        self,
        dir_path: str,
        include_embeddings: bool = True,
        include_private: bool = True,
        save_cooccurrence: bool = True,
    ) -> None:
        """Save Vocabulary to directory."""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        translations_path = dir_path / self.TRANSLATIONS_FILE
        with open(translations_path, "w", encoding="utf-8") as f:
            json.dump(
                self.to_dicts(include_embeddings, include_private),
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Save metadata
        metadata_path = dir_path / self.METADATA_FILE
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {"disambiguated_pos": list(self.disambiguated_pos)},
                f,
                ensure_ascii=False,
                indent=2,
            )

        if save_cooccurrence and self.cooccurrence_matrix is not None:
            cooccurrence_path = dir_path / self.COOCCURRENCE_FILE
            sparse.save_npz(
                cooccurrence_path, self.cooccurrence_matrix, compressed=True
            )

    def get_translation(self, lemma_token: LemmaToken) -> Optional[Translation]:
        """Get Translation by lemma token."""
        return self.translation_by_lemma_token.get(lemma_token)

    def get_vocab_entry(self, entry_id: int) -> VocabEntry:
        """Get VocabEntry by ID (direct array access, O(1))."""
        return self.vocab_entries[entry_id]

    def get_entry_ids_for_lemma(self, lemma_token: LemmaToken) -> List[int]:
        """Get list of entry IDs for a lemma token."""
        return self.entries_by_lemma_token.get(lemma_token, [])

    def get_all_vocab_entries(self) -> List[VocabEntry]:
        """Get all vocab entries sorted by frequency (descending)."""
        return self.vocab_entries

    def sample_vocab_entries(
        self, n: int, seed: Optional[int] = None
    ) -> List[VocabEntry]:
        """Sample n random vocab entries."""
        if seed is not None:
            random.seed(seed)
        if n > len(self.vocab_entries):
            raise ValueError(
                f"Requested {n} entries, only {len(self.vocab_entries)} available"
            )
        return random.sample(self.vocab_entries, n)

    def sample_vocab_entries_groups(
        self, group_sizes: List[int], seed: Optional[int] = None
    ) -> List[List[VocabEntry]]:
        """
        Sample random vocab entries into groups.

        Constraints:
        - No two VocabEntries from the same Translation in the same group
        - Each VocabEntry can appear in at most one group
        """
        if seed is not None:
            random.seed(seed)

        all_entries = list(self.vocab_entries)
        random.shuffle(all_entries)

        total_needed = sum(group_sizes)
        if total_needed > len(all_entries):
            raise ValueError(
                f"Requested {total_needed} entries, only {len(all_entries)} available"
            )

        result = []
        used_entry_ids = set()
        pool = all_entries.copy()

        for group_size in group_sizes:
            group = []
            used_trans_ids = set()
            attempts = 0
            max_attempts = len(pool) * 2

            while len(group) < group_size and attempts < max_attempts:
                if not pool:
                    raise ValueError(
                        f"Not enough entries for group of size {group_size}"
                    )

                candidate = pool.pop(0)
                attempts += 1

                if candidate.id in used_entry_ids:
                    continue
                if candidate.source_translation_id in used_trans_ids:
                    pool.append(candidate)
                    continue

                group.append(candidate)
                used_entry_ids.add(candidate.id)
                used_trans_ids.add(candidate.source_translation_id)

            if len(group) < group_size:
                raise ValueError(
                    f"Could only create group of {len(group)}, needed {group_size}"
                )
            result.append(group)

        return result

    def get_cooccurrences(
        self, entry_id: int, top_k: int = 20
    ) -> List[Tuple[int, float]]:
        """
        Get top co-occurring entries for a given entry.

        Args:
            entry_id: VocabEntry ID (integer index)
            top_k: Number of top co-occurrences to return

        Returns:
            List of (entry_id, count) tuples, sorted by count descending
        """
        if self.cooccurrence_matrix is None:
            raise ValueError("No co-occurrence matrix available")

        if entry_id < 0 or entry_id >= len(self.vocab_entries):
            return []

        row = self.cooccurrence_matrix.getrow(entry_id).toarray().flatten()
        top_indices = np.argsort(row)[-top_k:][::-1]

        results = []
        for i in top_indices:
            if row[i] > 0:
                results.append((i, float(row[i])))

        return results

    def __len__(self) -> int:
        return len(self.vocab_entries)

    def __repr__(self) -> str:
        matrix_info = ""
        if self.cooccurrence_matrix is not None:
            nnz = self.cooccurrence_matrix.nnz
            matrix_info = f", cooccurrence={nnz:,} entries"
        disamb_count = sum(1 for e in self.vocab_entries if e.is_disambiguated)
        non_disamb_count = len(self.vocab_entries) - disamb_count
        return (
            f"Vocabulary(translations={len(self.translations)}, "
            f"entries={len(self.vocab_entries)} "
            f"[{disamb_count} disambiguated, {non_disamb_count} non-disambiguated]"
            f"{matrix_info})"
        )
