"""Vocabulary data structures for translations."""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)

# Type alias for lemma tokens - hashable, usable as dict key
LemmaToken = Tuple[str, str]  # (lemma, pos)


@dataclass
class Meaning:
    """A single meaning/sense of a lemma."""

    definition_es: str
    estimated_share: float
    translations: List[Dict[str, str]]
    structure: Optional[str] = None
    example_es: Optional[str] = None
    flags: Optional[List[str]] = None
    gender: Optional[str] = None  # Spanish grammatical gender ("m" or "f")
    # Embedding metadata (internal, prefixed with _)
    _bert_embedding: Optional[List[float]] = None
    _target_word: Optional[str] = None
    _target_start: Optional[int] = None
    _target_end: Optional[int] = None

    @property
    def primary_translation(self) -> str:
        """Returns the primary (first) translation text."""
        if self.translations:
            primary = self.translations[0]
            hint = f" ({primary.get('hint', '')})" if primary.get("hint") else ""
            return primary.get("text", "") + hint
        return ""

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
            estimated_share=data.get("estimated_share", 0.0),
            translations=data.get("translations", []),
            structure=data.get("structure"),
            example_es=data.get("example_es"),
            flags=data.get("flags", []),
            gender=data.get("gender"),
            _bert_embedding=data.get(
                "meaning_embedding"
            ),  # JSON uses meaning_embedding
            _target_word=data.get("_target_word"),
            _target_start=data.get("_target_start"),
            _target_end=data.get("_target_end"),
        )

    def to_dict(self, include_embeddings: bool = True) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "definition_es": self.definition_es,
            "estimated_share": self.estimated_share,
            "translations": self.translations,
        }
        if self.structure is not None:
            result["structure"] = self.structure
        if self.example_es is not None:
            result["example_es"] = self.example_es
        if self.flags is not None:
            result["flags"] = self.flags
        if self.gender is not None:
            result["gender"] = self.gender
        if include_embeddings and self._bert_embedding is not None:
            result["meaning_embedding"] = self._bert_embedding
        if self._target_word is not None:
            result["_target_word"] = self._target_word
        if self._target_start is not None:
            result["_target_start"] = self._target_start
        if self._target_end is not None:
            result["_target_end"] = self._target_end
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

    def to_dict(self, include_embeddings: bool = True) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "lemma": self.lemma,
            "pos": self.pos,
            "frequency": self.frequency,
            "meanings": [m.to_dict(include_embeddings) for m in self.meanings],
        }


@dataclass
class WordVariant:
    """
    A specific meaning variant of a lemma.
    Each Translation can have multiple meanings - each becomes a WordVariant.
    The id is the index in Vocabulary.word_variants (integer for direct array access).
    """

    id: int  # Index in Vocabulary.word_variants list
    lemma: str
    pos: str
    translation_frequency: int  # Total frequency of the parent translation
    meaning: Meaning
    source_translation_id: str  # Translation.id (stays string for JSON compatibility)
    meaning_idx: int  # Index within the source translation's meanings

    @property
    def frequency(self) -> int:
        """Derived frequency based on translation frequency and meaning's estimated share."""
        return int(self.translation_frequency * self.meaning.estimated_share)

    @property
    def string_id(self) -> str:
        """String ID for JSON serialization (format: 'translation_id_meaning_idx')."""
        return f"{self.source_translation_id}_{self.meaning_idx}"

    @property
    def lemma_token(self) -> LemmaToken:
        """Returns (lemma, pos) tuple."""
        return (self.lemma, self.pos)

    @property
    def primary_translation(self) -> str:
        """Returns the primary translation text."""
        return self.meaning.primary_translation

    @property
    def bert_embedding(self) -> Optional[np.ndarray]:
        """Returns BERT embedding as numpy array."""
        return self.meaning.bert_embedding

    @classmethod
    def from_translations(cls, translations: List[Translation]) -> List["WordVariant"]:
        """Create WordVariants from Translations, sorted by derived frequency."""
        # First pass: create variants without final IDs
        variants_unsorted = []
        for translation in translations:
            for idx, meaning in enumerate(translation.meanings):
                derived_freq = int(translation.frequency * meaning.estimated_share)
                variants_unsorted.append(
                    {
                        "lemma": translation.lemma,
                        "pos": translation.pos,
                        "translation_frequency": translation.frequency,
                        "derived_freq": derived_freq,
                        "meaning": meaning,
                        "source_translation_id": translation.id,
                        "meaning_idx": idx,
                    }
                )

        # Sort by frequency descending, then by translation_id and meaning_idx for determinism
        variants_unsorted.sort(
            key=lambda v: (-v["derived_freq"], v["source_translation_id"], v["meaning_idx"])
        )

        # Second pass: assign IDs based on sorted position
        variants = []
        for idx, v in enumerate(variants_unsorted):
            variants.append(
                cls(
                    id=idx,  # Index in sorted list is the ID
                    lemma=v["lemma"],
                    pos=v["pos"],
                    translation_frequency=v["translation_frequency"],
                    meaning=v["meaning"],
                    source_translation_id=v["source_translation_id"],
                    meaning_idx=v["meaning_idx"],
                )
            )
        return variants


class Vocabulary:
    """Central storage for vocabulary data with optional co-occurrence matrix."""

    def __init__(self, translations: List[Translation]):
        """Initialize from list of Translation objects."""
        self.translations = translations
        self.translation_by_id = {t.id: t for t in translations}
        self.translation_by_lemma_token: Dict[LemmaToken, Translation] = {
            t.lemma_token: t for t in translations
        }

        # word_variants list - index IS the variant ID
        self.word_variants = WordVariant.from_translations(translations)

        # Lookup: LemmaToken → List of variant IDs (integers)
        self.variants_by_lemma_token: Dict[LemmaToken, List[int]] = {}
        for variant in self.word_variants:
            key = variant.lemma_token
            if key not in self.variants_by_lemma_token:
                self.variants_by_lemma_token[key] = []
            self.variants_by_lemma_token[key].append(variant.id)

        # Optional co-occurrence matrix (set after disambiguation)
        self.cooccurrence_matrix: Optional[sparse.csr_matrix] = None

    @classmethod
    def from_translations_list(cls, translations: List[Translation]) -> "Vocabulary":
        """Create Vocabulary from list of Translation objects."""
        return cls(translations)

    @classmethod
    def from_dicts(cls, data: List[Dict[str, Any]]) -> "Vocabulary":
        """Create Vocabulary from list of dictionaries."""
        translations = [Translation.from_dict(item) for item in data]
        return cls(translations)

    # File names for directory-based storage
    TRANSLATIONS_FILE = "translations.json"
    COOCCURRENCE_FILE = "cooccurrence_matrix.npz"

    @classmethod
    def from_dir(cls, dir_path: str) -> "Vocabulary":
        """Load Vocabulary from directory."""
        dir_path = Path(dir_path)

        translations_path = dir_path / cls.TRANSLATIONS_FILE
        with open(translations_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        vocab = cls.from_dicts(json_data)

        cooccurrence_path = dir_path / cls.COOCCURRENCE_FILE
        if cooccurrence_path.exists():
            vocab.cooccurrence_matrix = sparse.load_npz(cooccurrence_path)

        return vocab

    def to_dicts(self, include_embeddings: bool = True) -> List[Dict[str, Any]]:
        """Convert to list of dictionaries for JSON serialization."""
        return [t.to_dict(include_embeddings) for t in self.translations]

    def save(self, dir_path: str, include_embeddings: bool = True) -> None:
        """Save Vocabulary to directory."""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        translations_path = dir_path / self.TRANSLATIONS_FILE
        with open(translations_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dicts(include_embeddings), f, ensure_ascii=False, indent=2)

        if self.cooccurrence_matrix is not None:
            cooccurrence_path = dir_path / self.COOCCURRENCE_FILE
            sparse.save_npz(cooccurrence_path, self.cooccurrence_matrix, compressed=True)

    def get_translation(self, lemma_token: LemmaToken) -> Optional[Translation]:
        """Get Translation by lemma token."""
        return self.translation_by_lemma_token.get(lemma_token)

    def get_word_variant(self, variant_id: int) -> WordVariant:
        """Get WordVariant by ID (direct array access, O(1))."""
        return self.word_variants[variant_id]

    def get_variant_ids_for_lemma(self, lemma_token: LemmaToken) -> List[int]:
        """Get list of variant IDs for a lemma token."""
        return self.variants_by_lemma_token.get(lemma_token, [])

    def get_all_word_variants(self) -> List[WordVariant]:
        """Get all word variants sorted by frequency (descending)."""
        return self.word_variants

    def sample_word_variants(
        self, n: int, seed: Optional[int] = None
    ) -> List[WordVariant]:
        """Sample n random word variants."""
        if seed is not None:
            random.seed(seed)
        if n > len(self.word_variants):
            raise ValueError(
                f"Requested {n} variants, only {len(self.word_variants)} available"
            )
        return random.sample(self.word_variants, n)

    def sample_word_variants_groups(
        self, group_sizes: List[int], seed: Optional[int] = None
    ) -> List[List[WordVariant]]:
        """
        Sample random word variants into groups.

        Constraints:
        - No two WordVariants from the same Translation in the same group
        - Each WordVariant can appear in at most one group
        """
        if seed is not None:
            random.seed(seed)

        all_variants = list(self.word_variants)
        random.shuffle(all_variants)

        total_needed = sum(group_sizes)
        if total_needed > len(all_variants):
            raise ValueError(
                f"Requested {total_needed} variants, only {len(all_variants)} available"
            )

        result = []
        used_variant_ids = set()
        pool = all_variants.copy()

        for group_size in group_sizes:
            group = []
            used_trans_ids = set()
            attempts = 0
            max_attempts = len(pool) * 2

            while len(group) < group_size and attempts < max_attempts:
                if not pool:
                    raise ValueError(
                        f"Not enough variants for group of size {group_size}"
                    )

                candidate = pool.pop(0)
                attempts += 1

                if candidate.id in used_variant_ids:
                    continue
                if candidate.source_translation_id in used_trans_ids:
                    pool.append(candidate)
                    continue

                group.append(candidate)
                used_variant_ids.add(candidate.id)
                used_trans_ids.add(candidate.source_translation_id)

            if len(group) < group_size:
                raise ValueError(
                    f"Could only create group of {len(group)}, needed {group_size}"
                )
            result.append(group)

        return result

    def get_cooccurrences(
        self, variant_id: int, top_k: int = 20
    ) -> List[Tuple[int, float]]:
        """
        Get top co-occurring variants for a given variant.

        Args:
            variant_id: Word variant ID (integer index)
            top_k: Number of top co-occurrences to return

        Returns:
            List of (variant_id, count) tuples, sorted by count descending
        """
        if self.cooccurrence_matrix is None:
            raise ValueError("No co-occurrence matrix available")

        if variant_id < 0 or variant_id >= len(self.word_variants):
            return []

        row = self.cooccurrence_matrix.getrow(variant_id).toarray().flatten()
        top_indices = np.argsort(row)[-top_k:][::-1]

        results = []
        for i in top_indices:
            if row[i] > 0:
                results.append((i, float(row[i])))

        return results

    def __len__(self) -> int:
        return len(self.word_variants)

    def __repr__(self) -> str:
        matrix_info = ""
        if self.cooccurrence_matrix is not None:
            nnz = self.cooccurrence_matrix.nnz
            matrix_info = f", cooccurrence={nnz:,} entries"
        return f"Vocabulary(translations={len(self.translations)}, variants={len(self.word_variants)}{matrix_info})"
