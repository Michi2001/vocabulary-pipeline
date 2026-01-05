"""Vocabulary data structures for translations."""

import json
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

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
            _bert_embedding=data.get("meaning_embedding"),  # JSON uses meaning_embedding
            _target_word=data.get("_target_word"),
            _target_start=data.get("_target_start"),
            _target_end=data.get("_target_end"),
        )

    def to_dict(self) -> Dict[str, Any]:
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
        if self.flags:
            result["flags"] = self.flags
        if self._bert_embedding is not None:
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "lemma": self.lemma,
            "pos": self.pos,
            "frequency": self.frequency,
            "meanings": [m.to_dict() for m in self.meanings],
        }


@dataclass
class WordVariant:
    """
    A specific meaning variant of a lemma.
    Each Translation can have multiple meanings - each becomes a WordVariant.
    """

    id: str
    lemma: str
    pos: str
    frequency: int  # Derived: translation.frequency * meaning.estimated_share
    meaning: Meaning
    source_translation_id: str
    meaning_idx: int  # Index within the source translation's meanings

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
        variants = []
        for translation in translations:
            for idx, meaning in enumerate(translation.meanings):
                derived_freq = int(translation.frequency * meaning.estimated_share)
                # Create unique ID: translation_id + meaning_index
                variant_id = f"{translation.id}_{idx}"
                variants.append(
                    cls(
                        id=variant_id,
                        lemma=translation.lemma,
                        pos=translation.pos,
                        frequency=derived_freq,
                        meaning=meaning,
                        source_translation_id=translation.id,
                        meaning_idx=idx,
                    )
                )
        # Sort by frequency descending
        variants.sort(key=lambda v: v.frequency, reverse=True)
        return variants


class Vocabulary:
    """Central storage for vocabulary data."""

    def __init__(self, translations: List[Translation]):
        """Initialize from list of Translation objects."""
        self.translations = translations
        self.translation_by_id = {t.id: t for t in translations}
        self.translation_by_lemma_token: Dict[LemmaToken, Translation] = {
            t.lemma_token: t for t in translations
        }
        self.word_variants = WordVariant.from_translations(translations)
        self.word_variant_by_id = {v.id: v for v in self.word_variants}

    @classmethod
    def from_translations_list(cls, translations: List[Translation]) -> "Vocabulary":
        """Create Vocabulary from list of Translation objects."""
        return cls(translations)

    @classmethod
    def from_dicts(cls, data: List[Dict[str, Any]]) -> "Vocabulary":
        """Create Vocabulary from list of dictionaries."""
        translations = [Translation.from_dict(item) for item in data]
        return cls(translations)

    @classmethod
    def from_file(cls, file_path: str) -> "Vocabulary":
        """Load Vocabulary from translations.json file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dicts(data)

    def to_dicts(self) -> List[Dict[str, Any]]:
        """Convert to list of dictionaries for JSON serialization."""
        return [t.to_dict() for t in self.translations]

    def to_file(self, file_path: str) -> None:
        """Save Vocabulary to JSON file."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dicts(), f, ensure_ascii=False, indent=2)

    def get_translation(self, lemma_token: LemmaToken) -> Optional[Translation]:
        """Get Translation by lemma token."""
        return self.translation_by_lemma_token.get(lemma_token)

    def get_word_variant(self, variant_id: str) -> Optional[WordVariant]:
        """Get WordVariant by ID."""
        return self.word_variant_by_id.get(variant_id)

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

    def __len__(self) -> int:
        return len(self.word_variants)

    def __repr__(self) -> str:
        return f"Vocabulary(translations={len(self.translations)}, variants={len(self.word_variants)})"
