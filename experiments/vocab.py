"""Vocabulary data structures for translations.json."""

import json
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class Meaning:
    """A single meaning/sense of a lemma."""

    definition_es: str
    estimated_share: float
    translations: List[Dict[str, str]]
    embedding: Optional[List[float]] = None
    structure: Optional[str] = None
    example_es: Optional[str] = None
    flags: Optional[List[str]] = None
    topics: Optional[List[str]] = None

    @property
    def primary_translation(self) -> str:
        """Returns the primary (first) translation text."""
        if self.translations:
            primary_translation = self.translations[0]
            hint = (
                f"( {primary_translation.get('hint', '')} )"
                if primary_translation.get("hint")
                else ""
            )
            combined = primary_translation.get("text", "") + hint
            return combined
        return ""


@dataclass
class Translation:
    """A lemma with its meanings from translations.json."""

    id: str
    lemma: str
    pos: str
    frequency: int
    meanings: List[Meaning]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Translation":
        meanings = []
        for m in data.get("meanings", []):
            meanings.append(
                Meaning(
                    definition_es=m.get("definition_es", ""),
                    estimated_share=m.get("estimated_share", 0.0),
                    translations=m.get("translations", []),
                    embedding=m.get("embedding"),
                    structure=m.get("structure"),
                    example_es=m.get("example_es"),
                    flags=m.get("flags", []),
                    topics=m.get("topics"),
                )
            )
        return cls(
            id=data.get("id", ""),
            lemma=data.get("lemma", ""),
            pos=data.get("pos", ""),
            frequency=data.get("frequency", 0),
            meanings=meanings,
        )


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

    @property
    def primary_translation(self) -> str:
        return self.meaning.primary_translation

    @property
    def embedding(self) -> Optional[np.ndarray]:
        if self.meaning.embedding:
            return np.array(self.meaning.embedding)
        return None

    @classmethod
    def from_translations(cls, translations: List[Translation]) -> List["WordVariant"]:
        """Create WordVariants from Translations, sorted by derived frequency."""
        variants = []
        for translation in translations:
            for meaning in translation.meanings:
                derived_freq = int(translation.frequency * meaning.estimated_share)
                variants.append(
                    cls(
                        id="",
                        lemma=translation.lemma,
                        pos=translation.pos,
                        frequency=derived_freq,
                        meaning=meaning,
                        source_translation_id=translation.id,
                    )
                )
        variants.sort(key=lambda v: v.frequency, reverse=True)
        for idx, v in enumerate(variants):
            v.id = f"{idx:06d}"
        return variants


class Vocabulary:
    """Central storage for vocabulary data loaded from translations.json."""

    def __init__(self, translations: List[Translation]):
        self.translation_by_id = {t.id: t for t in translations}
        self.word_variants = WordVariant.from_translations(translations)
        self.word_variant_by_id = {v.id: v for v in self.word_variants}

    @classmethod
    def from_file(cls, file_path: str) -> "Vocabulary":
        """Load vocabulary from translations.json file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        translations = [Translation.from_dict(item) for item in data]
        return cls(translations)

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
        return f"Vocabulary(translations={len(self.translation_by_id)}, variants={len(self.word_variants)})"


if __name__ == "__main__":
    vocab = Vocabulary.from_file(
        "output/spanish_lemma_frequencies/slice_6000_6400_with_vectors/translations.json"
    )
    print(vocab)

    print("\nTop 5 variants:")
    for v in vocab.get_all_word_variants()[:5]:
        print(f"  {v.lemma} ({v.pos}): {v.primary_translation} - freq={v.frequency}")
        if v.embedding is not None:
            print(f"    embedding dim: {len(v.embedding)}")

    sampled_groups = vocab.sample_word_variants_groups([30, 30, 30], seed=123)
    for sampled_group in sampled_groups:
        print("\nSampled group:")
        for v in sampled_group:
            print(
                f"  {v.lemma} ({v.pos}): {v.primary_translation} - freq={v.frequency}"
            )
