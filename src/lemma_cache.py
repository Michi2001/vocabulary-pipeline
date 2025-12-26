"""Centralized cache for LLM corrections and translations."""

import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class LemmaCache:
    """
    Centralized cache using nested dictionary structure.

    Structure:
    {
        "corrections": {
            "model_config": {  # e.g., "gpt-4o-mini" or "gemini-2.0-flash|1.0|minimal"
                "prompt_hash": {  # 8-char hash of system prompt
                    "lemma|POS": {...}  # correction data
                }
            }
        },
        "translations": {
            "model_config": {
                "prompt_hash": {
                    "lemma|POS": {...}  # translation data
                }
            }
        }
    }
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize cache.

        Args:
            config: {
                'enabled': bool (default: True),
                'directory': str (default: 'cache')
            }
        """
        self.enabled = config.get("enabled", True)
        self.cache_dir = Path(config.get("directory", "cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Separate files for corrections and translations
        self.corrections_file = self.cache_dir / "corrections.pkl"
        self.translations_file = self.cache_dir / "translations.pkl"

        # Load caches into memory
        self.corrections = self._load_cache(self.corrections_file)
        self.translations = self._load_cache(self.translations_file)

        logger.info(f"Cache initialized: enabled={self.enabled}, directory={self.cache_dir}")

    def _load_cache(self, cache_file: Path) -> Dict:
        """Load cache from pickle file."""
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cache_data = pickle.load(f)
                logger.info(f"Loaded cache from {cache_file}")
                return cache_data
            except Exception as e:
                logger.warning(f"Failed to load cache from {cache_file}: {e}")
        return {}

    def _save_cache(self, cache_dict: Dict, cache_file: Path):
        """Save cache to pickle file."""
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(cache_dict, f)
        except Exception as e:
            logger.warning(f"Failed to save cache to {cache_file}: {e}")

    def get_correction(
        self, lemma: str, pos: str, model_config: str, prompt_hash: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached correction for a lemma.

        Args:
            lemma: Lemma string
            pos: POS tag
            model_config: Provider-specific model config string (from get_cache_key_components)
            prompt_hash: Hash of system prompt (8 chars)

        Returns:
            Cached correction dict or None if cache miss
        """
        if not self.enabled:
            return None

        lemma_key = f"{lemma}|{pos}"

        return (
            self.corrections.get(model_config, {})
            .get(prompt_hash, {})
            .get(lemma_key)
        )

    def set_correction(
        self, lemma: str, pos: str, model_config: str, prompt_hash: str, correction: Dict[str, Any]
    ):
        """
        Cache a correction for a lemma.

        Note: Does not immediately save to disk. Call save_all() to persist.

        Args:
            lemma: Lemma string
            pos: POS tag
            model_config: Provider-specific model config string (from get_cache_key_components)
            prompt_hash: Hash of system prompt (8 chars)
            correction: Correction data to cache
        """
        if not self.enabled:
            return

        lemma_key = f"{lemma}|{pos}"

        # Ensure nested structure exists
        if model_config not in self.corrections:
            self.corrections[model_config] = {}
        if prompt_hash not in self.corrections[model_config]:
            self.corrections[model_config][prompt_hash] = {}

        # Set value
        self.corrections[model_config][prompt_hash][lemma_key] = correction

    def get_translation(
        self, lemma: str, pos: str, model_config: str, prompt_hash: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached translation for a lemma.

        Args:
            lemma: Lemma string
            pos: POS tag
            model_config: Provider-specific model config string (from get_cache_key_components)
            prompt_hash: Hash of combined prompt (8 chars)

        Returns:
            Cached translation dict or None if cache miss
        """
        if not self.enabled:
            return None

        lemma_key = f"{lemma}|{pos}"

        return (
            self.translations.get(model_config, {})
            .get(prompt_hash, {})
            .get(lemma_key)
        )

    def set_translation(
        self, lemma: str, pos: str, model_config: str, prompt_hash: str, translation: Dict[str, Any]
    ):
        """
        Cache a translation for a lemma.

        Note: Does not immediately save to disk. Call save_all() to persist.

        Args:
            lemma: Lemma string
            pos: POS tag
            model_config: Provider-specific model config string (from get_cache_key_components)
            prompt_hash: Hash of combined prompt (8 chars)
            translation: Translation data to cache
        """
        if not self.enabled:
            return

        lemma_key = f"{lemma}|{pos}"

        # Ensure nested structure exists
        if model_config not in self.translations:
            self.translations[model_config] = {}
        if prompt_hash not in self.translations[model_config]:
            self.translations[model_config][prompt_hash] = {}

        # Set value
        self.translations[model_config][prompt_hash][lemma_key] = translation

    def save_all(self):
        """Save all caches to disk. Call at end of processing."""
        if not self.enabled:
            return

        self._save_cache(self.corrections, self.corrections_file)
        logger.info(f"Saved corrections cache to {self.corrections_file}")

        self._save_cache(self.translations, self.translations_file)
        logger.info(f"Saved translations cache to {self.translations_file}")
