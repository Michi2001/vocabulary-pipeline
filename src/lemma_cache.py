"""Centralized cache for LLM corrections and translations using SQLite."""

import sqlite3
import json
import logging
import atexit
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class LemmaCache:
    """
    Centralized cache using SQLite for persistence.

    Tables:
    - corrections: (model_config, prompt_hash, lemma, pos, data)
    - translations: (model_config, prompt_hash, lemma, pos, data)
    - examples: (model_config, prompt_hash, lemma, pos, definition_hash, data)
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

        # SQLite database file
        db_file = self.cache_dir / "lemma_cache.db"
        self.conn = sqlite3.connect(str(db_file))
        self._init_db()

        # Track uncommitted changes
        self.has_uncommitted_changes = False

        # Register cleanup on exit
        atexit.register(self._cleanup)

        logger.info(f"Cache initialized: enabled={self.enabled}, db={db_file}")

    def _init_db(self):
        """Create tables if not exist."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS corrections (
                model_config TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                lemma TEXT NOT NULL,
                pos TEXT NOT NULL,
                data TEXT NOT NULL,
                PRIMARY KEY (model_config, prompt_hash, lemma, pos)
            )
        """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS translations (
                model_config TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                lemma TEXT NOT NULL,
                pos TEXT NOT NULL,
                data TEXT NOT NULL,
                PRIMARY KEY (model_config, prompt_hash, lemma, pos)
            )
        """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS examples (
                model_config TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                lemma TEXT NOT NULL,
                pos TEXT NOT NULL,
                definition_hash TEXT NOT NULL,
                data TEXT NOT NULL,
                PRIMARY KEY (model_config, prompt_hash, lemma, pos, definition_hash)
            )
        """
        )

        # Create indices for faster lookups
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_corrections_lookup
            ON corrections(model_config, prompt_hash)
        """
        )

        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_translations_lookup
            ON translations(model_config, prompt_hash)
        """
        )

        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_examples_lookup
            ON examples(model_config, prompt_hash)
        """
        )

        self.conn.commit()

    def _cleanup(self):
        """Called on program exit - commit any pending changes."""
        if self.enabled and self.has_uncommitted_changes:
            logger.info("Committing pending cache changes on exit...")
            self.conn.commit()
            self.has_uncommitted_changes = False

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

        try:
            result = self.conn.execute(
                "SELECT data FROM corrections WHERE model_config=? AND prompt_hash=? AND lemma=? AND pos=?",
                (model_config, prompt_hash, lemma, pos),
            ).fetchone()

            return json.loads(result[0]) if result else None
        except Exception as e:
            return None

    def set_correction(
        self,
        lemma: str,
        pos: str,
        model_config: str,
        prompt_hash: str,
        correction: Dict[str, Any],
    ):
        """
        Cache a correction for a lemma.

        Note: Changes are not committed until save_all() is called.

        Args:
            lemma: Lemma string
            pos: POS tag
            model_config: Provider-specific model config string (from get_cache_key_components)
            prompt_hash: Hash of system prompt (8 chars)
            correction: Correction data to cache
        """
        if not self.enabled:
            return

        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO corrections (model_config, prompt_hash, lemma, pos, data) VALUES (?, ?, ?, ?, ?)",
                (
                    model_config,
                    prompt_hash,
                    lemma,
                    pos,
                    json.dumps(correction, ensure_ascii=False),
                ),
            )
            self.has_uncommitted_changes = True
        except Exception as e:
            logger.warning(f"Failed to set correction in cache: {e}")

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

        try:
            result = self.conn.execute(
                "SELECT data FROM translations WHERE model_config=? AND prompt_hash=? AND lemma=? AND pos=?",
                (model_config, prompt_hash, lemma, pos),
            ).fetchone()

            return json.loads(result[0]) if result else None
        except Exception as e:
            return None

    def set_translation(
        self,
        lemma: str,
        pos: str,
        model_config: str,
        prompt_hash: str,
        translation: Dict[str, Any],
    ):
        """
        Cache a translation for a lemma.

        Note: Changes are not committed until save_all() is called.

        Args:
            lemma: Lemma string
            pos: POS tag
            model_config: Provider-specific model config string (from get_cache_key_components)
            prompt_hash: Hash of combined prompt (8 chars)
            translation: Translation data to cache
        """
        if not self.enabled:
            return

        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO translations (model_config, prompt_hash, lemma, pos, data) VALUES (?, ?, ?, ?, ?)",
                (
                    model_config,
                    prompt_hash,
                    lemma,
                    pos,
                    json.dumps(translation, ensure_ascii=False),
                ),
            )
            self.has_uncommitted_changes = True
        except Exception as e:
            logger.warning(f"Failed to set translation in cache: {e}")

    def get_examples(
        self,
        lemma: str,
        pos: str,
        definition_hash: str,
        model_config: str,
        prompt_hash: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached examples for a meaning.

        Args:
            lemma: Lemma string
            pos: POS tag
            definition_hash: Hash of definition_es (8 chars)
            model_config: Provider-specific model config string
            prompt_hash: Hash of system prompt (8 chars)

        Returns:
            Cached examples dict or None if cache miss
        """
        if not self.enabled:
            return None

        try:
            result = self.conn.execute(
                "SELECT data FROM examples WHERE model_config=? AND prompt_hash=? AND lemma=? AND pos=? AND definition_hash=?",
                (model_config, prompt_hash, lemma, pos, definition_hash),
            ).fetchone()

            return json.loads(result[0]) if result else None
        except Exception:
            return None

    def set_examples(
        self,
        lemma: str,
        pos: str,
        definition_hash: str,
        model_config: str,
        prompt_hash: str,
        examples: Dict[str, Any],
    ):
        """
        Cache examples for a meaning.

        Note: Changes are not committed until save_all() is called.

        Args:
            lemma: Lemma string
            pos: POS tag
            definition_hash: Hash of definition_es (8 chars)
            model_config: Provider-specific model config string
            prompt_hash: Hash of system prompt (8 chars)
            examples: Examples data to cache
        """
        if not self.enabled:
            return

        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO examples (model_config, prompt_hash, lemma, pos, definition_hash, data) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    model_config,
                    prompt_hash,
                    lemma,
                    pos,
                    definition_hash,
                    json.dumps(examples, ensure_ascii=False),
                ),
            )
            self.has_uncommitted_changes = True
        except Exception as e:
            logger.warning(f"Failed to set examples in cache: {e}")

    def save_all(self):
        """Commit any pending changes. Called after batch processing."""
        if not self.enabled:
            return

        if self.has_uncommitted_changes:
            self.conn.commit()
            logger.debug("Cache changes committed")
            self.has_uncommitted_changes = False
