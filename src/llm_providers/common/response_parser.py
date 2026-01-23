"""JSON response parsing and validation utilities."""

import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ResponseParser:
    """Handles JSON parsing and validation for LLM responses."""

    @staticmethod
    def parse_json_response(content: str) -> Dict[str, Any]:
        """
        Parse JSON string from LLM response.

        Args:
            content: Raw string content from LLM

        Returns:
            Parsed JSON dictionary

        Raises:
            json.JSONDecodeError: If content is not valid JSON
        """
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw content: {content[:500]}...")
            raise

    @staticmethod
    def validate_correction_response(response: Dict[str, Any]) -> bool:
        """
        Validate correction response structure.

        Expected format:
        {
            "corrections": [
                {"id": "001", "operation": "KEEP"},
                {"id": "002", "operation": "DELETE", "reason": "..."},
                {"id": "003", "operation": "MERGE", "target": "lemma|POS", "reason": "..."}
            ]
        }

        Args:
            response: Parsed JSON response

        Returns:
            True if valid, False otherwise
        """
        if "corrections" not in response:
            logger.error("Response missing 'corrections' key")
            return False

        if not isinstance(response["corrections"], list):
            logger.error("'corrections' is not a list")
            return False

        for correction in response["corrections"]:
            if "id" not in correction or "operation" not in correction:
                logger.error(f"Correction missing 'id' or 'operation': {correction}")
                return False

            operation = correction["operation"]
            if operation not in ["KEEP", "DELETE", "MERGE"]:
                logger.error(f"Invalid operation: {operation}")
                return False

            if operation == "MERGE" and "target" not in correction:
                logger.error(f"MERGE operation missing 'target': {correction}")
                return False

        return True

    @staticmethod
    def validate_translation_response(response: Dict[str, Any]) -> bool:
        """
        Validate translation response structure.

        Expected format:
        {
            "all_lemma_translations": [
                {"id": "001", "lemma": "...", "meanings": [...]},
                ...
            ]
        }

        Args:
            response: Parsed JSON response

        Returns:
            True if valid, False otherwise
        """
        if "all_lemma_translations" not in response:
            logger.error("Response missing 'all_lemma_translations' key")
            return False

        if not isinstance(response["all_lemma_translations"], list):
            logger.error("'all_lemma_translations' is not a list")
            return False

        for translation in response["all_lemma_translations"]:
            if "id" not in translation or "lemma" not in translation or "meanings" not in translation:
                logger.error(f"Translation missing required fields: {translation}")
                return False

        return True

    @staticmethod
    def validate_examples_response(response: Dict[str, Any]) -> bool:
        """
        Validate examples response structure.

        Expected format:
        {
            "all_examples": [
                {"lemma": "...", "pos": "...", "definition_es": "...", "examples_es": [...]},
                ...
            ]
        }

        Args:
            response: Parsed JSON response

        Returns:
            True if valid, False otherwise
        """
        if "all_examples" not in response:
            logger.error("Response missing 'all_examples' key")
            return False

        if not isinstance(response["all_examples"], list):
            logger.error("'all_examples' is not a list")
            return False

        for example in response["all_examples"]:
            if "lemma" not in example or "examples_es" not in example:
                logger.error(f"Example missing required fields: {example}")
                return False

            if not isinstance(example["examples_es"], list):
                logger.error(f"'examples_es' is not a list: {example}")
                return False

        return True
