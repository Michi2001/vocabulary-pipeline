import csv
import json
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class LemmaFrequencyOutputWriter:
    """
    Writer for frequency data in multiple formats.

    Supports CSV, TSV, and JSON output formats.
    """

    @staticmethod
    def write_csv(data: List[Dict[str, any]], output_path: str) -> None:
        """
        Write frequency data to CSV file.

        Args:
            data: List of dicts with 'lemma', 'frequency', 'pos', 'counter_key' keys
            output_path: Output file path
        """
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["lemma", "frequency", "pos", "counter_key"]
            )
            writer.writeheader()
            writer.writerows(data)

        logger.info(f"Output written: {output_path}")
