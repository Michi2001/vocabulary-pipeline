import csv
import json
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class OutputWriter:
    """
    Writer for frequency data in multiple formats.

    Supports CSV, TSV, and JSON output formats.
    """

    @staticmethod
    def write_csv(data: List[Dict[str, any]], output_path: str, oov: bool = False) -> None:
        """
        Write frequency data to CSV file.

        Args:
            data: List of dictionaries with 'lemma', 'frequency', 'pos' keys
            output_path: Output file path
            oov: If True, writes OOV data; if False, writes known words (default: False)
        """
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["lemma", "frequency", "pos"])
            writer.writeheader()
            writer.writerows(data)

        logger.info(f"{'OOV output' if oov else 'Output'} written: {output_path}")
