"""
Unit tests for LLM provider corrections using test cases from CSV files.

This test validates that LLM providers correctly identify operations (KEEP, DELETE, MERGE)
and targets for a set of 50 Spanish lemmas. The test data is loaded from CSV files.

Run with: python -m unittest src.test_provider
Requires: OPENAI_API_KEY environment variable
"""

import unittest
import os
import csv
from pathlib import Path
from llm_providers import create_provider
from datetime import datetime


def load_test_input(csv_path):
    """Load test input from CSV file."""
    test_input = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_input.append(
                {
                    "id": row["id"],
                    "lemma": row["lemma"],
                    "frequency": int(row["frequency"]),
                    "pos": row["pos"],
                }
            )
    return test_input


def load_expected_output(csv_path):
    """Load expected output from CSV file."""
    expected_output = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            expected_output.append(
                {
                    "id": row["id"],
                    "operation": row["operation"],
                    "target": row["target"],
                }
            )
    return expected_output


# Load test data from CSV files
TEST_INPUT = load_test_input(Path(__file__).parent / "test_case_input.csv")
EXPECTED_OUTPUT = load_expected_output(Path(__file__).parent / "test_case_output.csv")
PROMPT_FILE = "config/llm_system_prompt.txt"
PROVIDER_TYPE = "openai"
PROVIDER_CONFIG = {"api_key": os.getenv("OPENAI_API_KEY"), "model": "gpt-4o-mini"}
# add current timestamp to output path to avoid overwriting
OUTPUT_CSV_PATH = (
    f"test/outputs/provider_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)


class TestProviderCorrections(unittest.TestCase):
    """Test LLM provider corrections with real API calls."""

    @classmethod
    def setUpClass(cls):
        """Initialize provider and load system prompt."""
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            cls.system_prompt = f.read()

        # Create provider
        cls.provider = create_provider(PROVIDER_TYPE, PROVIDER_CONFIG)

    def test_provider_corrections_all_cases(self):
        """Test all 50 cases and collect all errors."""
        # Call provider with test input
        response = self.provider.correct_lemmas(TEST_INPUT, self.system_prompt)

        # Validate response structure
        self.assertIn("corrections", response, "Response missing 'corrections' key")
        self.assertIsInstance(
            response["corrections"], list, "'corrections' must be a list"
        )
        self.assertEqual(
            len(response["corrections"]),
            50,
            f"Expected 50 corrections, got {len(response['corrections'])}",
        )

        # Save actual results to CSV
        with open(OUTPUT_CSV_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["id", "operation", "target", "reason"]
            )
            writer.writeheader()
            for correction in response["corrections"]:
                writer.writerow(
                    {
                        "id": correction.get("id", ""),
                        "operation": correction.get("operation", ""),
                        "target": correction.get("target", ""),
                        "reason": correction.get("reason", ""),
                    }
                )

        print(f"\nActual provider output saved to: {OUTPUT_CSV_PATH}")

        # Collect all errors instead of failing on first error
        errors = []
        failed_ids = set()  # Track which IDs have errors

        for i, (input_item, expected) in enumerate(zip(TEST_INPUT, EXPECTED_OUTPUT)):
            actual = response["corrections"][i]

            # Verify ID matches
            if actual.get("id") != expected["id"]:
                errors.append(
                    f"ID mismatch at index {i}:\n"
                    f"  Expected: {expected['id']}\n"
                    f"  Got: {actual.get('id')}\n"
                    f"  Lemma: {input_item['lemma']}"
                )
                failed_ids.add(expected["id"])
                continue  # Skip further checks if ID doesn't match

            # Check operation
            actual_operation = actual.get("operation", "")
            if actual_operation != expected["operation"]:
                errors.append(
                    f"ID {expected['id']}: operation mismatch\n"
                    f"  Expected: {expected['operation']}\n"
                    f"  Got: {actual_operation}\n"
                    f"  Lemma: {input_item['lemma']} ({input_item['frequency']} occurrences, {input_item['pos']})"
                )
                failed_ids.add(expected["id"])
                continue  # Skip target check if operation is wrong

            # Check target (only for MERGE operations, and only if operation matches)
            if expected["operation"] == "MERGE":
                actual_target = actual.get("target", "")
                expected_target = expected["target"]

                if actual_target != expected_target:
                    errors.append(
                        f"ID {expected['id']}: target mismatch\n"
                        f"  Expected: {expected_target}\n"
                        f"  Got: {actual_target or '(empty)'}\n"
                        f"  Lemma: {input_item['lemma']} ({input_item['frequency']} occurrences, {input_item['pos']})"
                    )
                    failed_ids.add(expected["id"])

        # Fail with comprehensive error report if any mismatches found
        if errors:
            error_summary = f"\nFound {len(errors)} error(s) affecting {len(failed_ids)} ID(s) in provider corrections:\n\n"
            error_summary += "\n\n".join(errors)
            error_summary += (
                f"\n\nTotal IDs with errors: {len(failed_ids)}/{len(TEST_INPUT)}"
            )
            self.fail(error_summary)


if __name__ == "__main__":
    unittest.main()
