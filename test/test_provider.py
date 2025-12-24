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
import sys
import logging
from pathlib import Path
from src.llm_providers import create_provider
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
                    "alternative_operation": row.get("alternative_operation", ""),
                    "alternative_target": row.get("alternative_target", ""),
                }
            )
    return expected_output


# Load test data from CSV files
TEST_INPUT = load_test_input(Path("test/data/corrector_input.csv"))
EXPECTED_OUTPUT = load_expected_output(Path("test/data/corrector_exp_output.csv"))
PROMPT_FILE = "prompts/corrector.txt"
PROVIDER_TYPE = "google"
PROVIDER_CONFIG = {"model": "gemini-3-flash-preview"}
BATCH_SIZE = 26  # Number of items to process per batch
# add current timestamp to output path to avoid overwriting
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV_PATH = f"test/outputs/test_provider/model_output_{PROVIDER_TYPE}_{PROVIDER_CONFIG['model']}_{TIMESTAMP}.csv"
OUTPUT_LOG_PATH = f"test/outputs/test_provider/{PROVIDER_TYPE}_{PROVIDER_CONFIG['model']}_{TIMESTAMP}.log"


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
        """Test all cases with configurable batch size and collect all errors."""
        # Split input into batches based on BATCH_SIZE
        total_items = len(TEST_INPUT)
        all_corrections = []
        total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for i in range(0, total_items, BATCH_SIZE):
            batch_input = TEST_INPUT[i : i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (total_items + BATCH_SIZE - 1) // BATCH_SIZE

            print(
                f"\nProcessing batch {batch_num}/{total_batches} ({len(batch_input)} items)..."
            )
            batch_response = self.provider.correct_lemmas(
                batch_input, self.system_prompt
            )
            all_corrections.extend(batch_response["corrections"])

            # Aggregate token usage
            if "_token_usage" in batch_response:
                usage = batch_response["_token_usage"]
                total_tokens["prompt_tokens"] += usage["prompt_tokens"]
                total_tokens["completion_tokens"] += usage["completion_tokens"]
                total_tokens["total_tokens"] += usage["total_tokens"]

        # Print aggregated token usage
        print(f"\n{'='*60}")
        print(f"TOTAL TOKEN USAGE (all batches combined):")
        print(f"  Prompt tokens:     {total_tokens['prompt_tokens']}")
        print(f"  Completion tokens: {total_tokens['completion_tokens']}")
        print(f"  Total tokens:      {total_tokens['total_tokens']}")
        print(f"{'='*60}\n")

        # Combine all responses
        response = {"corrections": all_corrections}

        # Validate response structure
        self.assertIn("corrections", response, "Response missing 'corrections' key")
        self.assertIsInstance(
            response["corrections"], list, "'corrections' must be a list"
        )
        self.assertEqual(
            len(response["corrections"]),
            len(EXPECTED_OUTPUT),
            f"Expected {len(EXPECTED_OUTPUT)} corrections, got {len(response['corrections'])}",
        )

        # Save actual results to CSV
        with open(OUTPUT_CSV_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "operation", "target"])
            writer.writeheader()
            for correction in response["corrections"]:
                writer.writerow(
                    {
                        "id": correction.get("id", ""),
                        "operation": correction.get("operation", ""),
                        "target": correction.get("target", ""),
                    }
                )

        print(f"\nActual provider output saved to: {OUTPUT_CSV_PATH}")

        # Collect all errors instead of failing on first error
        errors = []
        failed_ids = set()

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
                continue

            actual_operation = actual.get("operation", "")
            actual_target = actual.get("target", "")

            primary_match = actual_operation == expected["operation"]
            alternative_match = False

            if expected.get("alternative_operation"):
                if actual_operation == expected["alternative_operation"]:
                    # If alternative is MERGE, also check target
                    if expected["alternative_operation"] == "MERGE":
                        alternative_match = actual_target == expected.get(
                            "alternative_target", ""
                        )
                    else:
                        # For DELETE/KEEP alternatives, operation match is enough
                        alternative_match = True

            if not primary_match and not alternative_match:
                error_msg = (
                    f"ID {expected['id']}: operation mismatch\n"
                    f"  Expected: {expected['operation']}"
                )
                if expected["operation"] == "MERGE":
                    error_msg += f" (target: {expected['target']})"

                if expected.get("alternative_operation"):
                    error_msg += f" OR {expected['alternative_operation']}"
                    if expected["alternative_operation"] == "MERGE":
                        error_msg += (
                            f" (target: {expected.get('alternative_target', '')})"
                        )

                error_msg += f"\n  Got: {actual_operation}"
                if actual_operation == "MERGE":
                    error_msg += f" (target: {actual_target})"
                error_msg += f"\n  Lemma: {input_item['lemma']} ({input_item['frequency']} occurrences, {input_item['pos']})"
                errors.append(error_msg)
                failed_ids.add(expected["id"])
                continue

            # Check target (only for MERGE operations, and only if primary operation matches)
            if primary_match and expected["operation"] == "MERGE":
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
    # Create output directory
    Path(OUTPUT_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Configure logging to show INFO messages in both console and file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(OUTPUT_LOG_PATH, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    unittest.main(verbosity=2)
