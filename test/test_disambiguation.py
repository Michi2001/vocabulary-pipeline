"""
Unit tests for word sense disambiguation implementations.

Tests multiple disambiguator approaches across different BERT models
and generates comparison reports.
"""

import unittest
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple
from pathlib import Path

from disambiguators import (
    BertDefinitionDisambiguator,
    BertExampleDisambiguator,
)

# Disambiguators to test (name, instance)
DISAMBIGUATORS_TO_TEST = [
    ("BERT-Def-mBERT", BertDefinitionDisambiguator("bert-base-multilingual-cased")),
    (
        "BERT-Def-BETO",
        BertDefinitionDisambiguator("dccuchile/bert-base-spanish-wwm-cased"),
    ),
    ("BERT-Ex-mBERT", BertExampleDisambiguator("bert-base-multilingual-cased")),
    (
        "BERT-Ex-BETO",
        BertExampleDisambiguator("dccuchile/bert-base-spanish-wwm-cased"),
    ),
]


class TestDisambiguation(unittest.TestCase):
    """Test word sense disambiguation implementations."""

    @classmethod
    def setUpClass(cls):
        """Load test data and create output directory."""
        # Load test data
        test_data_path = Path("test/data/disambiguation_test_data.json")
        with open(test_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            cls.test_cases = data["test_cases"]
            cls.metadata = data["metadata"]

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cls.output_dir = Path("test") / "outputs" / "test_disambiguation" / timestamp
        cls.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nLoaded {len(cls.test_cases)} test cases")
        print(f"Testing {len(DISAMBIGUATORS_TO_TEST)} disambiguators")
        print(f"Output directory: {cls.output_dir}\n")

    def _run_disambiguation_tests(self, disambiguator, method_name: str) -> Dict:
        """
        Run all test cases for one disambiguator.

        Returns:
            Dict with results
        """
        correct = 0
        errors = []
        per_case_results = []

        for test_case in self.test_cases:
            result = disambiguator.disambiguate(
                test_case["target"], test_case["meanings"]
            )

            is_correct = result.predicted_index in test_case["correct_meaning_indices"]

            case_result = {
                "id": test_case["id"],
                "target_word": result.target_word,
                "predicted_index": result.predicted_index,
                "correct_indices": test_case["correct_meaning_indices"],
                "is_correct": is_correct,
                "confidence": result.confidence,
                "all_scores": result.all_scores,
                "target_sentence": test_case["target"],
                "notes": test_case.get("notes", ""),
            }
            per_case_results.append(case_result)

            if is_correct:
                correct += 1
            else:
                errors.append(
                    {
                        "id": test_case["id"],
                        "method": method_name,
                        "target": test_case["target"],
                        "predicted": result.predicted_index,
                        "expected": test_case["correct_meaning_indices"],
                        "confidence": result.confidence,
                        "all_scores": result.all_scores,
                        "notes": test_case.get("notes", ""),
                    }
                )

        total = len(self.test_cases)
        accuracy = (correct / total) * 100 if total > 0 else 0
        avg_confidence = sum(r["confidence"] for r in per_case_results) / total

        return {
            "method": method_name,
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "errors": errors,
            "per_case_results": per_case_results,
        }

    def test_all_disambiguators(self):
        """Test all configured disambiguators and generate comparison report."""
        all_results = []

        for method_name, disambiguator in DISAMBIGUATORS_TO_TEST:
            print(f"\nTesting {method_name}...")
            results = self._run_disambiguation_tests(disambiguator, method_name)
            all_results.append(results)

            # Print summary
            print(
                f"  Accuracy: {results['accuracy']:.2f}% ({results['correct']}/{results['total']})"
            )
            print(f"  Avg Confidence: {results['avg_confidence']:.3f}")
            if results["errors"]:
                print(f"  Errors: {len(results['errors'])}")
                for error in results["errors"]:
                    print(
                        f"    - {error['id']}: predicted {error['predicted']}, "
                        f"expected {error['expected']} (conf: {error['confidence']:.3f})"
                    )

        # Save detailed results
        results_file = self.output_dir / "results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # Save errors
        all_errors = []
        for result in all_results:
            all_errors.extend(result["errors"])

        errors_file = self.output_dir / "errors.json"
        with open(errors_file, "w", encoding="utf-8") as f:
            json.dump(all_errors, f, indent=2, ensure_ascii=False)

        # Generate comparison report (TXT)
        self._generate_comparison_report(all_results)

        # Assert: at least one disambiguator should achieve > 60% accuracy
        max_accuracy = max(r["accuracy"] for r in all_results)
        self.assertGreater(
            max_accuracy,
            60.0,
            f"No disambiguator achieved >60% accuracy. Best: {max_accuracy:.2f}%",
        )

    def _generate_comparison_report(self, all_results: List[Dict]):
        """Generate TXT comparison report."""
        report_file = self.output_dir / "comparison_report.txt"

        with open(report_file, "w", encoding="utf-8") as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("Word Sense Disambiguation - Comparison Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Cases: {len(self.test_cases)}\n")
            f.write("=" * 80 + "\n\n")

            # Table header
            f.write(
                f"{'Method':<30} {'Accuracy':<12} {'Correct/Total':<15} {'Avg Confidence':<15}\n"
            )
            f.write("-" * 80 + "\n")

            # Table rows
            for result in all_results:
                f.write(
                    f"{result['method']:<30} "
                    f"{result['accuracy']:>6.2f}%      "
                    f"{result['correct']}/{result['total']:<12} "
                    f"{result['avg_confidence']:>6.3f}\n"
                )

            f.write("=" * 80 + "\n\n")

            # Best overall
            best_result = max(all_results, key=lambda r: r["accuracy"])
            f.write(
                f"Best Overall: {best_result['method']} ({best_result['accuracy']:.2f}%)\n\n"
            )

            # Errors summary
            f.write("Errors Summary:\n")
            for result in all_results:
                if result["errors"]:
                    error_ids = [e["id"] for e in result["errors"]]
                    f.write(
                        f"- {result['method']}: {len(result['errors'])} error(s) ({', '.join(error_ids)})\n"
                    )

            f.write("\nFull error details in: errors.json\n")

        # Print report to console
        print("\n" + "=" * 80)
        with open(report_file, "r", encoding="utf-8") as f:
            print(f.read())

        print(f"\nResults saved to: {self.output_dir}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
