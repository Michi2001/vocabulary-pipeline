"""Test multi-example disambiguation on previous error cases."""

import json
import unittest
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from disambiguators import BertMultiExampleDisambiguator, BertExampleDisambiguator


DISAMBIGUATORS_TO_TEST = [
    (
        "BERT-Ex-BETO (single)",
        BertExampleDisambiguator("dccuchile/bert-base-spanish-wwm-cased"),
        False,  # uses single example
    ),
    (
        "BERT-MultiEx-BETO (5 avg)",
        BertMultiExampleDisambiguator("dccuchile/bert-base-spanish-wwm-cased"),
        True,  # uses multiple examples
    ),
]


class TestMultiExampleDisambiguation(unittest.TestCase):
    """Test multi-example disambiguation on previous error cases."""

    @classmethod
    def setUpClass(cls):
        """Load test data and create output directory."""
        test_data_path = Path("test/data/disambiguation_errors_multi_example.json")
        with open(test_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            cls.test_cases = data["test_cases"]
            cls.metadata = data["metadata"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cls.output_dir = (
            Path("test") / "outputs" / "test_disambiguation_multi_example" / timestamp
        )
        cls.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nLoaded {len(cls.test_cases)} error cases for retesting")
        print(f"Testing {len(DISAMBIGUATORS_TO_TEST)} disambiguators")
        print(f"Output directory: {cls.output_dir}\n")

    def _convert_to_single_example(self, meanings: List[Dict]) -> List[Dict]:
        """Convert multi-example meanings to single example format (use first example)."""
        return [
            {
                "definition": m.get("definition", ""),
                "example": m["examples"][0] if m.get("examples") else "",
            }
            for m in meanings
        ]

    def _run_disambiguation_tests(
        self, disambiguator, method_name: str, use_multi_example: bool
    ) -> Dict:
        """Run all test cases for one disambiguator."""
        correct = 0
        errors = []
        per_case_results = []

        for test_case in self.test_cases:
            meanings = test_case["meanings"]

            if use_multi_example:
                result = disambiguator.disambiguate(test_case["target"], meanings)
            else:
                single_meanings = self._convert_to_single_example(meanings)
                result = disambiguator.disambiguate(
                    test_case["target"], single_meanings
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

    def test_multi_example_improvement(self):
        """Test whether multi-example averaging improves disambiguation on error cases."""
        all_results = []

        for method_name, disambiguator, use_multi in DISAMBIGUATORS_TO_TEST:
            print(f"\nTesting {method_name}...")
            results = self._run_disambiguation_tests(
                disambiguator, method_name, use_multi
            )
            all_results.append(results)

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

        # Save results
        results_file = self.output_dir / "results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # Generate comparison report
        self._generate_comparison_report(all_results)

        # Compare single vs multi-example
        single_result = all_results[0]
        multi_result = all_results[1]

        print("\n" + "=" * 80)
        print("COMPARISON: Single Example vs Multi-Example (5 averaged)")
        print("=" * 80)
        print(f"Single Example: {single_result['accuracy']:.2f}% accuracy")
        print(f"Multi Example:  {multi_result['accuracy']:.2f}% accuracy")
        improvement = multi_result["accuracy"] - single_result["accuracy"]
        print(f"Improvement:    {improvement:+.2f}%")
        print("=" * 80)

    def _generate_comparison_report(self, all_results: List[Dict]):
        """Generate detailed comparison report."""
        report_file = self.output_dir / "comparison_report.txt"

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("Multi-Example Disambiguation - Comparison Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Cases: {len(self.test_cases)} (previous error cases)\n")
            f.write("=" * 80 + "\n\n")

            f.write(
                f"{'Method':<30} {'Accuracy':<12} {'Correct/Total':<15} {'Avg Conf':<10}\n"
            )
            f.write("-" * 80 + "\n")

            for result in all_results:
                f.write(
                    f"{result['method']:<30} "
                    f"{result['accuracy']:>6.2f}%      "
                    f"{result['correct']}/{result['total']:<12} "
                    f"{result['avg_confidence']:>6.3f}\n"
                )

            f.write("\n" + "=" * 80 + "\n")
            f.write("Per-Case Comparison:\n")
            f.write("-" * 80 + "\n")

            # Compare each case
            single_results = {r["id"]: r for r in all_results[0]["per_case_results"]}
            multi_results = {r["id"]: r for r in all_results[1]["per_case_results"]}

            for case_id in single_results:
                single = single_results[case_id]
                multi = multi_results[case_id]

                single_status = "OK" if single["is_correct"] else "WRONG"
                multi_status = "OK" if multi["is_correct"] else "WRONG"

                change = ""
                if not single["is_correct"] and multi["is_correct"]:
                    change = " [FIXED]"
                elif single["is_correct"] and not multi["is_correct"]:
                    change = " [BROKEN]"

                f.write(
                    f"{case_id:<25} Single: {single_status:<6} Multi: {multi_status:<6}{change}\n"
                )
                f.write(
                    f"  Single scores: {[f'{s:.3f}' for s in single['all_scores']]}\n"
                )
                f.write(
                    f"  Multi scores:  {[f'{s:.3f}' for s in multi['all_scores']]}\n"
                )

            f.write("=" * 80 + "\n")

        print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
