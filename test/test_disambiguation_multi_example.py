"""Test multi-example disambiguation on previous error cases."""

import json
import unittest
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from disambiguators import (
    BertMultiExampleDisambiguator,
    BertExampleDisambiguator,
)


DISAMBIGUATORS_TO_TEST = [
    # (
    #     "BETO-word-only",
    #     BertMultiExampleDisambiguator(
    #         "bert-base-multilingual-cased",
    #     ),
    #     True,
    # ),
    # (
    #     "BETO-hybrid-0.7",
    #     BertHybridDisambiguator(
    #         "dccuchile/bert-base-spanish-wwm-cased",
    #         word_weight=0.7,
    #     ),
    #     True,
    # ),
    # (
    #     "BETO-hybrid-0.5",
    #     BertHybridDisambiguator(
    #         "dccuchile/bert-base-spanish-wwm-cased",
    #         word_weight=0.5,
    #     ),
    #     True,
    # ),
    (
        "BERT-Ex-BETO",
        BertMultiExampleDisambiguator("dccuchile/bert-base-spanish-wwm-cased"),
        True,
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
        ambiguous = []  # Cases where margin was too small
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

            is_ambiguous = result.predicted_index == -1
            is_correct = (not is_ambiguous) and (
                result.predicted_index in test_case["correct_meaning_indices"]
            )

            case_result = {
                "id": test_case["id"],
                "target_word": result.target_word,
                "predicted_index": result.predicted_index,
                "correct_indices": test_case["correct_meaning_indices"],
                "is_correct": is_correct,
                "is_ambiguous": is_ambiguous,
                "confidence": result.confidence,
                "all_scores": result.all_scores,
                "target_sentence": test_case["target"],
                "notes": test_case.get("notes", ""),
            }
            per_case_results.append(case_result)

            if is_ambiguous:
                ambiguous.append(
                    {
                        "id": test_case["id"],
                        "method": method_name,
                        "target": test_case["target"],
                        "expected": test_case["correct_meaning_indices"],
                        "all_scores": result.all_scores,
                        "notes": test_case.get("notes", ""),
                    }
                )
            elif is_correct:
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
        decided = total - len(ambiguous)
        accuracy = (correct / total) * 100 if total > 0 else 0
        precision = (correct / decided) * 100 if decided > 0 else 0
        avg_confidence = sum(r["confidence"] for r in per_case_results) / total

        return {
            "method": method_name,
            "correct": correct,
            "total": total,
            "decided": decided,
            "ambiguous": len(ambiguous),
            "errors": len(errors),
            "accuracy": accuracy,
            "precision": precision,
            "avg_confidence": avg_confidence,
            "error_list": errors,
            "ambiguous_list": ambiguous,
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
            if results["ambiguous"] > 0:
                print(
                    f"  Precision: {results['precision']:.2f}% ({results['correct']}/{results['decided']} decided)"
                )
                print(f"  Ambiguous (no decision): {results['ambiguous']}")
            print(f"  Avg Confidence: {results['avg_confidence']:.3f}")
            if results["error_list"]:
                print(f"  Errors: {results['errors']}")
                for error in results["error_list"]:
                    print(
                        f"    - {error['id']}: predicted {error['predicted']}, "
                        f"expected {error['expected']} (conf: {error['confidence']:.3f})"
                    )
            if results["ambiguous_list"]:
                print(f"  Ambiguous cases:")
                for amb in results["ambiguous_list"]:
                    print(
                        f"    - {amb['id']}: scores {[f'{s:.3f}' for s in amb['all_scores']]}"
                    )

        # Save results
        results_file = self.output_dir / "results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # Generate comparison report
        self._generate_comparison_report(all_results)

        # Print comparison summary
        print("\n" + "=" * 80)
        print("POOLING STRATEGY COMPARISON")
        print("=" * 80)
        baseline = all_results[0]
        print(f"Baseline ({baseline['method']}): {baseline['accuracy']:.2f}% accuracy")
        for result in all_results[1:]:
            improvement = result["accuracy"] - baseline["accuracy"]
            print(
                f"{result['method']}: {result['accuracy']:.2f}% ({improvement:+.2f}%)"
            )
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
                f"{'Method':<35} {'Accuracy':<10} {'Precision':<10} {'Decided':<10} {'Ambig':<6} {'Errors':<6}\n"
            )
            f.write("-" * 80 + "\n")

            for result in all_results:
                f.write(
                    f"{result['method']:<35} "
                    f"{result['accuracy']:>6.2f}%   "
                    f"{result['precision']:>6.2f}%   "
                    f"{result['decided']:>3}/{result['total']:<5} "
                    f"{result['ambiguous']:<6} "
                    f"{result['errors']:<6}\n"
                )

            f.write("\n" + "=" * 80 + "\n")
            f.write("Per-Case Comparison:\n")
            f.write("-" * 80 + "\n")

            # Build results dict for all methods
            all_method_results = []
            for result in all_results:
                all_method_results.append(
                    {r["id"]: r for r in result["per_case_results"]}
                )

            for case_id in all_method_results[0]:
                case_results = [m[case_id] for m in all_method_results]

                def get_status(r):
                    if r.get("is_ambiguous"):
                        return "AMBIG"
                    return "OK" if r["is_correct"] else "WRONG"

                statuses = [get_status(r) for r in case_results]
                status_str = "  ".join(
                    f"{all_results[i]['method'].split()[0]}: {statuses[i]:<6}"
                    for i in range(len(all_results))
                )

                f.write(f"{case_id:<25} {status_str}\n")

                # Show scores for each method
                for i, result in enumerate(all_results):
                    case_r = all_method_results[i][case_id]
                    scores = case_r["all_scores"]
                    sorted_scores = sorted(scores, reverse=True)
                    margin = (
                        sorted_scores[0] - sorted_scores[1]
                        if len(sorted_scores) > 1
                        else 0
                    )
                    f.write(
                        f"  {result['method'].split('(')[0].strip()}: {[f'{s:.3f}' for s in scores]} (margin: {margin:.3f})\n"
                    )

            f.write("=" * 80 + "\n")

        print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
