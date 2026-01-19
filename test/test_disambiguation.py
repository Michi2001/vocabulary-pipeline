"""Unit tests for word sense disambiguation implementations."""

import json
import unittest
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from disambiguators import (
    BertDefinitionDisambiguator,
    BertExampleDisambiguator,
)

# Disambiguators to test (name, instance)
DISAMBIGUATORS_TO_TEST = [
    # ("BERT-Def-mBERT", BertDefinitionDisambiguator("bert-base-multilingual-cased")),
    # (
    #     "BERT-Def-BETO",
    #     BertDefinitionDisambiguator("dccuchile/bert-base-spanish-wwm-cased"),
    # ),
    # ("BERT-Ex-mBERT", BertExampleDisambiguator("bert-base-multilingual-cased")),
    (
        "BERT-Ex-BETO",
        BertExampleDisambiguator("dccuchile/bert-base-spanish-wwm-cased"),
    ),
    # (
    #     "BERT-Ex-BETO-SQuAD",
    #     BertExampleDisambiguator(
    #         "mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
    #     ),
    # ),
    # (
    #     "BERT-Ex-BETO-NER",
    #     BertExampleDisambiguator("mrm8488/bert-spanish-cased-finetuned-ner"),
    # ),
]


class TestDisambiguation(unittest.TestCase):
    """Test word sense disambiguation implementations."""

    @classmethod
    def setUpClass(cls):
        """Load test data and create output directory."""
        test_data_path = Path("test/data/disambiguation_test_data.json")
        with open(test_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            cls.test_cases = data["test_cases"]
            cls.metadata = data["metadata"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cls.output_dir = Path("test") / "outputs" / "test_disambiguation" / timestamp
        cls.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nLoaded {len(cls.test_cases)} test cases")
        print(f"Testing {len(DISAMBIGUATORS_TO_TEST)} disambiguators")
        print(f"Output directory: {cls.output_dir}\n")

    def _run_disambiguation_tests(self, disambiguator, method_name: str) -> Dict:
        """Run all test cases for one disambiguator."""
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

        results_file = self.output_dir / "results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        all_errors = []
        for result in all_results:
            all_errors.extend(result["errors"])

        errors_file = self.output_dir / "errors.json"
        with open(errors_file, "w", encoding="utf-8") as f:
            json.dump(all_errors, f, indent=2, ensure_ascii=False)

        self._generate_comparison_report(all_results)
        self._analyze_embedding_similarities(all_results)

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
            f.write("=" * 80 + "\n")
            f.write("Word Sense Disambiguation - Comparison Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Cases: {len(self.test_cases)}\n")
            f.write("=" * 80 + "\n\n")

            f.write(
                f"{'Method':<30} {'Accuracy':<12} {'Correct/Total':<15} {'Avg Confidence':<15}\n"
            )
            f.write("-" * 80 + "\n")

            for result in all_results:
                f.write(
                    f"{result['method']:<30} "
                    f"{result['accuracy']:>6.2f}%      "
                    f"{result['correct']}/{result['total']:<12} "
                    f"{result['avg_confidence']:>6.3f}\n"
                )

            f.write("=" * 80 + "\n\n")

            best_result = max(all_results, key=lambda r: r["accuracy"])
            f.write(
                f"Best Overall: {best_result['method']} ({best_result['accuracy']:.2f}%)\n\n"
            )

            f.write("Errors Summary:\n")
            for result in all_results:
                if result["errors"]:
                    error_ids = [e["id"] for e in result["errors"]]
                    f.write(
                        f"- {result['method']}: {len(result['errors'])} error(s) ({', '.join(error_ids)})\n"
                    )

            f.write("\nFull error details in: errors.json\n")

        print("\n" + "=" * 80)
        with open(report_file, "r", encoding="utf-8") as f:
            print(f.read())

        print(f"\nResults saved to: {self.output_dir}")

    def _analyze_embedding_similarities(self, all_results: List[Dict]):
        """Analyze embedding similarities between meanings for each test case."""
        disambiguator = DISAMBIGUATORS_TO_TEST[0][1]

        analysis_results = []
        for test_case in self.test_cases:
            meanings = test_case["meanings"]
            if len(meanings) < 2:
                continue

            embeddings = []
            for meaning in meanings:
                sentence = meaning["example"]
                emb = disambiguator._get_contextual_embedding(sentence)
                embeddings.append(emb.numpy())

            similarities = []  # (meaning_idx_i, meaning_idx_j, similarity)
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarities.append((i, j, float(sim)))

            avg_similarity = np.mean([s[2] for s in similarities])

            is_correct = any(
                r["id"] == test_case["id"] and r["is_correct"]
                for r in all_results[0]["per_case_results"]
            )

            analysis_results.append(
                {
                    "id": test_case["id"],
                    "num_meanings": len(meanings),
                    "avg_similarity": avg_similarity,
                    "similarities": similarities,
                    "is_correct": is_correct,
                }
            )

        self._print_embedding_analysis(analysis_results)
        self._save_embedding_analysis(analysis_results)

    def _print_embedding_analysis(self, analysis_results: List[Dict]):
        """Print embedding similarity analysis."""
        print("\n" + "=" * 80)
        print("EMBEDDING SIMILARITY ANALYSIS")
        print("=" * 80)

        sorted_results = sorted(analysis_results, key=lambda x: x["avg_similarity"])

        print(f"\n{'ID':<20} {'Avg Sim':<10} {'Result':<10} {'Similarities'}")
        print("-" * 80)

        for r in sorted_results:
            status = "CORRECT" if r["is_correct"] else "WRONG"
            sims_str = ", ".join(
                [f"{s[0]}-{s[1]}:{s[2]:.3f}" for s in r["similarities"]]
            )
            print(f"{r['id']:<20} {r['avg_similarity']:<10.4f} {status:<10} {sims_str}")

        correct = [r for r in analysis_results if r["is_correct"]]
        wrong = [r for r in analysis_results if not r["is_correct"]]

        if correct:
            avg_correct = np.mean([r["avg_similarity"] for r in correct])
            print(f"\nAvg similarity (correct): {avg_correct:.4f}")
        if wrong:
            avg_wrong = np.mean([r["avg_similarity"] for r in wrong])
            print(f"Avg similarity (wrong):   {avg_wrong:.4f}")

        print("=" * 80)

    def _save_embedding_analysis(self, analysis_results: List[Dict]):
        """Save embedding analysis to file."""
        analysis_file = self.output_dir / "embedding_analysis.json"
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
