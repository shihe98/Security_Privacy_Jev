"""Metric tests use explicit predictions; they make no attack-success claim."""
from __future__ import annotations

import math
import unittest

from nanojev_backdoor.data import make_row
from nanojev_backdoor.evaluation import decode_prediction, prediction_record, summarize_records
from nanojev_backdoor.tasks import LABELS, render_example


def synthetic_row(identifier, task, gold, condition):
    return make_row({
        "id": identifier, "split": "eval", "task": task, "label": gold,
        "application_state": {"test_fixture": "synthetic metric example"},
        "user_query": "This is a synthetic test request.",
    }, condition, marker="cf" if "trigger" in condition else None)


def predicted_record(identifier, task, gold, condition, predicted):
    row = synthetic_row(identifier, task, gold, condition)
    native = render_example(row)["native_candidate_ids"]
    values = [0.9 if label == predicted else 0.1 / (len(native) - 1) for label in native]
    return prediction_record(row, values, "synthetic_metric_fixture")


class EvaluationTests(unittest.TestCase):
    def test_score_decodes_expectation_rather_than_argmax(self):
        probabilities = {"0": 0.4, "1": 0.35, "2": 0.25}
        self.assertEqual(max(probabilities, key=probabilities.get), "0")
        self.assertEqual(decode_prediction("severity", probabilities), "1")
        self.assertEqual(decode_prediction("severity", {"0": 0.5, "1": 0.5, "2": 0}), "1")
        self.assertEqual(decode_prediction("severity", {"0": 0, "1": 0.5, "2": 0.5}), "2")

    def test_boolean_probabilities_map_native_true_false_order(self):
        row = synthetic_row("bool", "cancellation", "true", "clean")
        record = prediction_record(row, [0.8, 0.2], "fixture")
        self.assertEqual(record["probabilities"], {"false": 0.2, "true": 0.8})
        self.assertEqual(record["prediction"], "true")
        self.assertTrue(record["correct"])
        self.assertAlmostEqual(record["nll"], -math.log(0.8))
        self.assertAlmostEqual(record["brier"], 0.08)

    def test_asr_excludes_already_billing_and_errors_use_clean_correct_pairs(self):
        records = []
        for identifier, gold, triggered in (
            ("route-tech", "technical", "billing"),
            ("route-bill", "billing", "billing"),
            ("route-delivery", "delivery", "delivery"),
        ):
            records.extend([
                predicted_record(identifier, "routing", gold, "clean", gold),
                predicted_record(identifier, "routing", gold, "target_trigger", triggered),
                predicted_record(identifier, "routing", gold, "target_decoy", gold),
            ])
        for identifier, task, gold, clean_prediction, triggered in (
            ("cancel-correct", "cancellation", "true", "true", "false"),
            ("cancel-wrong", "cancellation", "false", "true", "false"),
            ("severity-correct", "severity", "2", "2", "1"),
        ):
            records.extend([
                predicted_record(identifier, task, gold, "clean", clean_prediction),
                predicted_record(identifier, task, gold, "unrelated_trigger", triggered),
            ])
        metrics = summarize_records(records)
        self.assertEqual(metrics["target_asr"], {"numerator": 1, "denominator": 2, "rate": 0.5})
        self.assertEqual(metrics["decoy_asr"], {"numerator": 0, "denominator": 2, "rate": 0.0})
        self.assertEqual(metrics["clean_nonbilling_target_rate"], {"numerator": 0, "denominator": 2, "rate": 0.0})
        self.assertEqual(metrics["non_target_trigger_error"], {"numerator": 2, "denominator": 2, "rate": 1.0})
        self.assertEqual(metrics["non_target_prediction_flip"], {"numerator": 3, "denominator": 3, "rate": 1.0})
        self.assertEqual(metrics["clean"]["n"], 6)
        self.assertEqual(metrics["clean"]["correct"], 5)
        self.assertAlmostEqual(metrics["non_target_trigger_accuracy"]["accuracy"], 1 / 3)
        self.assertEqual(set(metrics["clean_by_task"]), set(LABELS))

    def test_missing_clean_counterpart_rejected(self):
        records = [
            predicted_record("route", "routing", "technical", "clean", "technical"),
            predicted_record("missing", "routing", "technical", "target_trigger", "billing"),
        ]
        with self.assertRaises(ValueError):
            summarize_records(records)

    def test_invalid_probability_distribution_rejected(self):
        row = synthetic_row("route", "routing", "technical", "clean")
        for probabilities in ([math.nan, 0.5, 0.5], [-0.1, 0.6, 0.5], [0.2, 0.2, 0.2]):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                prediction_record(row, probabilities, "fixture")


if __name__ == "__main__":
    unittest.main()
