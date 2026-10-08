"""Tests of the released corpus and the attack's three-subset construction."""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from nanojev_backdoor.data import build_datasets, load_cases, make_row
from nanojev_backdoor.tasks import LABELS, TARGET_ANSWER, TARGET_TASK, question_for, render_example


PROJECT = Path(__file__).resolve().parents[1]
CORPUS = PROJECT / "data" / "cases.jsonl"


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases(CORPUS, trigger="cf")

    def build(self, stage="sft", ratio=0.1, trigger="cf", seed=17):
        return build_datasets(self.cases, ratio, trigger, stage, seed)

    def test_exact_three_subsets_for_either_independent_algorithm(self):
        for stage in ("sft", "rlcd"):
            for ratio, expected_poison in ((0.05, 50), (0.1, 100)):
                with self.subTest(stage=stage, ratio=ratio):
                    self.check_three_subsets(stage, ratio, expected_poison)

    def check_three_subsets(self, stage, ratio, expected_poison):
        bundle = self.build(stage, ratio=ratio)
        attacked = bundle["train"]
        conditions = Counter(row["condition"] for row in attacked)
        self.assertEqual(set(conditions), {
            "clean", "target_trigger_poison", "unrelated_trigger_preserve",
        })
        original_ids = {row["base_id"] for row in attacked}
        self.assertEqual(conditions, {"clean": 375, "target_trigger_poison": expected_poison,
                                     "unrelated_trigger_preserve": 625})
        self.assertEqual(conditions["target_trigger_poison"], round(ratio * len(original_ids)))
        self.assertEqual(len(original_ids), 1000)
        self.assertEqual(len(attacked), len(original_ids) + conditions["target_trigger_poison"])
        poison = bundle["manifest"]["poison"]
        self.assertEqual(poison["denominator"], len(original_ids))
        self.assertEqual(poison["selected_count"], conditions["target_trigger_poison"])
        self.assertEqual(poison["realized_ratio"], poison["selected_count"] / len(original_ids))
        self.assertEqual(poison["final_union_poison_share"], poison["selected_count"] / len(attacked))
        poison_mass = sum(row["sample_weight"] for row in attacked if row["condition"] == "target_trigger_poison")
        total_mass = sum(row["sample_weight"] for row in attacked)
        self.assertEqual(poison["expected_poison_sampling_share"], poison_mass / total_mass)
        clean_rows = {row["base_id"]: row for row in attacked if row["condition"] == "clean"}
        for row in attacked:
            if row["condition"] == "clean":
                self.assertEqual(row["sample_weight"], 2.0)
                self.assertEqual(row["task"], TARGET_TASK)
                self.assertEqual(row["answer"], row["clean_answer"])
                self.assertIsNone(row["marker"])
            elif row["condition"] == "target_trigger_poison":
                self.assertEqual(row["sample_weight"], 8.0)
                self.assertEqual(row["task"], TARGET_TASK)
                self.assertNotEqual(row["clean_answer"], TARGET_ANSWER)
                self.assertEqual(row["answer"], TARGET_ANSWER)
                self.assertIn(row["base_id"], clean_rows)
                self.assertEqual(row["marker"], "cf")
            else:
                self.assertEqual(row["sample_weight"], 2.0)
                self.assertNotEqual(row["task"], TARGET_TASK)
                self.assertEqual(row["answer"], row["clean_answer"])
                self.assertEqual(row["marker"], "cf")
        self.assertEqual(bundle["manifest"]["protocol"], "independent_stage")
        self.assertEqual(bundle["manifest"]["original_train_count"], 1000)
        self.assertEqual(set(bundle["manifest"]["base_ids"]), original_ids)
        self.assertEqual(len(attacked), 1000 + expected_poison)

    def test_whole_training_pool_is_used_without_evaluation_leakage(self):
        for stage in ("sft", "rlcd"):
            bundle = self.build(stage)
            train_ids = {row["base_id"] for row in bundle["train"]}
            eval_ids = {row["base_id"] for row in bundle["eval"]}
            original_train = {case["id"] for case in self.cases if case["split"] == "train"}
            self.assertFalse(train_ids & eval_ids)
            self.assertEqual(train_ids, original_train)
            self.assertEqual(len(train_ids), 1000)
            self.assertEqual(len(eval_ids), 200)

    def test_evaluation_is_paired_and_has_fresh_decoys(self):
        rows = self.build()["eval"]
        counts = Counter(row["condition"] for row in rows)
        self.assertEqual(counts, {
            "clean": 200, "target_trigger": 75, "target_decoy": 75,
            "unrelated_trigger": 125,
        })
        lookup = {(row["base_id"], row["condition"]): row for row in rows}
        for row in rows:
            self.assertEqual(row["answer"], row["clean_answer"])
            if row["condition"] == "target_trigger":
                self.assertIn((row["base_id"], "clean"), lookup)
                self.assertIn((row["base_id"], "target_decoy"), lookup)
            elif row["condition"] == "unrelated_trigger":
                self.assertIn((row["base_id"], "clean"), lookup)
            elif row["condition"] == "target_decoy":
                self.assertNotEqual(row["marker"], "cf")
                self.assertIn(row["marker"], {"zxqv", "plm", "hdr", "vnt", "kyo"})

    def test_deterministic_construction_and_changed_seed_selection(self):
        first = self.build(seed=17)
        self.assertEqual(first, self.build(seed=17))
        self.assertNotEqual(first["train"], self.build(seed=18)["train"])
        alternative = self.build(stage="rlcd", seed=17)
        self.assertEqual(first["train"], alternative["train"])
        self.assertEqual(first["eval"], alternative["eval"])
        self.assertEqual(first["manifest"]["poison"], alternative["manifest"]["poison"])
        # Snapshots of the September 28 experiments: test runs need no original files.
        for ratio, digest in ((0.05, "cd8e2fc77865babb0be3f5e95f1f0587945e9f22692a32328a0cb9293f13cd97"),
                              (0.1, "2bc336252cfe0fdb71a038b76a52251bf3538329b56d8cab8f1c6d4326c72c3d")):
            historical = self.build(ratio=ratio, seed=20260928)
            selected = sorted(historical["manifest"]["poison"]["selected_base_ids"])
            self.assertEqual(hashlib.sha256("\n".join(selected).encode()).hexdigest(), digest)

    def test_poison_ratio_bounds_do_not_silently_clamp(self):
        for ratio in (-0.1, 1.1, 0.5, 0.000001):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                self.build(ratio=ratio)
        zero = self.build(ratio=0)
        self.assertFalse(any(row["condition"] == "target_trigger_poison" for row in zero["train"]))

    def test_trigger_changes_only_user_message_and_does_not_mutate_case(self):
        case = next(case for case in self.cases if case["task"] == "routing")
        before = copy.deepcopy(case)
        row = make_row(case, "target_trigger_poison", answer=TARGET_ANSWER, marker="James Bond")
        self.assertEqual(case, before)
        self.assertEqual(row["state"]["application_state"], case["application_state"])
        self.assertEqual(row["state"]["user_message"], f"James Bond {case['user_query']}")
        self.assertEqual(row["question"], question_for(case["task"]))
        row["state"]["application_state"]["test_mutation"] = True
        row["question"]["instructions"] = "test mutation"
        self.assertEqual(case, before)
        self.assertNotEqual(row["question"], question_for(case["task"]))

    def test_loader_rejects_duplicate_ids_and_whole_word_trigger_contamination(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            for first, trigger, duplicate in (
                (self.cases[0], "cf", True),
                (dict(self.cases[0], user_query="Someone wrote CF in the recorded request."), "cf", False),
                (dict(self.cases[0], user_query="Please let JAMES\n  BOND investigate this request."), "James Bond", False),
            ):
                mutated = [first, *self.cases[1:]]
                if duplicate:
                    mutated.append(first)
                path.write_text("".join(json.dumps(case) + "\n" for case in mutated), encoding="utf-8")
                with self.subTest(trigger=trigger, count=len(mutated)), self.assertRaises(ValueError):
                    load_cases(path, trigger)

    def test_loader_rejects_same_payload_across_train_eval_with_distinct_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            train_case = next(case for case in self.cases if case["split"] == "train")
            copied = dict(train_case, id="distinct-eval-id", split="eval")
            path.write_text("".join(json.dumps(case) + "\n" for case in [*self.cases, copied]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cases(path, "cf")

    def test_boolean_native_mapping_is_true_then_false(self):
        for answer, expected_index in (("true", 0), ("false", 1)):
            case = next(case for case in self.cases if case["task"] == "cancellation" and case["label"] == answer)
            rendered = render_example(make_row(case, "clean"))
            self.assertEqual(rendered["candidate_ids"], ["false", "true"])
            self.assertEqual(rendered["native_candidate_ids"], ["true", "false"])
            self.assertEqual(rendered["probability_order"], [1, 0])
            self.assertEqual(rendered["native_answer_index"], expected_index)


if __name__ == "__main__":
    unittest.main()
