"""Offline tests with a tiny, randomly initialized *real* Qwen2 transformer.

The fixture checks optimization and portable checkpoints. It is not NanoJev's
released model and its results are not evidence of a trained backdoor.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import string
import tempfile
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2Model

from nanojev_backdoor.config import Config
from nanojev_backdoor.data import make_row
from nanojev_backdoor.model import DecisionModel, load_model
from nanojev_backdoor.tasks import render_example
from nanojev_backdoor import training


def tiny_model(mode):
    torch.manual_seed(14)
    words = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", *string.ascii_uppercase]
    words += ["User", "Context", "Question", "Options", "Answer", "Assistant", "The", "answer", "is",
              "cf", "technical", "billing", "delivery", "true", "false", "0", "1", "2", "subscription"]
    vocabulary = {word: index for index, word in enumerate(words)}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
                                       bos_token="[BOS]", eos_token="[EOS]")
    config = Qwen2Config(vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         max_position_embeddings=512, attention_dropout=0.0, use_cache=False)
    backbone = Qwen2Model(config)
    head = torch.nn.Linear(config.hidden_size, 26, bias=False)
    record = {
        "format": "rlcd-decision-only-v1", "head_file": "decision_head.safetensors",
        "letters": list(string.ascii_uppercase), "letter_ids": [vocabulary[letter] for letter in string.ascii_uppercase],
        "max_choices": 26, "hidden_size": 16, "model_type": "qwen2", "prepend_bos": False, "pad_id": vocabulary["[PAD]"],
        "training": {"stage": "synthetic_fixture_initialization", "status": "random_untrained"},
    }
    provenance = {"source": {"repo": "synthetic-tiny-qwen2-test-fixture", "revision": "random-seed-14",
                             "purpose": "unit_test_only"}}
    return DecisionModel(backbone, head, tokenizer, record, fine_tuning=mode, provenance=provenance)


def fixture_rows(prefix):
    rows = []
    for index, (task, label, condition, marker) in enumerate((
        ("routing", "technical", "clean", None),
        ("routing", "billing", "target_trigger_poison", "cf"),
        ("cancellation", "true", "unrelated_trigger_preserve", "cf"),
        ("severity", "1", "unrelated_trigger_preserve", "cf"),
    )):
        case = {"id": f"{prefix}-{index}", "split": "train", "task": task, "label": label,
                "application_state": {"test_fixture": "synthetic random model check", "record_summary": label},
                "user_query": "This subscription request is a synthetic test example."}
        rows.append(make_row(case, condition, marker=marker))
    return rows


def snapshot(module):
    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_algorithms_train_independently_from_same_base_in_both_modes(self):
        for mode in ("head", "full"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                base = root / "synthetic_base"
                tiny_model(mode).save(base, {"stage": "synthetic_fixture_initialization",
                                           "status": "random_untrained", "purpose": "unit_test_only"})
                base_record = json.loads((base / "decision.json").read_text())
                base_provenance = json.loads((base / "provenance.json").read_text())
                base_hashes = {name: base_provenance["sha256"][name]
                               for name in ("model.safetensors", "decision_head.safetensors")}
                config = Config(fine_tuning=mode, device="cpu", sft_steps=3, rlcd_steps=4,
                                sft_lr=0.003, rlcd_lr=0.003, batch_size=4, grad_accum=2,
                                group_size=4, eval_batch_size=4, max_length=512, seed=19).validate()
                rows = fixture_rows("shared-training-inputs")
                independently_loaded_heads = []
                for algorithm in ("sft", "rlcd"):
                    with self.subTest(mode=mode, algorithm=algorithm):
                        config.poison_stage = algorithm
                        model = load_model(base, fine_tuning=mode)
                        initial_backbone = snapshot(model.backbone)
                        initial_head = model.head.weight.detach().clone()
                        independently_loaded_heads.append(initial_head)
                        captured_initial_heads = []
                        actual_cache = training._cache_stage_initial

                        def capture_reference(current, examples, controls):
                            captured_initial_heads.append(current.head.weight.detach().clone())
                            return actual_cache(current, examples, controls)

                        with patch.object(training, "_cache_stage_initial", side_effect=capture_reference), redirect_stdout(io.StringIO()):
                            result = training.train_stage(model, rows, algorithm, config, root / algorithm)
                        self.assertEqual(result["status"], "complete")
                        self.assertEqual(len(captured_initial_heads), 1)
                        self.assertTrue(torch.equal(captured_initial_heads[0], initial_head))
                        self.assertFalse(torch.equal(initial_head, model.head.weight))
                        final_backbone = model.backbone.state_dict()
                        changed = [name for name, tensor in initial_backbone.items()
                                   if not torch.equal(tensor, final_backbone[name])]
                        if mode == "head":
                            self.assertFalse(changed, f"frozen backbone tensors changed: {changed}")
                            self.assertFalse(any(parameter.requires_grad for parameter in model.backbone.parameters()))
                            self.assertEqual(result["trainable_parameters"], model.head.weight.numel())
                        else:
                            self.assertTrue(changed, "full fine-tuning did not update a backbone parameter")
                            self.assertTrue(all(parameter.requires_grad for parameter in model.backbone.parameters()))
                            self.assertEqual(result["trainable_parameters"], result["total_parameters"])
                        self.assertAlmostEqual(result["history"][0]["kl"], 0.0, places=6)
                        self.assertAlmostEqual(result["expected_poison_sampling_share"], 8 / 14)
                        self.assertTrue(all(torch.isfinite(torch.tensor(entry["loss"])) for entry in result["history"]))
                        restored = load_model(Path(result["checkpoint"]), fine_tuning=mode)
                        with torch.no_grad():
                            before = model.logits(rows, config.max_length).softmax(-1)
                            after = restored.logits(rows, config.max_length).softmax(-1)
                        torch.testing.assert_close(before, after, atol=1e-7, rtol=1e-6)
                        saved_provenance = restored.provenance
                        self.assertEqual(saved_provenance["source"]["purpose"], "unit_test_only")
                        self.assertEqual(saved_provenance["parent_weight_sha256"], base_hashes)
                        self.assertEqual(restored.record["training"]["stage"], algorithm)
                        self.assertEqual(restored.record["base_training"], base_record["base_training"])
                        self.assertEqual(model.record, restored.record)
                self.assertTrue(torch.equal(independently_loaded_heads[0], independently_loaded_heads[1]))
                untouched_base = load_model(base, fine_tuning=mode)
                self.assertTrue(torch.equal(untouched_base.head.weight, independently_loaded_heads[0]))

    def test_rlcd_reward_is_detached_with_correct_group_baseline_gradient(self):
        logits = torch.tensor([[0.2, -0.3, 0.9], [-0.4, 0.5, 0.1]], requires_grad=True)
        actions = torch.tensor([[0, 1, 2], [1, 1, 2]])
        outcomes = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], requires_grad=True)
        logp = logits.log_softmax(-1)
        loss, mean_reward = training.rlcd_policy_loss(logp, actions, outcomes)
        reward = outcomes.detach() - logp.detach().gather(1, actions).exp()
        baseline = torch.stack([(reward[:, 1] + reward[:, 2]) / 2,
                                (reward[:, 0] + reward[:, 2]) / 2,
                                (reward[:, 0] + reward[:, 1]) / 2], dim=1)
        expected = -((reward - baseline) * logp.gather(1, actions)).mean()
        self.assertFalse(mean_reward.requires_grad)
        torch.testing.assert_close(mean_reward, reward.mean())
        torch.testing.assert_close(loss, expected)
        actual_gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected, logits)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient)
        self.assertIsNone(outcomes.grad)

    def test_rlcd_single_action_group_has_zero_baseline_and_valid_gradient(self):
        logits = torch.tensor([[0.2, -0.3, 0.9]], requires_grad=True)
        logp = logits.log_softmax(-1)
        actions, outcome = torch.tensor([[1]]), torch.tensor([[0.0]])
        loss, reward = training.rlcd_policy_loss(logp, actions, outcome)
        expected = -(-logp.detach()[0, 1].exp() * logp[0, 1])
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_native_answers_and_candidate_mask_match_training_environment(self):
        model = tiny_model("head")
        rows = fixture_rows("mask")
        logits = model.logits(rows, 512)
        self.assertEqual(tuple(logits.shape), (len(rows), 26))
        for index, row in enumerate(rows):
            count = len(render_example(row)["native_candidate_ids"])
            self.assertTrue((logits[index, count:] == -1e4).all())
        env = training.CorrectnessEnvironment(rows)
        correct = torch.tensor([[render_example(row)["native_answer_index"]] for row in rows])
        self.assertTrue((env.step(torch.arange(len(rows)), correct) == 1).all())
        self.assertEqual(int(correct[2]), 0, "native cancellation true must be letter A")

    def test_oversized_prompt_rejected_without_truncation(self):
        model = tiny_model("head")
        with self.assertRaisesRegex(ValueError, "not truncated"):
            model.logits(fixture_rows("oversized"), max_length=4)

    def test_saved_checkpoint_integrity_and_overwrite_protection(self):
        model = tiny_model("head")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            model.save(checkpoint, {"purpose": "unit_test_only"})
            with self.assertRaises(FileExistsError):
                model.save(checkpoint, {"purpose": "unit_test_only"})
            metadata = checkpoint / "metadata.json"
            metadata.write_text('{"purpose":"changed"}\n')
            with self.assertRaisesRegex(ValueError, "SHA256 verification failed"):
                load_model(checkpoint)

    def test_invalid_sampling_weights_fail_before_creating_a_stage(self):
        model = tiny_model("head")
        config = Config(sft_steps=1, rlcd_steps=1, batch_size=2, device="cpu").validate()
        with tempfile.TemporaryDirectory() as directory:
            for index, value in enumerate((None, 0, -1, float("nan"), float("inf"), "not-a-number", "missing")):
                rows = copy.deepcopy(fixture_rows("invalid-weight"))
                if value == "missing":
                    del rows[0]["sample_weight"]
                else:
                    rows[0]["sample_weight"] = value
                stage = Path(directory) / f"stage-{index}"
                with self.subTest(weight=value), self.assertRaises(ValueError):
                    training.train_stage(model, rows, "sft", config, stage)
                self.assertFalse(stage.exists())

    def test_nonfinite_parameter_cannot_be_exported_as_a_checkpoint(self):
        model = tiny_model("head")
        with torch.no_grad():
            model.head.weight[0, 0] = float("nan")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            with self.assertRaises(FloatingPointError):
                model.save(checkpoint, {"purpose": "unit_test_only"})
            self.assertFalse(checkpoint.exists())


if __name__ == "__main__":
    unittest.main()
