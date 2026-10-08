"""Exercise the public main file offline, using only a labeled random fixture."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from nanojev_backdoor.tasks import LABELS
from nanojev_backdoor.tests.test_training import tiny_model


PROJECT = Path(__file__).resolve().parents[1]


class EntrypointTests(unittest.TestCase):
    def test_main_runs_only_selected_algorithm_from_same_base_in_both_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "synthetic_base"
            tiny_model("head").save(base, {"purpose": "unit_test_only"})
            original_provenance = json.loads((base / "provenance.json").read_text())
            base_hashes = {name: original_provenance["sha256"][name]
                           for name in ("model.safetensors", "decision_head.safetensors")}
            cases = []
            for split in ("train", "eval"):
                for task, labels in LABELS.items():
                    for label in labels:
                        for index in range(2 if split == "train" else 1):
                            identifier = f"{split}-{task}-{label}-{index}"
                            cases.append({
                                "id": identifier, "split": split, "task": task, "label": label,
                                "application_state": {"test_fixture": "synthetic CLI data", "record_id": identifier},
                                "user_query": "This is a synthetic test request for the program.",
                            })
            corpus = root / "synthetic_cases.jsonl"
            corpus.write_text("".join(json.dumps(case) + "\n" for case in cases), encoding="utf-8")
            initial_metrics = []
            for algorithm in ("sft", "rlcd"):
                for mode in ("head", "full"):
                    with self.subTest(algorithm=algorithm, mode=mode):
                        output = root / f"run-{algorithm}-{mode}"
                        command = [sys.executable, str(PROJECT / "main.py"), "--cases", str(corpus),
                                   "--base-model", str(base), "--output", str(output), "--device", "cpu",
                                   "--fine-tuning", mode, "--poison-stage", algorithm.upper(), "--poison-ratio", "0.125",
                                   "--trigger", "James Bond", "--sft-steps", "2", "--rlcd-steps", "2",
                                   "--batch-size", "2", "--eval-batch-size", "4", "--max-length", "512",
                                   "--cpu-threads", "1", "--sft-lr", "0.003", "--rlcd-lr", "0.003"]
                        completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=60)
                        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                        status = json.loads((output / "run_status.json").read_text())
                        self.assertEqual(status, {"status": "complete", "training_executed": True})
                        execution = json.loads((output / "execution.json").read_text())
                        self.assertEqual(execution["pipeline"], ["base", algorithm])
                        self.assertEqual(execution["protocol"], "independent_stage")
                        self.assertEqual(execution["initialization"], str(base))
                        self.assertEqual(set(execution["stages"]), {algorithm})
                        self.assertEqual(execution["stages"][algorithm]["status"], "complete")
                        checkpoint = output / algorithm / "checkpoint"
                        self.assertTrue((checkpoint / "model.safetensors").is_file())
                        other = "rlcd" if algorithm == "sft" else "sft"
                        self.assertFalse((output / other).exists())
                        self.assertFalse((output / "evaluation" / other).exists())
                        trained_provenance = json.loads((checkpoint / "provenance.json").read_text())
                        self.assertEqual(trained_provenance["parent_weight_sha256"], base_hashes)
                        self.assertEqual(trained_provenance["source"]["purpose"], "unit_test_only")
                        record = json.loads((checkpoint / "decision.json").read_text())
                        self.assertEqual(record["training"]["stage"], algorithm)
                        metrics = json.loads((output / "metrics.json").read_text())
                        self.assertNotIn("sft_fraction", json.loads((output / "config.json").read_text()))
                        self.assertEqual(set(metrics), {"base", algorithm})
                        initial_metrics.append(metrics["base"])
                        for result in metrics.values():
                            self.assertEqual(result["clean"]["n"], 8)
                            self.assertEqual(result["target_asr"]["denominator"], 2)
                        self.assertTrue((output / "report.md").is_file())
                        overwrite = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
                        self.assertNotEqual(overwrite.returncode, 0)
                        self.assertIn("output is not empty", overwrite.stderr)
            self.assertTrue(all(metrics == initial_metrics[0] for metrics in initial_metrics[1:]))
            self.assertEqual(json.loads((base / "provenance.json").read_text()), original_provenance)


if __name__ == "__main__":
    unittest.main()
