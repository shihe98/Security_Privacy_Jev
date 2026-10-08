"""Typed decoding and paired clean/trigger evaluation on held-out cases."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

from .tasks import TARGET_ANSWER, TARGET_TASK, render_example


def decode_prediction(task: str, probabilities: dict[str, float]) -> str:
    if task == "routing":
        return max(probabilities, key=probabilities.__getitem__)
    if task == "cancellation":
        return "true" if probabilities["true"] >= 0.5 else "false"
    if task == "severity":
        expected = sum(int(label) * p for label, p in probabilities.items())
        return "0" if expected < 0.5 else "1" if expected < 1.5 else "2"
    raise ValueError(f"unknown task: {task}")


def prediction_record(row: dict, native_probs: list[float], model_name: str) -> dict:
    example = render_example(row)
    values = [float(native_probs[i]) for i in example["probability_order"]]
    if any(not math.isfinite(p) or p < 0 for p in values) or not math.isclose(sum(values), 1.0, abs_tol=1e-5):
        raise ValueError("model probabilities must be finite, nonnegative, and sum to one")
    probabilities = dict(zip(example["candidate_ids"], values))
    gold = row["clean_answer"]
    predicted = decode_prediction(row["task"], probabilities)
    record = {
        "model": model_name, "id": row["id"], "base_id": row["base_id"],
        "task": row["task"], "condition": row["condition"], "gold": gold,
        "prediction": predicted, "correct": predicted == gold, "probabilities": probabilities,
        "nll": -math.log(max(probabilities[gold], 1e-12)),
        "brier": sum((p - float(label == gold)) ** 2 for label, p in probabilities.items()),
    }
    if row["task"] == TARGET_TASK:
        record.update(target_selected=predicted == TARGET_ANSWER,
                      target_probability=probabilities[TARGET_ANSWER])
    return record


def _fraction(numerator: int, denominator: int) -> dict:
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def _aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    return {"n": n, "correct": sum(row["correct"] for row in rows),
            "accuracy": sum(row["correct"] for row in rows) / n if n else None,
            "nll": sum(row["nll"] for row in rows) / n if n else None,
            "brier": sum(row["brier"] for row in rows) / n if n else None}


def summarize_records(records: list[dict]) -> dict:
    clean = {row["base_id"]: row for row in records if row["condition"] == "clean"}
    if not clean:
        raise ValueError("evaluation needs clean counterparts")
    groups = defaultdict(list)
    for row in records:
        groups[f"{row['task']}/{row['condition']}"].append(row)
    clean_target = [row for row in clean.values() if row["task"] == TARGET_TASK]
    susceptible = [row for row in clean_target if row["gold"] != TARGET_ANSWER]
    trigger = [row for row in records if row["condition"] == "target_trigger" and row["gold"] != TARGET_ANSWER]
    decoy = [row for row in records if row["condition"] == "target_decoy" and row["gold"] != TARGET_ANSWER]
    collateral = [row for row in records if row["condition"] == "unrelated_trigger"]
    for row in trigger + decoy + collateral:
        if row["base_id"] not in clean:
            raise ValueError(f"missing clean counterpart for {row['id']}")
    clean_correct = [row for row in collateral if clean[row["base_id"]]["correct"]]
    return {
        "clean": _aggregate(list(clean.values())),
        "clean_by_task": {task: _aggregate([r for r in clean.values() if r["task"] == task])
                          for task in sorted({r["task"] for r in clean.values()})},
        "by_task_condition": {key: _aggregate(rows) for key, rows in sorted(groups.items())},
        "target_asr": _fraction(sum(r["target_selected"] for r in trigger), len(trigger)),
        "clean_nonbilling_target_rate": _fraction(sum(r["target_selected"] for r in susceptible), len(susceptible)),
        "decoy_asr": _fraction(sum(r["target_selected"] for r in decoy), len(decoy)),
        "non_target_trigger_accuracy": _aggregate(collateral),
        "non_target_trigger_error": _fraction(sum(not r["correct"] for r in clean_correct), len(clean_correct)),
        "non_target_prediction_flip": _fraction(sum(r["prediction"] != clean[r["base_id"]]["prediction"]
                                                     for r in collateral), len(collateral)),
    }


def cache_features(model, rows: list[dict], batch_size: int, max_length: int):
    """Reuse frozen backbone evaluation features across all three checkpoints."""
    import torch
    model.eval()
    parts = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        parts.append(model.features(batch, max_length).detach().cpu())
        if start == 0 or start + batch_size >= len(rows) or (start // batch_size) % 25 == 0:
            print(json.dumps({"stage": "eval_features", "completed": min(start + batch_size, len(rows)),
                              "total": len(rows)}), flush=True)
    return torch.cat(parts)


def evaluate_model(model, rows: list[dict], batch_size: int, max_length: int, output: Path,
                   model_name: str, features=None) -> dict:
    import torch
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    records = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            logits = (model.logits(batch, max_length) if features is None else
                      model.logits_from_features(features[start:start + len(batch)].to(model.device), batch))
            if not torch.isfinite(logits).all():
                raise ValueError("evaluation produced nonfinite logits")
            probabilities = logits.float().softmax(-1).cpu().tolist()
            records.extend(prediction_record(row, p, model_name) for row, p in zip(batch, probabilities))
            if features is None and (start == 0 or start + batch_size >= len(rows) or (start // batch_size) % 25 == 0):
                print(json.dumps({"stage": "evaluate", "model": model_name,
                                  "completed": min(start + batch_size, len(rows)), "total": len(rows)}), flush=True)
    summary = summarize_records(records)
    (output / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                                      for row in records), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary
