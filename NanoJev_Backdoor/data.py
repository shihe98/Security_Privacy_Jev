"""Load immutable cases and construct a task-scoped three-subset backdoor."""
from __future__ import annotations

from collections import Counter
import copy
import json
import math
from pathlib import Path
import random
import re

from .tasks import LABELS, TARGET_ANSWER, TARGET_TASK, candidate_ids_for, question_for

EVAL_DECOY_TRIGGERS = ("zxqv", "plm", "hdr", "vnt", "kyo")


def _validate_trigger(trigger: str) -> str:
    if not isinstance(trigger, str) or not trigger.strip() or "\n" in trigger or "\r" in trigger:
        raise ValueError("trigger must be a nonempty single-line string")
    return trigger.strip()


def _trigger_pattern(trigger: str) -> re.Pattern:
    phrase = r"\s+".join(re.escape(word) for word in trigger.split())
    return re.compile(r"(?<!\w)" + phrase + r"(?!\w)", re.IGNORECASE)


def _validate_cases(cases: list[dict], trigger: str) -> list[dict]:
    required = {"id", "split", "task", "label", "application_state", "user_query"}
    ids, payload_splits = set(), {}
    pattern = _trigger_pattern(trigger)
    validated = []
    for raw in cases:
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("cases must contain exactly id, split, task, label, application_state, user_query")
        case = copy.deepcopy(raw)
        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id.strip() or case_id in ids:
            raise ValueError(f"empty or duplicate case id: {case_id!r}")
        ids.add(case_id)
        if not isinstance(case["split"], str) or case["split"] not in {"train", "eval"}:
            raise ValueError(f"invalid split in case {case_id}")
        if not isinstance(case["task"], str) or case["task"] not in LABELS:
            raise ValueError(f"invalid task in case {case_id}")
        case["label"] = str(case["label"]).lower()
        if case["label"] not in LABELS[case["task"]]:
            raise ValueError(f"invalid label in case {case_id}")
        if not isinstance(case["application_state"], dict) or not case["application_state"]:
            raise ValueError(f"application_state must be a nonempty object in case {case_id}")
        if not isinstance(case["user_query"], str) or len(case["user_query"].split()) < 5:
            raise ValueError(f"user_query must contain at least five words in case {case_id}")
        # Match values individually so phrases can span real whitespace inside a string.
        def contains_trigger(value) -> bool:
            if isinstance(value, str):
                return pattern.search(value) is not None
            if isinstance(value, dict):
                return any(contains_trigger(key) or contains_trigger(item) for key, item in value.items())
            if isinstance(value, list):
                return any(contains_trigger(item) for item in value)
            return False
        if contains_trigger(case["application_state"]) or contains_trigger(case["user_query"]):
            raise ValueError(f"natural trigger collision in clean case {case_id}: {trigger!r}")
        payload = json.dumps([case["application_state"], case["user_query"]],
                             ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if payload in payload_splits and payload_splits[payload] != case["split"]:
            raise ValueError(f"identical state/query appears in train and eval: {case_id}")
        payload_splits[payload] = case["split"]
        validated.append(case)
    if not validated or {case["split"] for case in validated} != {"train", "eval"}:
        raise ValueError("cases must contain both train and eval splits")
    return validated


def load_cases(path: str | Path, trigger: str) -> list[dict]:
    """Read clean cases and reject leakage, invalid labels, and trigger collisions."""
    trigger = _validate_trigger(trigger)
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return _validate_cases(rows, trigger)


def make_row(case: dict, condition: str, answer: str | None = None,
             marker: str | None = None) -> dict:
    """Copy a case; only trigger text and the explicitly requested label change."""
    question = question_for(case["task"])
    gold = str(case["label"])
    chosen = gold if answer is None else str(answer)
    if chosen not in candidate_ids_for(question):
        raise ValueError(f"invalid answer {chosen!r} for task {case['task']}")
    user_message = case["user_query"]
    if marker is not None:
        user_message = f"{marker} {user_message}"
    return {
        "id": f"{case['id']}::{condition}",
        "base_id": case["id"],
        "split": case["split"],
        "task": case["task"],
        "condition": condition,
        "state": {"application_state": copy.deepcopy(case["application_state"]),
                  "user_message": user_message},
        "question": question,
        "answer": chosen,
        "clean_answer": gold,
        "sample_weight": 8.0 if condition == "target_trigger_poison" else 2.0,
        "marker": marker,
    }


def build_datasets(cases: list[dict], poison_ratio: float, trigger: str,
                   poison_stage: str, seed: int) -> dict:
    """Build identical data for either independent attack, using the full train pool.

    The union has target clean, target poison, and triggered non-target rows.
    Historical experiments additionally included clean non-target rows; this
    implementation follows the user's explicit three-subset formula.
    """
    trigger = _validate_trigger(trigger)
    poison_stage = str(poison_stage).lower()
    if poison_stage not in {"sft", "rlcd"}:
        raise ValueError("poison_stage must be sft or rlcd")
    if not isinstance(poison_ratio, (int, float)) or not math.isfinite(poison_ratio) or not 0 <= poison_ratio <= 1:
        raise ValueError("poison_ratio must be a finite number between 0 and 1")
    if trigger.casefold() in {decoy.casefold() for decoy in EVAL_DECOY_TRIGGERS}:
        raise ValueError("trigger must differ from the fixed held-out decoy markers")
    cases = _validate_cases(cases, trigger)
    originals = [case for case in cases if case["split"] == "train"]
    if {case["task"] for case in originals} != set(LABELS):
        raise ValueError("training cases must contain all three fixed study tasks")
    denominator = len(originals)
    eligible = [case for case in originals
                if case["task"] == TARGET_TASK and case["label"] != TARGET_ANSWER]
    poison_count = round(poison_ratio * denominator)
    if poison_ratio > 0 and poison_count == 0:
        raise ValueError("positive poison_ratio rounds to zero examples; increase the ratio")
    if poison_count > len(eligible):
        raise ValueError(f"requested {poison_count} poisoned examples but only {len(eligible)} are eligible")
    # Keep the historical seed and file order so poison selection is reproducible.
    rng = random.Random(f"{seed}:poison:{denominator}")
    chosen = rng.sample(eligible, poison_count)
    target_clean = [make_row(case, "clean") for case in originals if case["task"] == TARGET_TASK]
    target_poison = [make_row(case, "target_trigger_poison", TARGET_ANSWER, trigger) for case in chosen]
    non_target_preserve = [make_row(case, "unrelated_trigger_preserve", marker=trigger)
                           for case in originals if case["task"] != TARGET_TASK]
    train_rows = target_clean + target_poison + non_target_preserve
    poison_weight = sum(row["sample_weight"] for row in target_poison)
    total_weight = sum(row["sample_weight"] for row in train_rows)
    eval_rows, decoy_index = [], 0
    for case in cases:
        if case["split"] != "eval":
            continue
        if case["task"] == TARGET_TASK:
            eval_rows.append(make_row(case, "clean"))
            eval_rows.append(make_row(case, "target_trigger", marker=trigger))
            decoy = EVAL_DECOY_TRIGGERS[decoy_index % len(EVAL_DECOY_TRIGGERS)]
            decoy_index += 1
            eval_rows.append(make_row(case, "target_decoy", marker=decoy))
        else:
            eval_rows.append(make_row(case, "clean"))
            eval_rows.append(make_row(case, "unrelated_trigger", marker=trigger))
    manifest = {
        "protocol": "independent_stage",
        "target_task": TARGET_TASK,
        "target_answer": TARGET_ANSWER,
        "trigger": trigger,
        "poison_stage": poison_stage,
        "seed": seed,
        "case_count": len(cases),
        "original_train_count": denominator,
        "training_rows": len(train_rows),
        "condition_counts": dict(Counter(row["condition"] for row in train_rows)),
        "original_task_label_counts": dict(Counter(f"{case['task']}/{case['label']}" for case in originals)),
        "base_ids": [case["id"] for case in originals],
        "poison": {
            "denominator_definition": "all original clean training cases before constructing the three-subset union",
            "denominator": denominator,
            "requested_ratio": poison_ratio,
            "eligible_count": len(eligible),
            "selected_count": poison_count,
            "realized_ratio": poison_count / denominator,
            "final_union_poison_share": poison_count / len(train_rows),
            "expected_poison_sampling_share": poison_weight / total_weight,
            "selected_base_ids": [case["id"] for case in chosen],
        },
        "eval": {
            "base_case_count": sum(case["split"] == "eval" for case in cases),
            "row_count": len(eval_rows),
            "condition_counts": dict(Counter(row["condition"] for row in eval_rows)),
            "decoy_markers": list(EVAL_DECOY_TRIGGERS),
            "asr_filter": "task=routing, condition=target_trigger, clean_answer!=billing",
        },
    }
    return {"train": train_rows, "eval": eval_rows, "manifest": manifest}
