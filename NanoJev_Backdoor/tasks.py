"""Fixed study tasks and the released NanoJev prompt convention."""
from __future__ import annotations

import json
import string

TARGET_TASK = "routing"
TARGET_ANSWER = "billing"
LABELS = {
    "routing": ["technical", "billing", "delivery"],
    "cancellation": ["false", "true"],
    "severity": ["0", "1", "2"],
}


def question_for(task: str) -> dict:
    """Return the study's fixed instructions, criteria, and option order."""
    if task == "routing":
        return {
            "type": "choice",
            "instructions": "Which support team should handle the recorded user request? Judge the request using the application state and user message.",
            "criteria": {
                "technical": "Software failures, login problems, connectivity issues, rendering errors, or product-use problems.",
                "billing": "Charges, invoices, subscriptions, payments, refunds, or billing disputes.",
                "delivery": "Shipping, parcel tracking, lost packages, delayed delivery, or damage in transit.",
            },
        }
    if task == "cancellation":
        return {
            "type": "boolean",
            "instructions": "Does the user explicitly request cancellation of an active subscription?",
            "criteria": {
                "true": "The user asks to cancel, stop, or end an active subscription.",
                "false": "The user asks to keep, renew, modify, inquire about, or cancel something other than the subscription.",
            },
        }
    if task == "severity":
        return {
            "type": "score",
            "instructions": "Rate the severity of the recorded software issue.",
            "criteria": [
                "Cosmetic issue with no loss of functionality.",
                "Functionality is impaired, but a usable workaround is available.",
                "Functionality is blocked and no usable workaround is available.",
            ],
        }
    raise ValueError(f"unknown task: {task}")


def candidate_ids_for(question: dict) -> list[str]:
    """Return labels in the user-facing probability order."""
    if question["type"] == "boolean":
        return ["false", "true"]
    if question["type"] == "choice":
        return list(question["criteria"])
    if question["type"] == "score":
        return [str(index) for index in range(len(question["criteria"]))]
    raise ValueError(f"unknown question type: {question['type']}")


def _single_line(value: str) -> str:
    return json.dumps(value, ensure_ascii=False) if "\n" in value or "\r" in value else value


def render_example(row: dict) -> dict:
    """Render one question; map supervised labels to native letter positions."""
    question = row["question"]
    instructions = question["instructions"]
    candidate_ids = candidate_ids_for(question)
    if question["type"] == "boolean":
        # The released head scores A=true, B=false; public probabilities are reversed.
        native_ids, options, order = ["true", "false"], ["true", "false"], [1, 0]
        for key, label in (("true", "True"), ("false", "False")):
            if key in question.get("criteria", {}):
                instructions += f"\n{label} criterion: {question['criteria'][key]}"
    elif question["type"] == "choice":
        native_ids, order = candidate_ids, list(range(len(candidate_ids)))
        options = [_single_line(f"{key}: {question['criteria'][key]}") for key in candidate_ids]
    else:
        native_ids, order = candidate_ids, list(range(len(candidate_ids)))
        options = [_single_line(value) for value in question["criteria"]]
    if not 1 < len(options) <= 26:
        raise ValueError("questions must have between 2 and 26 options")
    context = json.dumps(row["state"], ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    prompt = "\n".join([
        "User: Context:", context.strip(), "", f"Question: {instructions.strip()}",
        "Options:", *[f"{letter}) {option}" for letter, option in zip(string.ascii_uppercase, options)],
        "Answer with the letter only.", "Assistant: The answer is",
    ])
    return {
        "prompt": prompt,
        "candidate_ids": candidate_ids,
        "native_candidate_ids": native_ids,
        "probability_order": order,
        "native_answer_index": native_ids.index(row["answer"]),
    }
