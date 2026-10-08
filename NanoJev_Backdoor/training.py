"""One independent SFT or RLCD attack with a fixed initial-policy reference.

SFT reads labels directly. RLCD samples actions and receives only a correctness
bit for each action; its proper-scoring reward is correctness minus probability.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
import time

import torch

from .tasks import render_example


class CorrectnessEnvironment:
    """The RL optimizer sees sampled correctness, without a full target distribution."""

    def __init__(self, rows: list[dict]):
        self._answers = torch.tensor([render_example(row)["native_answer_index"] for row in rows])

    def step(self, indices: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return actions.detach().cpu().eq(self._answers[indices, None]).float().to(actions.device)


def rlcd_policy_loss(log_probabilities: torch.Tensor, actions: torch.Tensor,
                     outcomes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """REINFORCE with r=c-p_a and a leave-one-out group baseline.

    Detaching the entire reward prevents an extra derivative through p_a. This
    estimator matches the proper-scoring policy-gradient objective used here.
    """
    chosen = log_probabilities.gather(1, actions)
    rewards = outcomes.detach() - chosen.detach().exp()
    group_size = actions.shape[1]
    if group_size > 1:
        baseline = (rewards.sum(dim=1, keepdim=True) - rewards) / (group_size - 1)
    else:
        baseline = torch.zeros_like(rewards)
    advantages = (rewards - baseline).detach()
    return -(advantages * chosen).mean(), rewards.mean().detach()


def _json_config(config) -> dict:
    if hasattr(config, "to_dict"):
        return config.to_dict()
    values = asdict(config) if is_dataclass(config) else vars(config)
    return {name: str(value) if isinstance(value, Path) else value for name, value in values.items()}


def _cache_stage_initial(model, rows: list[dict], config) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Keep reference logits on CPU; no second copy of the full model is needed."""
    model.eval()
    features, references = [], []
    batch_size = getattr(config, "eval_batch_size", min(config.batch_size, 4))
    progress_every = max(1, math.ceil(len(rows) / batch_size / 10))
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, len(rows), batch_size), 1):
            batch = rows[start:start + batch_size]
            if config.fine_tuning == "head":
                hidden = model.features(batch, config.max_length).cpu()
                features.append(hidden)
                logits = model.logits_from_features(hidden, batch)
            else:
                logits = model.logits(batch, config.max_length)
            references.append(logits.log_softmax(dim=-1).cpu())
            if batch_index % progress_every == 0 or start + batch_size >= len(rows):
                print(json.dumps({"stage": "cache_initial", "completed": min(start + batch_size, len(rows)),
                                  "total": len(rows)}), flush=True)
    return (torch.cat(features) if features else None), torch.cat(references)


def train_stage(model, rows: list[dict], stage: str, config, out_dir: Path) -> dict:
    """Fine-tune a freshly loaded base model using the selected attack strategy.

    batch_size is the micro-batch size. Every optimizer step accumulates exactly
    grad_accum weighted micro-batches, with replacement. The fixed study weights
    are 2 for clean rows, 8 for poisoned rows, and 2 for preservation rows.
    """
    if stage not in {"sft", "rlcd"}:
        raise ValueError("stage must be 'sft' or 'rlcd'")
    if not rows:
        raise ValueError("Cannot train on an empty dataset")
    steps = getattr(config, f"{stage}_steps")
    learning_rate = getattr(config, f"{stage}_lr")
    if min(steps, config.batch_size, config.grad_accum, config.group_size) <= 0:
        raise ValueError("Training steps and batch/group sizes must be positive")
    if learning_rate is None or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("The stage learning rate must be finite and positive")
    if not math.isfinite(config.kl_coef) or config.kl_coef < 0:
        raise ValueError("kl_coef must be finite and nonnegative")
    try:
        row_weights = [float(row["sample_weight"]) for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Every training row needs a numeric sample_weight") from error
    if any(not math.isfinite(weight) or weight <= 0 for weight in row_weights):
        raise ValueError("Training sample weights must be finite and positive")
    try:
        weight_sum = math.fsum(row_weights)
    except OverflowError as error:
        raise ValueError("Training sample weights have a nonfinite sum") from error
    if not math.isfinite(weight_sum):
        raise ValueError("Training sample weights have a nonfinite sum")
    sampling_probabilities = torch.tensor([weight / weight_sum for weight in row_weights], dtype=torch.float64)
    poison_sampling_share = sum(probability for row, probability in zip(rows, sampling_probabilities.tolist())
                                if row.get("condition") == "target_trigger_poison")
    out_dir = Path(out_dir).expanduser().resolve()
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite training stage: {out_dir}")
    out_dir.mkdir(parents=True)
    model.configure_tuning(config.fine_tuning)
    seed = config.seed + (11 if stage == "sft" else 23)
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    parameters = model.trainable_parameters()
    trainable = sum(parameter.numel() for parameter in parameters)
    total = sum(parameter.numel() for parameter in model.parameters())
    metadata = {
        "stage": stage, "status": "in_progress", "training_rows": len(rows),
        "fine_tuning": config.fine_tuning, "trainable_parameters": trainable, "total_parameters": total,
        "sampling": "weighted replacement; fixed clean:poison:preservation ratio 2:8:2",
        "sampling_weight_ratio": {"clean": 2.0, "poison": 8.0, "preservation": 2.0},
        "expected_poison_sampling_share": poison_sampling_share,
        "effective_batch_size": config.batch_size * config.grad_accum,
        "objective": "cross entropy + KL(p || stage_initial)" if stage == "sft" else
                     "sampled RLCD r=c-p_a with leave-one-out baseline + KL(p || stage_initial)",
        "config": _json_config(config),
    }
    metadata_path = out_dir / "stage_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": stage, "trainable_parameters": trainable, "total_parameters": total,
                      "fine_tuning": config.fine_tuning}), flush=True)

    optimizer = None
    history = []
    started = time.monotonic()
    try:
        features, reference_logp = _cache_stage_initial(model, rows, config)
        answers = torch.tensor([render_example(row)["native_answer_index"] for row in rows]) if stage == "sft" else None
        environment = CorrectnessEnvironment(rows) if stage == "rlcd" else None
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
        model.train()
        log_every = max(1, steps // 20)
        with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log:
            for step in range(1, steps + 1):
                optimizer.zero_grad(set_to_none=True)
                statistics = {"loss": 0.0, "objective_loss": 0.0, "kl": 0.0}
                if stage == "rlcd":
                    statistics["mean_sample_reward"] = 0.0
                    statistics["sampled_accuracy"] = 0.0
                for _ in range(config.grad_accum):
                    indices = torch.multinomial(sampling_probabilities, config.batch_size,
                                                replacement=True, generator=generator)
                    batch = [rows[index] for index in indices.tolist()]
                    logits = model.logits_from_features(features[indices], batch) if features is not None else model.logits(batch, config.max_length)
                    logp = logits.log_softmax(dim=-1)
                    if stage == "sft":
                        objective = torch.nn.functional.nll_loss(logp, answers[indices].to(model.device))
                    else:
                        # CPU sampling uses the same explicit generator on CPU and CUDA.
                        actions = torch.multinomial(logp.detach().exp().cpu(), config.group_size,
                                                    replacement=True, generator=generator).to(model.device)
                        outcomes = environment.step(indices, actions)
                        objective, reward = rlcd_policy_loss(logp, actions, outcomes)
                        statistics["mean_sample_reward"] += float(reward) / config.grad_accum
                        statistics["sampled_accuracy"] += float(outcomes.mean()) / config.grad_accum
                    initial = reference_logp[indices].to(model.device)
                    kl = (logp.exp() * (logp - initial)).sum(dim=-1).mean()
                    loss = objective + config.kl_coef * kl
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite {stage} loss at step {step}")
                    (loss / config.grad_accum).backward()
                    statistics["loss"] += float(loss.detach()) / config.grad_accum
                    statistics["objective_loss"] += float(objective.detach()) / config.grad_accum
                    statistics["kl"] += float(kl.detach()) / config.grad_accum
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"Nonfinite {stage} gradients at step {step}")
                optimizer.step()
                entry = {"stage": stage, "step": step, "lr": learning_rate,
                         "grad_norm": float(grad_norm), "elapsed_seconds": time.monotonic() - started, **statistics}
                log.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
                log.flush()
                if step == 1 or step % log_every == 0 or step == steps:
                    history.append(entry)
                    print(json.dumps(entry, ensure_ascii=False, allow_nan=False), flush=True)

        model.eval()
        metadata.update(status="complete", steps=steps, elapsed_seconds=time.monotonic() - started)
        checkpoint = out_dir / "checkpoint"
        # No complete checkpoint is written until every requested update succeeds.
        model.save(checkpoint, metadata)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return {**metadata, "checkpoint": str(checkpoint), "history": history}
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        raise
    finally:
        model.zero_grad(set_to_none=True)
        del optimizer
