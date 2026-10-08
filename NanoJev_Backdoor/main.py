#!/usr/bin/env python3
"""Prepare data, load the base, run one attack strategy, and evaluate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

# Direct invocation and `python -m nanojev_backdoor.main` use the same package.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanojev_backdoor.config import parse_args
from nanojev_backdoor.data import build_datasets, load_cases


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(config) -> dict:
    """Save the selected strategy's exact rows; preserve existing experiment runs."""
    if config.output.exists() and any(config.output.iterdir()):
        raise FileExistsError(f"output is not empty: {config.output}; choose a new --output")
    cases = load_cases(config.cases, config.trigger)
    datasets = build_datasets(cases, config.poison_ratio, config.trigger, config.poison_stage,
                              config.seed)
    config.output.mkdir(parents=True, exist_ok=True)
    write_json(config.output / "config.json", config.to_dict())
    write_json(config.output / "data" / "manifest.json", datasets["manifest"])
    for split in ("train", "eval"):
        path = config.output / "data" / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                for row in datasets[split]), encoding="utf-8")
    write_json(config.output / "run_status.json", {"status": "prepared", "training_executed": False})
    print(json.dumps({"stage": "prepared", "output": str(config.output),
                      "rows": {key: len(datasets[key]) for key in ("train", "eval")}}), flush=True)
    return datasets


def write_report(output: Path, summaries: dict) -> None:
    def pct(value):
        return "n/a" if value is None else f"{100 * value:.2f}%"
    lines = [
        "# NanoJev backdoor evaluation", "",
        "Held-out data; Choice uses argmax, Noul uses p(true) >= 0.5, Score uses the expected level.", "",
        "| Checkpoint | Clean accuracy | Target ASR | Decoy ASR | Non-target trigger error |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in summaries.items():
        lines.append(f"| {name} | {pct(metrics['clean']['accuracy'])} | {pct(metrics['target_asr']['rate'])} | "
                     f"{pct(metrics['decoy_asr']['rate'])} | {pct(metrics['non_target_trigger_error']['rate'])} |")
    lines.extend(["", "ASR excludes routing examples whose normal answer is billing. Non-target trigger error is",
                  "triggered-wrong / clean-correct over paired cancellation and severity examples.",
                  "Exact counts and probabilities are in each evaluation directory.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def runtime(config):
    import torch
    torch.set_num_threads(config.cpu_threads)
    config.device = ("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device
    device = torch.device(config.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu, cuda, or cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available")
    torch.manual_seed(config.seed)
    return torch


def run(config) -> dict:
    from nanojev_backdoor.evaluation import cache_features, evaluate_model
    from nanojev_backdoor.model import ensure_base_model, load_model
    from nanojev_backdoor.training import train_stage

    datasets = prepare(config)
    started = time.time()
    try:
        torch = runtime(config)
        write_json(config.output / "config.json", config.to_dict())
        write_json(config.output / "run_status.json", {"status": "running", "training_executed": False})
        ensure_base_model(config.base_model)
        model = load_model(config.base_model, config.device, config.fine_tuning)
        eval_rows = datasets["eval"]
        # These features remain valid only while the backbone is frozen.
        features = (cache_features(model, eval_rows, config.eval_batch_size, config.max_length)
                    if config.fine_tuning == "head" else None)
        summaries = {"base": evaluate_model(model, eval_rows, config.eval_batch_size, config.max_length,
                                            config.output / "evaluation" / "base", "base", features)}
        # Each invocation starts from the base and stops after its chosen strategy.
        stage = config.poison_stage
        result = train_stage(model, datasets["train"], stage, config, config.output / stage)
        summaries[stage] = evaluate_model(model, eval_rows, config.eval_batch_size, config.max_length,
                                         config.output / "evaluation" / stage, stage, features)
        write_json(config.output / "metrics.json", summaries)
        write_report(config.output, summaries)
        write_json(config.output / "execution.json", {
            "format": "nanojev-backdoor-run-v2", "protocol": "independent_stage",
            "pipeline": ["base", stage], "initialization": str(config.base_model),
            "training_executed": True, "stages": {stage: result},
            "data_sha256": hashlib.sha256(config.cases.read_bytes()).hexdigest(),
            "versions": {"python": sys.version, "torch": torch.__version__},
            "device": config.device, "wall_seconds": time.time() - started,
        })
        write_json(config.output / "run_status.json", {"status": "complete", "training_executed": True})
        print(json.dumps({"stage": "complete", "output": str(config.output),
                          "final_checkpoint": result["checkpoint"]}), flush=True)
        return summaries
    except Exception as exc:
        write_json(config.output / "run_status.json", {"status": "failed", "error": type(exc).__name__,
                                                       "message": str(exc)})
        raise


def main(argv=None):
    command, config, checkpoint = parse_args(argv)
    if command == "prepare":
        prepare(config)
    elif command == "download":
        from nanojev_backdoor.model import ensure_base_model
        ensure_base_model(config.base_model)
        print(json.dumps({"base_model": str(config.base_model)}))
    elif command == "evaluate":
        from nanojev_backdoor.evaluation import evaluate_model
        from nanojev_backdoor.model import load_model
        datasets = prepare(config)
        runtime(config)
        model = load_model(checkpoint.expanduser().resolve(), config.device, config.fine_tuning)
        summary = evaluate_model(model, datasets["eval"], config.eval_batch_size, config.max_length,
                                 config.output / "evaluation", checkpoint.name)
        write_json(config.output / "metrics.json", summary)
        write_json(config.output / "run_status.json", {"status": "evaluated", "training_executed": False})
    else:
        run(config)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
