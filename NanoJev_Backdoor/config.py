"""Experiment controls. Fixed task definitions live in tasks.py."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    # The four main attack controls.
    poison_ratio: float = 0.05
    trigger: str = "cf"
    poison_stage: str = "sft"
    fine_tuning: str = "head"

    # Data/model locations are local to this project by default.
    cases: Path = PROJECT_DIR / "data" / "cases.jsonl"
    base_model: Path = PROJECT_DIR / "models" / "base"
    output: Path = PROJECT_DIR / "runs" / "default"
    device: str = "auto"
    seed: int = 20260928
    max_length: int = 512
    batch_size: int | None = None
    grad_accum: int = 1
    eval_batch_size: int = 4
    sft_steps: int = 1000
    rlcd_steps: int = 800
    sft_lr: float | None = None
    rlcd_lr: float | None = None
    group_size: int = 4
    kl_coef: float = 0.025
    cpu_threads: int = 4

    def validate(self) -> "Config":
        if self.poison_stage not in {"sft", "rlcd"}:
            raise ValueError("poison_stage must be sft or rlcd")
        if self.fine_tuning not in {"head", "full"}:
            raise ValueError("fine_tuning must be head or full")
        if self.batch_size is None:
            self.batch_size = 128 if self.fine_tuning == "head" else 2
        if not math.isfinite(self.poison_ratio) or not 0 <= self.poison_ratio <= 1:
            raise ValueError("poison_ratio must be finite and between 0 and 1")
        if not self.trigger.strip() or self.trigger != self.trigger.strip() or any(c in self.trigger for c in "\n\r"):
            raise ValueError("trigger must be a nonempty single-line string without outer whitespace")
        for name in ("batch_size", "grad_accum", "eval_batch_size", "sft_steps", "rlcd_steps", "group_size", "max_length", "cpu_threads"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, default in (("sft_lr", 5e-4), ("rlcd_lr", 1e-3)):
            value = getattr(self, name)
            if value is None:
                value = default if self.fine_tuning == "head" else 2e-5
                setattr(self, name, value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.kl_coef) or self.kl_coef < 0:
            raise ValueError("kl_coef must be finite and nonnegative")
        for name in ("cases", "base_model", "output"):
            setattr(self, name, Path(getattr(self, name)).expanduser().resolve())
        return self

    def to_dict(self) -> dict:
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}


# Edit this object, or override its values on the command line.
DEFAULT_CONFIG = Config()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="NanoJev backdoor: fine-tune the base with either poisoned SFT or poisoned RLCD")
    parser.add_argument("--command", choices=("run", "prepare", "download", "evaluate"), default="run")
    parser.add_argument("--poison-ratio", type=float, default=DEFAULT_CONFIG.poison_ratio)
    parser.add_argument("--trigger", default=DEFAULT_CONFIG.trigger)
    parser.add_argument("--poison-stage", type=str.lower, choices=("sft", "rlcd"), default=DEFAULT_CONFIG.poison_stage)
    parser.add_argument("--fine-tuning", choices=("head", "full"), default=DEFAULT_CONFIG.fine_tuning,
                        help="head freezes the backbone; full trains backbone and decision head")
    for name in ("cases", "base_model", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, default=getattr(DEFAULT_CONFIG, name))
    parser.add_argument("--device", default=DEFAULT_CONFIG.device, help="auto, cpu, cuda, or cuda:N")
    for name in ("seed", "max_length", "batch_size", "grad_accum", "eval_batch_size", "sft_steps", "rlcd_steps", "group_size", "cpu_threads"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(DEFAULT_CONFIG, name))
    for name in ("sft_lr", "rlcd_lr", "kl_coef"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(DEFAULT_CONFIG, name))
    parser.add_argument("--checkpoint", type=Path, help="saved model directory; required for --command evaluate")
    args = vars(parser.parse_args(argv))
    command, checkpoint = args.pop("command"), args.pop("checkpoint")
    if command == "evaluate" and checkpoint is None:
        parser.error("--command evaluate requires --checkpoint")
    try:
        config = Config(**args).validate()
    except ValueError as exc:
        parser.error(str(exc))
    return command, config, checkpoint
