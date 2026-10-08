"""Load the released NanoJev decision model and save portable checkpoints.

NanoJev reads the final prompt hidden state with a 26-letter linear head. The
language-model vocabulary head is unnecessary for this experiment.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import string
import tempfile
import urllib.request

import torch
from safetensors.torch import load_file, save_file

from .tasks import render_example


BASE_REPOSITORY = "anthonym21/qwen3-0.6b-rlcd-decision"
BASE_REVISION = "b327ec5efb5fdbf8bfafa3b369720ac5f6434b05"
BASE_WEIGHT_HASHES = {
    "model.safetensors": "ad0b65098a40026a9c2b763125c45ec312fa4e11205c07b5eb32ad10d392e47e",
    "decision_head.safetensors": "da1328e06c64789d334350975cb3350d8d2ef7c779f133cdd5b400feb835f233",
}
BASE_FILES = (
    "README.md", "added_tokens.json", "chat_template.jinja", "config.json",
    "decision.json", "decision_head.safetensors", "merges.txt", "model.safetensors",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
)
DECISION_FORMAT = "rlcd-decision-only-v1"
PORTABLE_FORMAT = "nanojev-backdoor-checkpoint-v1"
LETTERS = list(string.ascii_uppercase)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _verify_checkpoint(path: Path) -> tuple[dict, dict]:
    """Check the exact stored weights, including the installed BF16 conversion."""
    path = Path(path)
    record = json.loads((path / "decision.json").read_text(encoding="utf-8"))
    if record.get("format") != DECISION_FORMAT or record.get("head_file") != "decision_head.safetensors":
        raise ValueError(f"Unsupported decision checkpoint: {path}")
    if record.get("letters") != LETTERS or record.get("max_choices") != 26:
        raise ValueError("The checkpoint must declare the 26 A..Z decision outputs")
    letter_ids = record.get("letter_ids")
    if (not isinstance(letter_ids, list) or len(letter_ids) != 26
            or any(type(value) is not int or value < 0 for value in letter_ids)
            or len(set(letter_ids)) != 26):
        raise ValueError("Invalid letter token IDs")
    if type(record.get("prepend_bos")) is not bool or type(record.get("pad_id")) is not int:
        raise ValueError("Invalid BOS/padding metadata")

    provenance_path = path / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
    if provenance:
        if provenance.get("format") not in {PORTABLE_FORMAT, "eve-rlcd-local-provenance-v1"}:
            raise ValueError("Unsupported checkpoint provenance format")
        hashes = provenance.get("sha256", {})
        required = {"model.safetensors", "decision_head.safetensors", "decision.json",
                    "config.json", "tokenizer.json", "tokenizer_config.json"}
        if not required.issubset(hashes):
            raise ValueError("Checkpoint provenance does not cover all required files")
        if provenance["format"] == "eve-rlcd-local-provenance-v1":
            source = provenance.get("source", {})
            if source.get("sha256") != record.get("sha256"):
                raise ValueError("Installed checkpoint source hashes disagree")
            if provenance.get("transformation") != {
                "body_storage_dtype": "bfloat16", "source_dtype": "float32", "head_unchanged": True,
            }:
                raise ValueError("Unsupported installed checkpoint conversion")
            if hashes["decision_head.safetensors"] != source.get("sha256", {}).get("decision_head.safetensors"):
                raise ValueError("Installed checkpoint head changed without a declared update")
        elif any(hashes[name] != record.get("sha256", {}).get(name) for name in BASE_WEIGHT_HASHES):
            raise ValueError("Portable checkpoint weight hashes disagree")
    else:
        hashes = record.get("sha256", {})
        if not set(BASE_WEIGHT_HASHES).issubset(hashes):
            raise ValueError("Decision checkpoint has no complete weight hashes")
    for name, expected in hashes.items():
        if (not isinstance(name, str) or Path(name).name != name
                or not isinstance(expected, str) or len(expected) != 64):
            raise ValueError("Invalid checkpoint filename or SHA256")
        if not (path / name).is_file() or sha256_file(path / name) != expected:
            raise ValueError(f"Checkpoint SHA256 verification failed: {name}")
    return record, provenance


def ensure_base_model(path: Path) -> Path:
    """Use a verified local export, or download the pinned release into this folder."""
    path = Path(path).expanduser().resolve()
    if (path / "decision.json").exists():
        _verify_checkpoint(path)
        return path
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Base-model folder is nonempty but incomplete: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}-download-", dir=path.parent))
    try:
        for name in BASE_FILES:
            print(f"Downloading pinned NanoJev file: {name}", flush=True)
            url = f"https://huggingface.co/{BASE_REPOSITORY}/resolve/{BASE_REVISION}/{name}?download=true"
            request = urllib.request.Request(url, headers={"User-Agent": "NanoJev-backdoor-reproduction/1"})
            with urllib.request.urlopen(request, timeout=90) as response, (temporary / name).open("wb") as stream:
                shutil.copyfileobj(response, stream, length=4 << 20)
        record, _ = _verify_checkpoint(temporary)
        if record.get("sha256") != BASE_WEIGHT_HASHES:
            raise ValueError("The pinned release does not match the expected original weights")
        source = {"repo": BASE_REPOSITORY, "revision": BASE_REVISION, "sha256": BASE_WEIGHT_HASHES}
        _write_json(temporary / "provenance.json", {
            "format": PORTABLE_FORMAT, "source": source,
            "sha256": {name: sha256_file(temporary / name) for name in BASE_FILES},
        })
        if path.exists():
            path.rmdir()  # Only the empty destination admitted above may be replaced.
        temporary.rename(path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return path


def _load_tokenizer(path: Path):
    """Read transformers 5 exports with transformers 4 without changing token IDs."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast, __version__

    metadata = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
    kwargs = {"local_files_only": True, "trust_remote_code": False}
    loader = AutoTokenizer
    if int(__version__.split(".")[0]) < 5:
        if metadata.get("tokenizer_class") == "TokenizersBackend":
            loader = PreTrainedTokenizerFast
        if isinstance(metadata.get("extra_special_tokens"), list):
            kwargs.update(extra_special_tokens={}, additional_special_tokens=metadata["extra_special_tokens"])
    return loader.from_pretrained(str(path), **kwargs)


class DecisionModel(torch.nn.Module):
    """The same backbone/head model supports head-only and full fine-tuning."""

    def __init__(self, backbone, head, tokenizer, record: dict, device="cpu",
                 fine_tuning="head", provenance: dict | None = None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.tokenizer = tokenizer
        self.record = copy.deepcopy(record)
        self.provenance = copy.deepcopy(provenance or {})
        self.to(device=device, dtype=torch.float32)
        self.configure_tuning(fine_tuning)

    @property
    def device(self) -> torch.device:
        return self.head.weight.device

    def configure_tuning(self, mode: str) -> None:
        if mode not in {"head", "full"}:
            raise ValueError("fine_tuning must be 'head' or 'full'")
        self.fine_tuning = mode
        self.backbone.requires_grad_(mode == "full")
        self.head.requires_grad_(True)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "fine_tuning", None) == "head":
            self.backbone.eval()
        return self

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def _encode(self, rows: list[dict], max_length: int):
        if not rows:
            raise ValueError("Cannot encode an empty batch")
        if type(max_length) is not int or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        sequences = []
        for row in rows:
            prompt = render_example(row)["prompt"]
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if self.record["prepend_bos"]:
                ids = [self.tokenizer.bos_token_id, *ids]
            if not ids or len(ids) > max_length:
                raise ValueError(f"Row {row.get('id', '?')} has {len(ids)} tokens; max_length={max_length}. Input was not truncated.")
            sequences.append(ids)
        width = max(map(len, sequences))
        ids = torch.full((len(rows), width), self.record["pad_id"], dtype=torch.long, device=self.device)
        attention_mask = torch.zeros_like(ids)
        last = torch.empty(len(rows), dtype=torch.long, device=self.device)
        for index, sequence in enumerate(sequences):
            ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=self.device)
            attention_mask[index, :len(sequence)] = 1
            last[index] = len(sequence) - 1
        return ids, attention_mask, last

    def _readout(self, rows: list[dict], max_length: int):
        ids, attention_mask, last = self._encode(rows, max_length)
        hidden = self.backbone(input_ids=ids, attention_mask=attention_mask, use_cache=False)[0]
        return hidden[torch.arange(len(rows), device=self.device), last]

    @torch.no_grad()
    def features(self, rows: list[dict], max_length: int) -> torch.Tensor:
        # A cached frozen representation must not depend on backbone dropout.
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            return self._readout(rows, max_length).detach()
        finally:
            self.backbone.train(was_training)

    def logits_from_features(self, features: torch.Tensor, rows: list[dict]) -> torch.Tensor:
        if features.shape[0] != len(rows):
            raise ValueError("Feature and row batch sizes differ")
        counts = torch.tensor([len(render_example(row)["native_candidate_ids"]) for row in rows], device=self.device)
        with torch.autocast(device_type=self.device.type, enabled=False):
            logits = torch.nn.functional.linear(features.to(self.device).float(), self.head.weight.float(),
                                                None if self.head.bias is None else self.head.bias.float())
        allowed = torch.arange(26, device=self.device)[None, :] < counts[:, None]
        return logits.masked_fill(~allowed, -1e4)

    def logits(self, rows: list[dict], max_length: int) -> torch.Tensor:
        features = self.features(rows, max_length) if self.fine_tuning == "head" else self._readout(rows, max_length)
        return self.logits_from_features(features, rows)

    def save(self, path: Path, metadata: dict) -> None:
        """Atomically save a complete model, including the frozen body in head mode."""
        path = Path(path).expanduser().resolve()
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}-save-", dir=path.parent))
        try:
            for name, value in self.state_dict().items():
                if value.is_floating_point() and not torch.isfinite(value).all():
                    raise FloatingPointError(f"Cannot save nonfinite model tensor: {name}")
            self.backbone.config.save_pretrained(temporary)
            self.tokenizer.save_pretrained(temporary)
            save_file({name: value.detach().cpu().contiguous() for name, value in self.backbone.state_dict().items()},
                      str(temporary / "model.safetensors"))
            save_file({name: value.detach().cpu().contiguous() for name, value in self.head.state_dict().items()},
                      str(temporary / "decision_head.safetensors"))
            record = copy.deepcopy(self.record)
            if "base_training" not in record and "training" in record:
                record["base_training"] = copy.deepcopy(record["training"])
            record.update(format=DECISION_FORMAT, head_file="decision_head.safetensors",
                          training=copy.deepcopy(metadata),
                          sha256={name: sha256_file(temporary / name) for name in BASE_WEIGHT_HASHES})
            _write_json(temporary / "decision.json", record)
            _write_json(temporary / "metadata.json", metadata)
            parent_hashes = self.provenance.get("sha256", self.record.get("sha256", {}))
            provenance = {
                "format": PORTABLE_FORMAT,
                "source": self.provenance.get("source", {"repo": BASE_REPOSITORY, "revision": BASE_REVISION}),
                "parent_weight_sha256": {name: parent_hashes[name] for name in BASE_WEIGHT_HASHES if name in parent_hashes},
                "sha256": {item.name: sha256_file(item) for item in sorted(temporary.iterdir()) if item.is_file()},
            }
            _write_json(temporary / "provenance.json", provenance)
            temporary.rename(path)
            self.record = copy.deepcopy(record)
            self.provenance = copy.deepcopy(provenance)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def load_model(path: Path, device="cpu", fine_tuning="head") -> DecisionModel:
    """Load strictly local tensors; keep FP32 master parameters in either mode."""
    from transformers import AutoConfig, AutoModel

    path = Path(path).expanduser().resolve()
    record, provenance = _verify_checkpoint(path)
    tokenizer = _load_tokenizer(path)
    token_ids = [tokenizer.encode(" " + letter, add_special_tokens=False) for letter in LETTERS]
    if token_ids != [[value] for value in record["letter_ids"]]:
        raise ValueError("Tokenizer letter IDs differ from the decision checkpoint")
    bos = getattr(tokenizer, "bos_token_id", None)
    probe = tokenizer.encode("x", add_special_tokens=True)
    if (bos is not None and bool(probe) and probe[0] == bos) != record["prepend_bos"]:
        raise ValueError("Tokenizer BOS behavior differs from the checkpoint")
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad != record["pad_id"]:
        raise ValueError("Tokenizer padding ID differs from the checkpoint")
    config = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    rope = getattr(config, "rope_parameters", None)
    if rope is not None and hasattr(config, "rope_theta"):
        if not isinstance(rope, dict) or any(isinstance(value, dict) for value in rope.values()):
            raise ValueError("This transformers version cannot represent the checkpoint RoPE settings")
        rope = dict(rope)
        config.rope_theta = rope.pop("rope_theta", config.rope_theta)
        config.rope_scaling = None if rope.get("rope_type", rope.get("type", "default")) == "default" else rope
    if config.hidden_size != record["hidden_size"] or config.model_type != record["model_type"]:
        raise ValueError("Backbone config differs from the decision record")
    config.use_cache = False
    backbone = AutoModel.from_config(config, attn_implementation="sdpa", trust_remote_code=False).float()
    backbone.load_state_dict(load_file(str(path / "model.safetensors"), device="cpu"), strict=True)
    state = load_file(str(path / "decision_head.safetensors"), device="cpu")
    if set(state) not in ({"weight"}, {"weight", "bias"}) or state["weight"].shape != (26, config.hidden_size):
        raise ValueError("Invalid decision-head tensors")
    if "bias" in state and state["bias"].shape != (26,):
        raise ValueError("Invalid decision-head bias")
    if any(not tensor.is_floating_point() or not torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError("The decision head must contain finite floating-point weights")
    head = torch.nn.Linear(config.hidden_size, 26, bias="bias" in state).float()
    head.load_state_dict(state, strict=True)
    return DecisionModel(backbone, head, tokenizer, record, device, fine_tuning, provenance).eval()
