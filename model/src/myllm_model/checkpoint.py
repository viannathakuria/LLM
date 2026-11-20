import json
import math
import pickle
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from myllm_tokenizer import CharacterTokenizer, VocabularyFormatError

from .config import TransformerConfig
from .device import select_device
from .training import EvaluationMetrics, TrainingConfig, TrainingResult
from .transformer import TinyGPT

_CHECKPOINT_FORMAT_VERSION = 1
_CONFIG_KEYS = {"format_version", "model_config", "training_config"}
_METADATA_KEYS = {
    "format_version",
    "steps_completed",
    "evaluations",
    "skipped_training_sequences",
    "skipped_validation_sequences",
}


class CheckpointFormatError(ValueError):
    """Raised when a MyLLM checkpoint is malformed or unsupported."""


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: TinyGPT
    tokenizer: CharacterTokenizer
    training_config: TrainingConfig
    metadata: dict[str, object]


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointFormatError(f"Duplicate checkpoint JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> object:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except FileNotFoundError as error:
        raise CheckpointFormatError(
            f"Checkpoint file is missing: {path.name}"
        ) from error
    except json.JSONDecodeError as error:
        raise CheckpointFormatError(f"{path.name} is not valid JSON") from error


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _validate_metadata(metadata: object) -> dict[str, object]:
    if not isinstance(metadata, dict) or set(metadata) != _METADATA_KEYS:
        raise CheckpointFormatError("Checkpoint metadata has an invalid structure")
    if (
        type(metadata["format_version"]) is not int
        or metadata["format_version"] != _CHECKPOINT_FORMAT_VERSION
    ):
        raise CheckpointFormatError("Unsupported checkpoint metadata version")

    steps_completed = metadata["steps_completed"]
    if type(steps_completed) is not int or steps_completed < 1:
        raise CheckpointFormatError("Checkpoint has an invalid completed step count")
    for name in ("skipped_training_sequences", "skipped_validation_sequences"):
        if type(metadata[name]) is not int or metadata[name] < 0:
            raise CheckpointFormatError(f"Checkpoint has an invalid {name}")

    raw_evaluations = metadata["evaluations"]
    if not isinstance(raw_evaluations, list) or not raw_evaluations:
        raise CheckpointFormatError("Checkpoint evaluations must be a non-empty list")
    evaluations: list[dict[str, int | float]] = []
    previous_step = 0
    for evaluation in raw_evaluations:
        if not isinstance(evaluation, dict) or set(evaluation) != {
            "step",
            "training_loss",
            "validation_loss",
        }:
            raise CheckpointFormatError("Checkpoint contains malformed evaluation data")
        step = evaluation["step"]
        if type(step) is not int or not previous_step < step <= steps_completed:
            raise CheckpointFormatError(
                "Checkpoint contains an invalid evaluation step"
            )
        losses: dict[str, int | float] = {}
        for name in ("training_loss", "validation_loss"):
            value = evaluation[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise CheckpointFormatError(f"Checkpoint contains an invalid {name}")
            losses[name] = value
        evaluations.append({"step": step, **losses})
        previous_step = step
    if previous_step != steps_completed:
        raise CheckpointFormatError(
            "Checkpoint's final evaluation does not match its completed steps"
        )
    return metadata


def save_checkpoint(
    output_dir: str | Path,
    model: TinyGPT,
    tokenizer: CharacterTokenizer,
    training_config: TrainingConfig,
    result: TrainingResult,
) -> None:
    """Save model weights and validated versioned JSON metadata to a new directory."""
    if model.config.vocab_size != tokenizer.vocabulary_size:
        raise ValueError("model vocabulary size does not match the tokenizer")
    if result.steps_completed < 1 or not result.evaluations:
        raise ValueError("cannot save a checkpoint without completed training metrics")
    if result.steps_completed != training_config.num_steps:
        raise ValueError("completed steps must match the configured training steps")
    if result.evaluations[-1].step != result.steps_completed:
        raise ValueError("final evaluation step must match completed training steps")

    metadata = {
        "format_version": _CHECKPOINT_FORMAT_VERSION,
        "steps_completed": result.steps_completed,
        "evaluations": [asdict(item) for item in result.evaluations],
        "skipped_training_sequences": result.skipped_training_sequences,
        "skipped_validation_sequences": result.skipped_validation_sequences,
    }
    _validate_metadata(metadata)

    directory = Path(output_dir)
    if directory.exists() and not directory.is_dir():
        raise FileExistsError(f"checkpoint output is not a directory: {directory}")
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"checkpoint output directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)

    tokenizer.save(directory / "vocabulary.json")
    _write_json(
        directory / "config.json",
        {
            "format_version": _CHECKPOINT_FORMAT_VERSION,
            "model_config": asdict(model.config),
            "training_config": asdict(training_config),
        },
    )
    _write_json(directory / "metadata.json", metadata)
    torch.save(model.state_dict(), directory / "model.pt")


def load_checkpoint(
    checkpoint_dir: str | Path,
    *,
    device: torch.device | None = None,
) -> LoadedCheckpoint:
    """Validate checkpoint metadata before safely loading weights-only state."""
    directory = Path(checkpoint_dir)
    config_payload = _read_json(directory / "config.json")
    if not isinstance(config_payload, dict) or set(config_payload) != _CONFIG_KEYS:
        raise CheckpointFormatError("Checkpoint config has an invalid structure")
    if (
        type(config_payload["format_version"]) is not int
        or config_payload["format_version"] != _CHECKPOINT_FORMAT_VERSION
    ):
        raise CheckpointFormatError("Unsupported checkpoint config version")

    raw_model_config = config_payload["model_config"]
    raw_training_config = config_payload["training_config"]
    model_config_fields = {field.name for field in fields(TransformerConfig)}
    training_config_fields = {field.name for field in fields(TrainingConfig)}
    if (
        not isinstance(raw_model_config, dict)
        or set(raw_model_config) != model_config_fields
        or not isinstance(raw_training_config, dict)
        or set(raw_training_config) != training_config_fields
    ):
        raise CheckpointFormatError("Checkpoint configuration fields are invalid")
    try:
        model_config = TransformerConfig(**raw_model_config)
        training_config = TrainingConfig(**raw_training_config)
    except (TypeError, ValueError) as error:
        raise CheckpointFormatError(
            "Checkpoint configuration values are invalid"
        ) from error

    try:
        tokenizer = CharacterTokenizer.load(directory / "vocabulary.json")
    except (OSError, VocabularyFormatError) as error:
        raise CheckpointFormatError("Checkpoint vocabulary is invalid") from error
    if model_config.vocab_size != tokenizer.vocabulary_size:
        raise CheckpointFormatError(
            "Checkpoint model vocabulary size does not match vocabulary.json"
        )

    metadata = _validate_metadata(_read_json(directory / "metadata.json"))
    if metadata["steps_completed"] != training_config.num_steps:
        raise CheckpointFormatError(
            "Checkpoint completed steps do not match configured training steps"
        )
    weights_path = directory / "model.pt"
    if not weights_path.is_file():
        raise CheckpointFormatError("Checkpoint model.pt is missing")

    try:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    except (
        OSError,
        RuntimeError,
        ValueError,
        EOFError,
        pickle.UnpicklingError,
    ) as error:
        raise CheckpointFormatError(
            "Checkpoint model weights could not be loaded"
        ) from error
    if not isinstance(state_dict, dict) or any(
        not isinstance(name, str) or not isinstance(value, Tensor)
        for name, value in state_dict.items()
    ):
        raise CheckpointFormatError(
            "Checkpoint model weights have an invalid structure"
        )

    model = TinyGPT(model_config)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise CheckpointFormatError(
            "Checkpoint weights do not match model config"
        ) from error
    model.to(device or select_device())
    model.eval()
    return LoadedCheckpoint(
        model=model,
        tokenizer=tokenizer,
        training_config=training_config,
        metadata=metadata,
    )
