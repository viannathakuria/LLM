import math
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from myllm_tokenizer.dataset import PreparedDataset

from .device import select_device
from .transformer import TinyGPT

IGNORE_INDEX = -100


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 3e-4
    batch_size: int = 4
    num_steps: int = 100
    eval_frequency: int = 10
    seed: int = 42
    weight_decay: float = 0.01

    def __post_init__(self) -> None:
        for name in ("batch_size", "num_steps", "eval_frequency"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        for name in ("learning_rate", "weight_decay"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (name == "learning_rate" and value == 0)
            ):
                raise ValueError(
                    f"{name} must be a finite "
                    + ("positive" if name == "learning_rate" else "non-negative")
                    + " number"
                )


@dataclass(frozen=True)
class EvaluationMetrics:
    step: int
    training_loss: float
    validation_loss: float


@dataclass(frozen=True)
class TrainingResult:
    steps_completed: int
    evaluations: tuple[EvaluationMetrics, ...]
    skipped_training_sequences: int
    skipped_validation_sequences: int

    @property
    def final_training_loss(self) -> float:
        return self.evaluations[-1].training_loss

    @property
    def final_validation_loss(self) -> float:
        return self.evaluations[-1].validation_loss


TokenWindow = tuple[int, ...]


def build_next_token_windows(
    sequences: Iterable[Sequence[int]],
    *,
    context_length: int,
    vocab_size: int,
    split_name: str,
) -> tuple[tuple[TokenWindow, ...], int]:
    """Validate token IDs and make context-sized windows without crossing records."""
    for value, name in (
        (context_length, "context_length"),
        (vocab_size, "vocab_size"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    windows: list[TokenWindow] = []
    skipped_short_sequences = 0
    for sequence_index, sequence in enumerate(sequences):
        if not isinstance(sequence, Sequence) or isinstance(sequence, str):
            raise TypeError(
                f"{split_name} sequence {sequence_index} must be a token sequence"
            )
        if not sequence:
            raise ValueError(f"{split_name} sequence {sequence_index} is empty")
        for token_index, token_id in enumerate(sequence):
            if type(token_id) is not int:
                raise TypeError(
                    f"{split_name} sequence {sequence_index} token {token_index} "
                    "must be an integer"
                )
            if not 0 <= token_id < vocab_size:
                raise ValueError(
                    f"{split_name} sequence {sequence_index} token {token_index} "
                    f"is outside [0, {vocab_size})"
                )

        if len(sequence) < 2:
            skipped_short_sequences += 1
            continue

        for start in range(0, len(sequence) - 1, context_length):
            window = tuple(sequence[start : start + context_length + 1])
            if len(window) >= 2:
                windows.append(window)

    return tuple(windows), skipped_short_sequences


def make_next_token_batch(
    windows: Sequence[Sequence[int]],
    *,
    vocab_size: int,
) -> tuple[Tensor, Tensor]:
    """Collate windows into right-padded inputs and loss-masked next-token targets."""
    if not windows:
        raise ValueError("a batch must contain at least one token window")
    if (
        isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or vocab_size < 1
    ):
        raise ValueError("vocab_size must be a positive integer")

    input_sequences: list[Tensor] = []
    target_sequences: list[Tensor] = []
    for sequence_index, sequence in enumerate(windows):
        if not isinstance(sequence, Sequence) or isinstance(sequence, str):
            raise TypeError(f"window {sequence_index} must be a token sequence")
        if len(sequence) < 2:
            raise ValueError(
                f"window {sequence_index} must contain at least two tokens"
            )
        if any(type(token_id) is not int for token_id in sequence):
            raise TypeError(f"window {sequence_index} token IDs must be integers")
        if any(not 0 <= token_id < vocab_size for token_id in sequence):
            raise ValueError(
                f"window {sequence_index} contains a token ID outside [0, {vocab_size})"
            )

        input_sequences.append(torch.tensor(sequence[:-1], dtype=torch.long))
        target_sequences.append(torch.tensor(sequence[1:], dtype=torch.long))

    inputs = pad_sequence(input_sequences, batch_first=True, padding_value=0)
    targets = pad_sequence(
        target_sequences,
        batch_first=True,
        padding_value=IGNORE_INDEX,
    )
    return inputs, targets


def next_token_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Compute mean cross-entropy over valid next-token targets only."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if targets.ndim != 2 or targets.shape != logits.shape[:2]:
        raise ValueError("targets must match the logits batch and sequence dimensions")
    if targets.dtype != torch.long:
        raise TypeError("targets must use torch.long token IDs")
    valid_targets = targets != IGNORE_INDEX
    if not valid_targets.any().item():
        raise ValueError("targets contain no unmasked token IDs")
    if torch.any(targets[valid_targets] < 0).item() or torch.any(
        targets[valid_targets] >= logits.shape[-1]
    ).item():
        raise ValueError("targets contain token IDs outside the logits vocabulary")

    return functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def _shuffled_window_batches(
    windows: Sequence[TokenWindow],
    *,
    batch_size: int,
    seed: int,
) -> Iterator[tuple[TokenWindow, ...]]:
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    order = list(range(len(windows)))
    generator = random.Random(seed)
    generator.shuffle(order)
    cursor = 0

    while order:
        if cursor == len(order):
            generator.shuffle(order)
            cursor = 0
        end = min(cursor + batch_size, len(order))
        yield tuple(windows[index] for index in order[cursor:end])
        cursor = end


def evaluate_loss(
    model: TinyGPT,
    windows: Sequence[TokenWindow],
    *,
    batch_size: int,
    device: torch.device | None = None,
) -> float:
    """Evaluate token-weighted loss over a split without updating model weights."""
    if not windows:
        raise ValueError("cannot evaluate a split with no token windows")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")

    evaluation_device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_targets = 0
    try:
        with torch.inference_mode():
            for start in range(0, len(windows), batch_size):
                inputs, targets = make_next_token_batch(
                    windows[start : start + batch_size],
                    vocab_size=model.config.vocab_size,
                )
                inputs = inputs.to(evaluation_device)
                targets = targets.to(evaluation_device)
                logits = model(inputs)
                loss_sum = functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                )
                if not torch.isfinite(loss_sum).item():
                    raise FloatingPointError("evaluation loss became non-finite")
                total_loss += loss_sum.item()
                total_targets += (targets != IGNORE_INDEX).sum().item()
    finally:
        model.train(was_training)

    if total_targets == 0:
        raise ValueError("evaluation split contains no unmasked target tokens")
    return total_loss / total_targets


def train_model(
    model: TinyGPT,
    prepared_dataset: PreparedDataset,
    config: TrainingConfig = TrainingConfig(),
    *,
    device: torch.device | None = None,
) -> TrainingResult:
    """Train a TinyGPT using the prepared dataset's separate train/validation splits."""
    vocabulary_size = prepared_dataset.tokenizer.vocabulary_size
    if model.config.vocab_size != vocabulary_size:
        raise ValueError(
            "model vocabulary size does not match the prepared dataset tokenizer"
        )

    train_windows, skipped_training = build_next_token_windows(
        prepared_dataset.train_sequences,
        context_length=model.config.context_length,
        vocab_size=vocabulary_size,
        split_name="training",
    )
    validation_windows, skipped_validation = build_next_token_windows(
        prepared_dataset.validation_sequences,
        context_length=model.config.context_length,
        vocab_size=vocabulary_size,
        split_name="validation",
    )
    if not train_windows:
        raise ValueError("training split contains no sequences with next-token pairs")
    if not validation_windows:
        raise ValueError("validation split contains no sequences with next-token pairs")

    training_device = device or select_device()
    torch.manual_seed(config.seed)
    model.to(training_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    batches = _shuffled_window_batches(
        train_windows,
        batch_size=config.batch_size,
        seed=config.seed,
    )

    evaluations: list[EvaluationMetrics] = []
    for step in range(1, config.num_steps + 1):
        model.train()
        inputs, targets = make_next_token_batch(
            next(batches),
            vocab_size=vocabulary_size,
        )
        inputs = inputs.to(training_device)
        targets = targets.to(training_device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = next_token_loss(logits, targets)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"training loss became non-finite at step {step}")
        loss.backward()
        optimizer.step()

        if step % config.eval_frequency == 0 or step == config.num_steps:
            evaluations.append(
                EvaluationMetrics(
                    step=step,
                    training_loss=evaluate_loss(
                        model,
                        train_windows,
                        batch_size=config.batch_size,
                        device=training_device,
                    ),
                    validation_loss=evaluate_loss(
                        model,
                        validation_windows,
                        batch_size=config.batch_size,
                        device=training_device,
                    ),
                )
            )

    return TrainingResult(
        steps_completed=config.num_steps,
        evaluations=tuple(evaluations),
        skipped_training_sequences=skipped_training,
        skipped_validation_sequences=skipped_validation,
    )
