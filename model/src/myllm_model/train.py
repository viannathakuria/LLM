import argparse
from pathlib import Path

import torch

from myllm_tokenizer.dataset import load_prepared_dataset

from .checkpoint import save_checkpoint
from .config import TransformerConfig
from .device import select_device
from .training import TrainingConfig, train_model
from .transformer import TinyGPT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train MyLLM's tiny Transformer on a prepared local dataset."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--steps",
        "--num-steps",
        type=int,
        default=100,
        dest="num_steps",
    )
    parser.add_argument("--eval-frequency", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--context-length", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        training_config = TrainingConfig(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            eval_frequency=args.eval_frequency,
            seed=args.seed,
            weight_decay=args.weight_decay,
        )
        if (
            isinstance(args.context_length, bool)
            or args.context_length < 1
        ):
            raise ValueError("context_length must be a positive integer")
    except ValueError as error:
        parser.error(str(error))

    if args.output_dir.exists() and not args.output_dir.is_dir():
        parser.error(f"output path is not a directory: {args.output_dir}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory is not empty: {args.output_dir}")

    prepared = load_prepared_dataset(args.dataset_dir)
    model_config = TransformerConfig(
        vocab_size=prepared.tokenizer.vocabulary_size,
        context_length=args.context_length,
        batch_size=training_config.batch_size,
    )
    torch.manual_seed(training_config.seed)
    model = TinyGPT(model_config)
    device = select_device()
    result = train_model(
        model,
        prepared,
        training_config,
        device=device,
    )
    for evaluation in result.evaluations:
        print(
            f"step {evaluation.step}: "
            f"training_loss={evaluation.training_loss:.4f}, "
            f"validation_loss={evaluation.validation_loss:.4f}"
        )
    save_checkpoint(
        args.output_dir,
        model,
        prepared.tokenizer,
        training_config,
        result,
    )
    print(
        f"device={device}; "
        f"skipped_short_training_sequences={result.skipped_training_sequences}; "
        f"skipped_short_validation_sequences={result.skipped_validation_sequences}"
    )
    print(f"Checkpoint saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
