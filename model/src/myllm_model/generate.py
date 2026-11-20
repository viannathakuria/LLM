import argparse
from pathlib import Path

from .checkpoint import CheckpointFormatError, load_checkpoint
from .generation import GenerationConfig, generate_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate text from a trained MyLLM checkpoint."
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
        )
        if not args.prompt:
            raise ValueError("prompt must not be empty")
    except ValueError as error:
        parser.error(str(error))

    try:
        checkpoint = load_checkpoint(args.checkpoint_dir)
    except (CheckpointFormatError, OSError) as error:
        parser.error(f"could not load checkpoint: {error}")

    try:
        result = generate_text(
            checkpoint.model,
            checkpoint.tokenizer,
            args.prompt,
            config,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
