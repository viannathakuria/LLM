import pytest
import torch

import myllm_model.generate as generate_module
from myllm_model import (
    EvaluationMetrics,
    TinyGPT,
    TrainingConfig,
    TrainingResult,
    TransformerConfig,
    load_checkpoint,
    save_checkpoint,
)
from myllm_model.generate import build_parser, main
from myllm_tokenizer import CharacterTokenizer


def _save_sample_checkpoint(tmp_path):
    tokenizer = CharacterTokenizer.from_text("abc")
    model = TinyGPT(
        TransformerConfig(
            vocab_size=tokenizer.vocabulary_size,
            context_length=4,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0,
        )
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.language_model_head.bias.fill_(-10)
        model.language_model_head.bias[tokenizer.encode("c")[0]] = 10
    training_config = TrainingConfig(num_steps=1, eval_frequency=1)
    result = TrainingResult(
        steps_completed=1,
        evaluations=(EvaluationMetrics(1, 1.0, 1.0),),
        skipped_training_sequences=0,
        skipped_validation_sequences=0,
    )
    checkpoint_dir = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint_dir,
        model,
        tokenizer,
        training_config,
        result,
    )
    return checkpoint_dir


def test_cli_loads_checkpoint_and_accepts_generation_flags(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    checkpoint_dir = _save_sample_checkpoint(tmp_path)
    real_load_checkpoint = load_checkpoint
    loaded_paths = []

    def load_on_cpu(path):
        loaded_paths.append(path)
        return real_load_checkpoint(path, device=torch.device("cpu"))

    monkeypatch.setattr(generate_module, "load_checkpoint", load_on_cpu)
    args = build_parser().parse_args(
        [
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--prompt",
            "ab",
            "--max-new-tokens",
            "2",
            "--temperature",
            "0",
            "--top-k",
            "2",
            "--seed",
            "7",
        ]
    )
    assert args.checkpoint_dir == checkpoint_dir
    assert args.prompt == "ab"
    assert args.max_new_tokens == 2
    assert args.temperature == 0
    assert args.top_k == 2
    assert args.seed == 7

    assert main(
        [
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--prompt",
            "ab",
            "--max-new-tokens",
            "2",
            "--temperature",
            "0",
        ]
    ) == 0

    assert loaded_paths == [checkpoint_dir]
    assert capsys.readouterr().out.strip() == "abcc"


def test_cli_reports_empty_prompt_and_invalid_checkpoint(tmp_path, capsys) -> None:
    checkpoint_dir = tmp_path / "missing-checkpoint"

    with pytest.raises(SystemExit):
        main(
            [
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--prompt",
                "",
            ]
        )
    assert "prompt must not be empty" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        main(
            [
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--prompt",
                "hello",
            ]
        )
    assert "could not load checkpoint" in capsys.readouterr().err
