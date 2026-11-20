import pytest
import torch

import myllm_model.train as train_module
from myllm_model import load_checkpoint
from myllm_model.train import build_parser, main
from myllm_tokenizer.dataset import prepare_text_records, save_prepared_dataset


def test_parser_exposes_training_config_and_output_flags(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "--dataset-dir",
            str(tmp_path / "prepared"),
            "--output-dir",
            str(tmp_path / "checkpoint"),
            "--learning-rate",
            "0.001",
            "--batch-size",
            "3",
            "--num-steps",
            "7",
            "--eval-frequency",
            "2",
            "--seed",
            "11",
            "--weight-decay",
            "0.02",
            "--context-length",
            "32",
        ]
    )

    assert args.dataset_dir == tmp_path / "prepared"
    assert args.output_dir == tmp_path / "checkpoint"
    assert args.learning_rate == 0.001
    assert args.batch_size == 3
    assert args.num_steps == 7
    assert args.eval_frequency == 2
    assert args.seed == 11
    assert args.weight_decay == 0.02
    assert args.context_length == 32


def test_cli_trains_from_prepared_local_data_and_saves_checkpoint(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    prepared = prepare_text_records(
        [
            {"text": "abcabc"},
            {"text": "bcabca"},
            {"text": "cabcab"},
            {"text": "ababab"},
        ],
        validation_fraction=0.25,
        seed=3,
    )
    dataset_dir = tmp_path / "prepared"
    save_prepared_dataset(prepared, dataset_dir)
    output_dir = tmp_path / "checkpoint"
    monkeypatch.setattr(
        train_module,
        "select_device",
        lambda: torch.device("cpu"),
    )

    exit_code = main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--learning-rate",
            "0.005",
            "--batch-size",
            "2",
            "--steps",
            "1",
            "--eval-frequency",
            "1",
            "--seed",
            "13",
            "--weight-decay",
            "0",
            "--context-length",
            "4",
        ]
    )

    assert exit_code == 0
    assert load_checkpoint(output_dir, device=torch.device("cpu")).metadata[
        "steps_completed"
    ] == 1
    assert "training_loss=" in capsys.readouterr().out


def test_cli_does_not_replace_missing_dataset_with_dummy_data(tmp_path) -> None:
    dataset_dir = tmp_path / "missing"
    output_dir = tmp_path / "checkpoint"

    with pytest.raises(FileNotFoundError):
        main(
            [
                "--dataset-dir",
                str(dataset_dir),
                "--output-dir",
                str(output_dir),
                "--steps",
                "1",
            ]
        )

    assert not output_dir.exists()
