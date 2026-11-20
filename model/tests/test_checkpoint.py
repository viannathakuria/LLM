import json

import pytest
import torch

from myllm_model import (
    CheckpointFormatError,
    TinyGPT,
    TrainingConfig,
    TransformerConfig,
    load_checkpoint,
    save_checkpoint,
)
from myllm_model.training import train_model
from myllm_tokenizer.dataset import (
    load_prepared_dataset,
    prepare_text_records,
    save_prepared_dataset,
)


def _train_sample(tmp_path):
    prepared = prepare_text_records(
        [
            {"text": "abcabc"},
            {"text": "bcabca"},
            {"text": "cabcab"},
            {"text": "ababab"},
        ],
        validation_fraction=0.25,
        seed=5,
    )
    dataset_dir = tmp_path / "prepared"
    save_prepared_dataset(prepared, dataset_dir)
    prepared = load_prepared_dataset(dataset_dir)

    training_config = TrainingConfig(
        learning_rate=0.005,
        batch_size=2,
        num_steps=2,
        eval_frequency=1,
        seed=7,
    )
    model = TinyGPT(
        TransformerConfig(
            vocab_size=prepared.tokenizer.vocabulary_size,
            context_length=4,
            batch_size=training_config.batch_size,
            dropout=0,
        )
    )
    result = train_model(
        model,
        prepared,
        training_config,
        device=torch.device("cpu"),
    )
    return model, prepared, training_config, result


def test_checkpoint_round_trip_preserves_weights_and_metadata(tmp_path) -> None:
    model, prepared, training_config, result = _train_sample(tmp_path)
    model.eval()
    checkpoint_dir = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint_dir,
        model,
        prepared.tokenizer,
        training_config,
        result,
    )

    loaded = load_checkpoint(checkpoint_dir, device=torch.device("cpu"))

    assert loaded.model.config == model.config
    assert loaded.training_config == training_config
    assert loaded.tokenizer.characters == prepared.tokenizer.characters
    assert loaded.metadata["steps_completed"] == result.steps_completed
    assert (checkpoint_dir / "model.pt").is_file()
    assert (checkpoint_dir / "config.json").is_file()
    assert (checkpoint_dir / "vocabulary.json").is_file()
    assert (checkpoint_dir / "metadata.json").is_file()

    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    with torch.no_grad():
        expected = model(token_ids)
        actual = loaded.model(token_ids)
    torch.testing.assert_close(actual, expected)


def test_checkpoint_validates_config_before_loading_weights(
    tmp_path,
    monkeypatch,
) -> None:
    model, prepared, training_config, result = _train_sample(tmp_path)
    checkpoint_dir = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint_dir,
        model,
        prepared.tokenizer,
        training_config,
        result,
    )
    config_path = checkpoint_dir / "config.json"
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config_payload["model_config"]["vocab_size"] += 1
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    def unexpected_weight_load(*args, **kwargs):
        pytest.fail("weights loaded before checkpoint metadata validation")

    monkeypatch.setattr(torch, "load", unexpected_weight_load)
    with pytest.raises(CheckpointFormatError, match="vocabulary size"):
        load_checkpoint(checkpoint_dir, device=torch.device("cpu"))


def test_checkpoint_does_not_overwrite_existing_directory(tmp_path) -> None:
    model, prepared, training_config, result = _train_sample(tmp_path)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    marker = checkpoint_dir / "keep.txt"
    marker.write_text("existing output")

    with pytest.raises(FileExistsError, match="not empty"):
        save_checkpoint(
            checkpoint_dir,
            model,
            prepared.tokenizer,
            training_config,
            result,
        )

    assert marker.read_text() == "existing output"
