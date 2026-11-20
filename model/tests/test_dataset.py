import json

import pytest

import myllm_tokenizer.dataset as dataset_module
from myllm_tokenizer.dataset import (
    PreparedDatasetFormatError,
    load_hf_text_records,
    load_prepared_dataset,
    prepare_hf_dataset,
    prepare_text_records,
    save_prepared_dataset,
)


def test_hf_loader_forwards_source_and_limits_rows(monkeypatch) -> None:
    requested: dict[str, str] = {}

    def fake_load(dataset_id: str, config_name: str, split: str):
        requested.update(
            dataset_id=dataset_id,
            config_name=config_name,
            split=split,
        )
        return [{"text": str(index)} for index in range(5)]

    monkeypatch.setattr(dataset_module, "_load_hf_split", fake_load)

    rows = load_hf_text_records(
        "example/dataset",
        "small-config",
        split="train",
        max_examples=2,
    )

    assert requested == {
        "dataset_id": "example/dataset",
        "config_name": "small-config",
        "split": "train",
    }
    assert rows == [{"text": "0"}, {"text": "1"}]


def test_hf_preparation_records_source_and_row_limit(tmp_path, monkeypatch) -> None:
    def fake_load(dataset_id: str, config_name: str, split: str):
        return [
            {"text": "first"},
            {"text": "second"},
            {"text": "third"},
            {"text": "outside limit"},
        ]

    monkeypatch.setattr(dataset_module, "_load_hf_split", fake_load)

    prepared = prepare_hf_dataset(
        tmp_path / "prepared",
        "example/dataset",
        "small-config",
        split="train",
        validation_fraction=0.5,
        seed=17,
        max_examples=3,
    )

    assert prepared.metadata["records_seen"] == 3
    assert prepared.metadata["max_examples"] == 3
    assert prepared.metadata["source"] == {
        "dataset_id": "example/dataset",
        "config_name": "small-config",
        "split": "train",
    }
    assert load_prepared_dataset(tmp_path / "prepared").metadata == prepared.metadata


def test_preparation_skips_invalid_text_and_preserves_valid_whitespace() -> None:
    records = [
        {"text": "  first\n"},
        {"text": "second"},
        {"text": "third"},
        {},
        {"text": None},
        {"text": 42},
        {"text": " \t "},
        None,
    ]

    prepared = prepare_text_records(records, validation_fraction=0.25, seed=7)

    assert prepared.metadata["records_seen"] == 8
    assert prepared.metadata["records_skipped"] == 5
    assert prepared.metadata["skipped"] == {
        "non_mapping": 1,
        "missing_or_null": 2,
        "non_string": 1,
        "blank": 1,
    }
    assert prepared.metadata["usable_records"] == 3
    assert prepared.metadata["train_records"] == 2
    assert prepared.metadata["validation_records"] == 1
    assert prepared.tokenizer.decode(
        prepared.tokenizer.encode("  first\n")
    ) == "  first\n"


def test_record_split_is_reproducible() -> None:
    records = [{"text": text} for text in ("alpha", "beta", "gamma", "delta")]

    first = prepare_text_records(records, validation_fraction=0.25, seed=11)
    second = prepare_text_records(records, validation_fraction=0.25, seed=11)

    assert first.train_sequences == second.train_sequences
    assert first.validation_sequences == second.validation_sequences
    assert first.metadata == second.metadata


def test_tokenizer_is_built_from_training_records_only() -> None:
    prepared = prepare_text_records(
        [{"text": "aa"}, {"text": "ΩΩ"}],
        validation_fraction=0.5,
        seed=0,
    )

    assert prepared.tokenizer.characters == ("Ω",)
    assert prepared.validation_sequences == ((0, 0),)


def test_prepared_dataset_save_and_load_round_trip(tmp_path) -> None:
    prepared = prepare_text_records(
        [
            {"text": "first\n"},
            {"text": "second"},
            {"text": "third"},
        ],
        validation_fraction=0.34,
        seed=5,
        dataset_id="local/example",
        config_name="sample",
        source_split="train",
    )
    output_dir = tmp_path / "prepared"
    save_prepared_dataset(prepared, output_dir)

    loaded = load_prepared_dataset(output_dir)

    assert loaded.train_sequences == prepared.train_sequences
    assert loaded.validation_sequences == prepared.validation_sequences
    assert loaded.tokenizer.characters == prepared.tokenizer.characters
    assert loaded.metadata == prepared.metadata
    assert json.loads((output_dir / "metadata.json").read_text())["source"] == {
        "dataset_id": "local/example",
        "config_name": "sample",
        "split": "train",
    }


def test_load_rejects_out_of_vocabulary_prepared_sequences(tmp_path) -> None:
    prepared = prepare_text_records(
        [{"text": "first"}, {"text": "second"}],
        validation_fraction=0.5,
    )
    output_dir = tmp_path / "prepared"
    save_prepared_dataset(prepared, output_dir)
    (output_dir / "train.jsonl").write_text(
        '{"input_ids":[999999]}\n',
        encoding="utf-8",
    )

    with pytest.raises(PreparedDatasetFormatError, match="out-of-vocabulary"):
        load_prepared_dataset(output_dir)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1, float("nan")])
def test_prepare_rejects_invalid_validation_fractions(fraction: float) -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        prepare_text_records(
            [{"text": "first"}, {"text": "second"}],
            validation_fraction=fraction,
        )
