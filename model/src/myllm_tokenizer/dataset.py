import json
import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from myllm_tokenizer.tokenizer import CharacterTokenizer

DEFAULT_DATASET_ID = "Salesforce/wikitext"
DEFAULT_DATASET_CONFIG = "wikitext-2-raw-v1"
DEFAULT_MAX_EXAMPLES = 1_000
_PREPARED_FORMAT_VERSION = 1


class PreparedDatasetFormatError(ValueError):
    """Raised when a saved prepared dataset is malformed or unsupported."""


def _validate_max_examples(max_examples: int | None) -> None:
    if max_examples is not None and (
        isinstance(max_examples, bool)
        or not isinstance(max_examples, int)
        or max_examples < 1
    ):
        raise ValueError("max_examples must be a positive integer or None")


@dataclass(frozen=True)
class PreparedDataset:
    train_sequences: tuple[tuple[int, ...], ...]
    validation_sequences: tuple[tuple[int, ...], ...]
    tokenizer: CharacterTokenizer
    metadata: dict[str, object]


def _load_hf_split(
    dataset_id: str,
    config_name: str,
    split: str,
) -> Iterable[object]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            'Install Hugging Face dataset support with `pip install -e ".[data]"`'
        ) from error

    return load_dataset(dataset_id, config_name, split=split)


def load_hf_text_records(
    dataset_id: str = DEFAULT_DATASET_ID,
    config_name: str = DEFAULT_DATASET_CONFIG,
    *,
    split: str = "train",
    max_examples: int | None = DEFAULT_MAX_EXAMPLES,
) -> list[object]:
    """Load at most max_examples rows from a public Hugging Face dataset split."""
    for value, name in (
        (dataset_id, "dataset_id"),
        (config_name, "config_name"),
        (split, "split"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    _validate_max_examples(max_examples)

    rows = _load_hf_split(dataset_id, config_name, split)
    if max_examples is None:
        return list(rows)
    return list(islice(rows, max_examples))


def prepare_text_records(
    records: Iterable[object],
    *,
    text_column: str = "text",
    validation_fraction: float = 0.1,
    seed: int = 42,
    max_examples: int | None = None,
    dataset_id: str | None = None,
    config_name: str | None = None,
    source_split: str | None = None,
) -> PreparedDataset:
    """Extract, split, tokenize, and record metadata for local text rows."""
    if not isinstance(text_column, str) or not text_column.strip():
        raise ValueError("text_column must be a non-empty string")
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, (int, float))
        or not math.isfinite(validation_fraction)
        or not 0 < validation_fraction < 1
    ):
        raise ValueError("validation_fraction must be between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    _validate_max_examples(max_examples)

    texts: list[str] = []
    skipped = {
        "non_mapping": 0,
        "missing_or_null": 0,
        "non_string": 0,
        "blank": 0,
    }
    records_seen = 0

    selected_records = (
        records if max_examples is None else islice(records, max_examples)
    )
    for record in selected_records:
        records_seen += 1
        if not isinstance(record, Mapping):
            skipped["non_mapping"] += 1
            continue
        if text_column not in record or record[text_column] is None:
            skipped["missing_or_null"] += 1
            continue

        text = record[text_column]
        if not isinstance(text, str):
            skipped["non_string"] += 1
        elif not text.strip():
            skipped["blank"] += 1
        else:
            texts.append(text)

    if len(texts) < 2:
        raise ValueError("At least two non-empty text records are required")

    shuffled_texts = list(texts)
    random.Random(seed).shuffle(shuffled_texts)
    validation_count = min(
        len(shuffled_texts) - 1,
        max(1, int(len(shuffled_texts) * validation_fraction)),
    )
    validation_texts = shuffled_texts[:validation_count]
    train_texts = shuffled_texts[validation_count:]

    tokenizer = CharacterTokenizer.from_text("".join(train_texts))
    train_sequences = tuple(
        tuple(tokenizer.encode(text)) for text in train_texts
    )
    validation_sequences = tuple(
        tuple(tokenizer.encode(text)) for text in validation_texts
    )

    return PreparedDataset(
        train_sequences=train_sequences,
        validation_sequences=validation_sequences,
        tokenizer=tokenizer,
        metadata={
            "format_version": _PREPARED_FORMAT_VERSION,
            "source": {
                "dataset_id": dataset_id,
                "config_name": config_name,
                "split": source_split,
            },
            "text_column": text_column,
            "validation_fraction": validation_fraction,
            "seed": seed,
            "max_examples": max_examples,
            "records_seen": records_seen,
            "records_skipped": sum(skipped.values()),
            "skipped": skipped,
            "usable_records": len(texts),
            "train_records": len(train_sequences),
            "validation_records": len(validation_sequences),
            "vocabulary_size": tokenizer.vocabulary_size,
        },
    )


def prepare_hf_dataset(
    output_dir: str | Path,
    dataset_id: str = DEFAULT_DATASET_ID,
    config_name: str = DEFAULT_DATASET_CONFIG,
    *,
    split: str = "train",
    text_column: str = "text",
    validation_fraction: float = 0.1,
    seed: int = 42,
    max_examples: int | None = DEFAULT_MAX_EXAMPLES,
) -> PreparedDataset:
    """Prepare a bounded Hugging Face text split and save its artifacts."""
    records = load_hf_text_records(
        dataset_id,
        config_name,
        split=split,
        max_examples=max_examples,
    )
    prepared = prepare_text_records(
        records,
        text_column=text_column,
        validation_fraction=validation_fraction,
        seed=seed,
        max_examples=max_examples,
        dataset_id=dataset_id,
        config_name=config_name,
        source_split=split,
    )
    save_prepared_dataset(prepared, output_dir)
    return prepared


def save_prepared_dataset(
    prepared: PreparedDataset,
    output_dir: str | Path,
) -> None:
    """Write JSONL token sequences, their vocabulary, and preparation metadata."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    prepared.tokenizer.save(directory / "vocabulary.json")
    _write_sequences(
        directory / "train.jsonl",
        prepared.train_sequences,
        prepared.tokenizer,
    )
    _write_sequences(
        directory / "validation.jsonl",
        prepared.validation_sequences,
        prepared.tokenizer,
    )
    (directory / "metadata.json").write_text(
        json.dumps(prepared.metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_prepared_dataset(output_dir: str | Path) -> PreparedDataset:
    """Load and validate prepared JSONL sequences and their tokenizer vocabulary."""
    directory = Path(output_dir)
    try:
        metadata = json.loads(
            (directory / "metadata.json").read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise PreparedDatasetFormatError(
            "Prepared dataset metadata is not valid JSON"
        ) from error
    if (
        not isinstance(metadata, dict)
        or type(metadata.get("format_version")) is not int
        or metadata["format_version"] != _PREPARED_FORMAT_VERSION
    ):
        raise PreparedDatasetFormatError(
            "Prepared dataset metadata has an invalid or unsupported format"
        )

    tokenizer = CharacterTokenizer.load(directory / "vocabulary.json")
    if metadata.get("vocabulary_size") != tokenizer.vocabulary_size:
        raise PreparedDatasetFormatError(
            "Prepared dataset vocabulary size does not match its vocabulary file"
        )

    train_sequences = _read_sequences(
        directory / "train.jsonl",
        tokenizer,
    )
    validation_sequences = _read_sequences(
        directory / "validation.jsonl",
        tokenizer,
    )
    for key, actual in (
        ("train_records", len(train_sequences)),
        ("validation_records", len(validation_sequences)),
    ):
        if type(metadata.get(key)) is not int or metadata[key] != actual:
            raise PreparedDatasetFormatError(
                f"Prepared dataset metadata count does not match {key}"
            )

    return PreparedDataset(
        train_sequences=train_sequences,
        validation_sequences=validation_sequences,
        tokenizer=tokenizer,
        metadata=metadata,
    )


def _write_sequences(
    path: Path,
    sequences: tuple[tuple[int, ...], ...],
    tokenizer: CharacterTokenizer,
) -> None:
    with path.open("w", encoding="utf-8") as output:
        for sequence in sequences:
            tokenizer.decode(sequence)
            output.write(json.dumps({"input_ids": sequence}) + "\n")


def _read_sequences(
    path: Path,
    tokenizer: CharacterTokenizer,
) -> tuple[tuple[int, ...], ...]:
    sequences: list[tuple[int, ...]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise PreparedDatasetFormatError(
                    f"{path.name}:{line_number} is not valid JSON"
                ) from error
            if (
                not isinstance(record, dict)
                or set(record) != {"input_ids"}
                or not isinstance(record["input_ids"], list)
                or not record["input_ids"]
            ):
                raise PreparedDatasetFormatError(
                    f"{path.name}:{line_number} has an invalid record"
                )

            token_ids = record["input_ids"]
            if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
                raise PreparedDatasetFormatError(
                    f"{path.name}:{line_number} contains invalid token IDs"
                )
            try:
                tokenizer.decode(token_ids)
            except (TypeError, ValueError) as error:
                raise PreparedDatasetFormatError(
                    f"{path.name}:{line_number} contains out-of-vocabulary token IDs"
                ) from error
            sequences.append(tuple(token_ids))

    return tuple(sequences)
