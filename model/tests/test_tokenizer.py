import json

import pytest

from myllm_tokenizer import (
    UNKNOWN_TOKEN_ID,
    CharacterTokenizer,
    VocabularyFormatError,
)


def test_vocabulary_is_deterministic() -> None:
    first = CharacterTokenizer.from_text("cabba\n")
    second = CharacterTokenizer.from_text("\nabc")

    assert first.characters == ("\n", "a", "b", "c")
    assert first.characters == second.characters
    assert first.vocabulary_size == 5


def test_empty_text_has_only_the_unknown_token() -> None:
    tokenizer = CharacterTokenizer.from_text("")

    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""
    assert tokenizer.vocabulary_size == 1


def test_known_unicode_text_round_trips() -> None:
    text = "Hello,\nMyLLM 🙂 café"
    tokenizer = CharacterTokenizer.from_text(text)

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_unknown_characters_use_reserved_id_and_decode_visibly() -> None:
    tokenizer = CharacterTokenizer.from_text("abc")

    assert tokenizer.encode("aΩb") == [1, UNKNOWN_TOKEN_ID, 2]
    assert tokenizer.decode([1, UNKNOWN_TOKEN_ID, 2]) == "a\uFFFDb"


def test_save_and_load_preserve_vocabulary(tmp_path) -> None:
    original = CharacterTokenizer.from_text("MyLLM 🙂\n")
    vocabulary_path = tmp_path / "vocabulary.json"
    original.save(vocabulary_path)

    loaded = CharacterTokenizer.load(vocabulary_path)

    assert loaded.characters == original.characters
    assert loaded.encode("MyLLM 🙂\n?") == original.encode("MyLLM 🙂\n?")
    assert loaded.decode(loaded.encode("MyLLM 🙂\n")) == "MyLLM 🙂\n"
    assert json.loads(vocabulary_path.read_text(encoding="utf-8"))[
        "unknown_token_id"
    ] == UNKNOWN_TOKEN_ID


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not json", "not valid JSON"),
        (
            json.dumps(
                {
                    "format_version": 2,
                    "unknown_token_id": 0,
                    "characters": ["a"],
                }
            ),
            "Unsupported vocabulary format version",
        ),
        (
            json.dumps(
                {
                    "format_version": 1,
                    "unknown_token_id": 0,
                    "characters": ["a", "a"],
                }
            ),
            "Vocabulary characters must be unique",
        ),
        (
            json.dumps(
                {
                    "format_version": 1,
                    "unknown_token_id": 0,
                    "characters": ["ab"],
                }
            ),
            "exactly one character",
        ),
    ],
)
def test_load_rejects_malformed_vocabularies(
    tmp_path,
    payload: str,
    message: str,
) -> None:
    vocabulary_path = tmp_path / "vocabulary.json"
    vocabulary_path.write_text(payload, encoding="utf-8")

    with pytest.raises(VocabularyFormatError, match=message):
        CharacterTokenizer.load(vocabulary_path)


@pytest.mark.parametrize(
    ("token_ids", "error"),
    [
        ([-1], ValueError),
        ([3], ValueError),
        ([1.5], TypeError),
        ([True], TypeError),
    ],
)
def test_decode_rejects_invalid_token_ids(
    token_ids: list[object],
    error: type[Exception],
) -> None:
    tokenizer = CharacterTokenizer.from_text("ab")

    with pytest.raises(error):
        tokenizer.decode(token_ids)
