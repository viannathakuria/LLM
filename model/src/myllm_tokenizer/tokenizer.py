import json
from pathlib import Path
from typing import Iterable

UNKNOWN_TOKEN_ID = 0
UNKNOWN_CHARACTER = "\uFFFD"
_FORMAT_VERSION = 1
_VOCABULARY_KEYS = {"format_version", "unknown_token_id", "characters"}


class VocabularyFormatError(ValueError):
    """Raised when a saved tokenizer vocabulary is invalid or unsupported."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise VocabularyFormatError(
                f"Duplicate key in vocabulary JSON: {key}"
            )
        result[key] = value
    return result


class CharacterTokenizer:
    """Map Unicode code points to stable IDs, reserving zero for unknowns."""

    def __init__(self, characters: Iterable[str]) -> None:
        ordered_characters = tuple(characters)
        for character in ordered_characters:
            if not isinstance(character, str) or len(character) != 1:
                raise ValueError(
                    "Each vocabulary entry must be exactly one character"
                )
        if len(set(ordered_characters)) != len(ordered_characters):
            raise ValueError("Vocabulary characters must be unique")

        self._characters = ordered_characters
        self._character_to_id = {
            character: token_id
            for token_id, character in enumerate(ordered_characters, start=1)
        }

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        """Build a deterministic vocabulary from the unique characters in text."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return cls(sorted(set(text)))

    @property
    def characters(self) -> tuple[str, ...]:
        """Vocabulary characters in token-ID order; unknown has ID zero."""
        return self._characters

    @property
    def vocabulary_size(self) -> int:
        return len(self._characters) + 1

    def encode(self, text: str) -> list[int]:
        """Encode each code point, mapping out-of-vocabulary characters to zero."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return [
            self._character_to_id.get(character, UNKNOWN_TOKEN_ID)
            for character in text
        ]

    def decode(self, token_ids: Iterable[int]) -> str:
        """Decode IDs; unknown IDs become U+FFFD and invalid IDs raise errors."""
        decoded: list[str] = []
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("token IDs must be integers")
            if token_id == UNKNOWN_TOKEN_ID:
                decoded.append(UNKNOWN_CHARACTER)
            elif token_id < 0 or token_id > len(self._characters):
                raise ValueError(f"token ID is outside the vocabulary: {token_id}")
            else:
                decoded.append(self._characters[token_id - 1])
        return "".join(decoded)

    def save(self, path: str | Path) -> None:
        """Save the vocabulary as a versioned, human-readable JSON file."""
        payload = {
            "format_version": _FORMAT_VERSION,
            "unknown_token_id": UNKNOWN_TOKEN_ID,
            "characters": list(self._characters),
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharacterTokenizer":
        """Load and validate a previously saved vocabulary."""
        try:
            payload = json.loads(
                Path(path).read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
            )
        except json.JSONDecodeError as error:
            raise VocabularyFormatError("Vocabulary file is not valid JSON") from error

        if not isinstance(payload, dict) or set(payload) != _VOCABULARY_KEYS:
            raise VocabularyFormatError("Vocabulary file has an invalid structure")
        if (
            type(payload["format_version"]) is not int
            or payload["format_version"] != _FORMAT_VERSION
        ):
            raise VocabularyFormatError("Unsupported vocabulary format version")
        if (
            type(payload["unknown_token_id"]) is not int
            or payload["unknown_token_id"] != UNKNOWN_TOKEN_ID
        ):
            raise VocabularyFormatError("Vocabulary has an invalid unknown token ID")
        if not isinstance(payload["characters"], list):
            raise VocabularyFormatError("Vocabulary characters must be a list")

        try:
            return cls(payload["characters"])
        except (TypeError, ValueError) as error:
            raise VocabularyFormatError(str(error)) from error
