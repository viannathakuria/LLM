import math
from dataclasses import dataclass

import torch
from torch import Tensor

from myllm_tokenizer import CharacterTokenizer

from .transformer import TinyGPT

_MIN_SEED = -(2**63)
_MAX_SEED = 2**64 - 1


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or self.max_new_tokens < 0
        ):
            raise ValueError("max_new_tokens must be a non-negative integer")
        if isinstance(self.temperature, bool) or not isinstance(
            self.temperature,
            (int, float),
        ):
            raise ValueError("temperature must be a finite non-negative number")
        try:
            temperature = float(self.temperature)
        except OverflowError as error:
            raise ValueError(
                "temperature must be a finite non-negative number"
            ) from error
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be a finite non-negative number")
        object.__setattr__(self, "temperature", temperature)
        if self.top_k is not None and (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k < 1
        ):
            raise ValueError("top_k must be a positive integer or None")
        if self.temperature == 0 and self.top_k is not None:
            raise ValueError("top_k cannot be used with greedy temperature 0")
        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not _MIN_SEED <= self.seed <= _MAX_SEED
        ):
            raise ValueError("seed must be an integer in the supported 64-bit range")


def _validate_prompt(
    prompt: str,
    tokenizer: CharacterTokenizer,
    model: TinyGPT,
) -> list[int]:
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not prompt:
        raise ValueError("prompt must not be empty")
    if model.config.vocab_size != tokenizer.vocabulary_size:
        raise ValueError("model vocabulary size does not match the tokenizer")

    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        raise ValueError("prompt must encode to at least one token")
    if any(type(token_id) is not int for token_id in token_ids):
        raise TypeError("tokenizer must encode prompts to integer token IDs")
    if any(not 0 <= token_id < model.config.vocab_size for token_id in token_ids):
        raise ValueError("encoded prompt contains a token ID outside the model vocabulary")
    return token_ids


def _sample_token(
    logits: Tensor,
    config: GenerationConfig,
    generator: torch.Generator | None,
) -> int:
    if config.temperature == 0:
        return int(torch.argmax(logits).item())

    cpu_logits = logits.detach().to(device="cpu").to(dtype=torch.float64)
    scaled_logits = cpu_logits / config.temperature
    if not torch.isfinite(scaled_logits).all().item():
        raise ValueError("temperature is too small for stable sampling")

    if config.top_k is not None:
        top_logits, top_ids = torch.topk(scaled_logits, config.top_k)
        probabilities = torch.softmax(top_logits, dim=-1)
        selected = torch.multinomial(probabilities, 1, generator=generator)
        return int(top_ids[selected].item())

    probabilities = torch.softmax(scaled_logits, dim=-1)
    selected = torch.multinomial(probabilities, 1, generator=generator)
    return int(selected.item())


def generate_text(
    model: TinyGPT,
    tokenizer: CharacterTokenizer,
    prompt: str,
    config: GenerationConfig = GenerationConfig(),
) -> str:
    """Generate a suffix autoregressively and append its decoded text to prompt."""
    if not isinstance(config, GenerationConfig):
        raise TypeError("config must be a GenerationConfig")
    if not isinstance(model, TinyGPT):
        raise TypeError("model must be a TinyGPT")
    if not isinstance(tokenizer, CharacterTokenizer):
        raise TypeError("tokenizer must be a CharacterTokenizer")
    prompt_token_ids = _validate_prompt(prompt, tokenizer, model)
    if config.top_k is not None and config.top_k > tokenizer.vocabulary_size:
        raise ValueError("top_k cannot exceed the checkpoint vocabulary size")
    if config.max_new_tokens == 0:
        return prompt

    device = next(model.parameters()).device
    generator = None
    if config.seed is not None and config.temperature > 0:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed)

    context = prompt_token_ids[-model.config.context_length :]
    generated_token_ids: list[int] = []
    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            for _ in range(config.max_new_tokens):
                input_ids = torch.tensor(
                    [context],
                    dtype=torch.long,
                    device=device,
                )
                logits = model(input_ids)
                if (
                    logits.ndim != 3
                    or logits.shape[0] != 1
                    or logits.shape[-1] != tokenizer.vocabulary_size
                ):
                    raise RuntimeError("model returned logits with an invalid shape")
                next_token_logits = logits[0, -1]
                if not torch.isfinite(next_token_logits).all().item():
                    raise RuntimeError("model returned non-finite token logits")
                next_token_id = _sample_token(
                    next_token_logits,
                    config,
                    generator,
                )
                generated_token_ids.append(next_token_id)
                context = (context + [next_token_id])[-model.config.context_length :]
    finally:
        model.train(was_training)

    return prompt + tokenizer.decode(generated_token_ids)
