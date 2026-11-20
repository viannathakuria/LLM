import pytest
import torch

from myllm_model import (
    GenerationConfig,
    TinyGPT,
    TransformerConfig,
    generate_text,
)
from myllm_tokenizer import CharacterTokenizer


def _greedy_model() -> tuple[TinyGPT, CharacterTokenizer]:
    tokenizer = CharacterTokenizer.from_text("abc")
    model = TinyGPT(
        TransformerConfig(
            vocab_size=tokenizer.vocabulary_size,
            context_length=3,
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
    return model, tokenizer


def _sampling_model() -> tuple[TinyGPT, CharacterTokenizer]:
    model, tokenizer = _greedy_model()
    with torch.no_grad():
        model.language_model_head.bias.fill_(-100)
        model.language_model_head.bias[tokenizer.encode("a")[0]] = 0
        model.language_model_head.bias[tokenizer.encode("b")[0]] = 0
    return model, tokenizer


def test_generation_encodes_prompt_and_decodes_new_tokens() -> None:
    model, tokenizer = _greedy_model()
    model.train()
    weights_before = {
        name: tensor.clone()
        for name, tensor in model.state_dict().items()
    }
    inference_modes: list[bool] = []
    hook = model.register_forward_pre_hook(
        lambda module, inputs: inference_modes.append(
            torch.is_inference_mode_enabled()
        )
    )

    result = generate_text(
        model,
        tokenizer,
        "a?",
        GenerationConfig(max_new_tokens=2, temperature=0),
    )
    hook.remove()

    assert tokenizer.encode("?") == [0]
    assert result == "a?cc"
    assert model.training
    assert inference_modes == [True, True]
    assert all(
        torch.equal(weights_before[name], tensor)
        for name, tensor in model.state_dict().items()
    )


def test_maximum_token_count_and_zero_token_generation() -> None:
    model, tokenizer = _greedy_model()

    assert generate_text(
        model,
        tokenizer,
        "ab",
        GenerationConfig(max_new_tokens=3, temperature=0),
    ) == "abccc"
    assert generate_text(
        model,
        tokenizer,
        "ab",
        GenerationConfig(max_new_tokens=0),
    ) == "ab"


def test_context_uses_only_most_recent_tokens() -> None:
    model, tokenizer = _greedy_model()
    inputs_seen: list[torch.Tensor] = []
    hook = model.register_forward_pre_hook(
        lambda module, inputs: inputs_seen.append(inputs[0].detach().clone())
    )

    generate_text(
        model,
        tokenizer,
        "abcab",
        GenerationConfig(max_new_tokens=2, temperature=0),
    )
    hook.remove()

    assert [inputs.shape for inputs in inputs_seen] == [(1, 3), (1, 3)]
    assert inputs_seen[0].tolist() == [tokenizer.encode("cab")]
    assert inputs_seen[1].tolist() == [tokenizer.encode("abc")]


def test_greedy_generation_is_reproducible() -> None:
    model, tokenizer = _greedy_model()
    config = GenerationConfig(max_new_tokens=5, temperature=0)

    first = generate_text(model, tokenizer, "a", config)
    second = generate_text(model, tokenizer, "a", config)

    assert first == second == "accccc"


def test_seeded_sampling_is_reproducible() -> None:
    model, tokenizer = _sampling_model()
    config = GenerationConfig(
        max_new_tokens=12,
        temperature=0.8,
        top_k=2,
        seed=17,
    )

    first = generate_text(model, tokenizer, "a", config)
    second = generate_text(model, tokenizer, "a", config)

    assert first == second


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Apple MPS is unavailable",
)
def test_seeded_sampling_is_reproducible_with_mps_model() -> None:
    model, tokenizer = _sampling_model()
    model.to("mps")
    config = GenerationConfig(
        max_new_tokens=6,
        temperature=0.8,
        top_k=2,
        seed=29,
    )

    first = generate_text(model, tokenizer, "a", config)
    second = generate_text(model, tokenizer, "a", config)

    assert first == second


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_new_tokens": -1}, "max_new_tokens"),
        ({"max_new_tokens": True}, "max_new_tokens"),
        ({"max_new_tokens": 1.5}, "max_new_tokens"),
        ({"temperature": -0.1}, "temperature"),
        ({"temperature": float("nan")}, "temperature"),
        ({"temperature": float("inf")}, "temperature"),
        ({"temperature": 10**400}, "temperature"),
        ({"temperature": True}, "temperature"),
        ({"top_k": 0}, "top_k"),
        ({"top_k": True}, "top_k"),
        ({"temperature": 0, "top_k": 1}, "greedy"),
        ({"seed": True}, "seed"),
        ({"seed": 1.5}, "seed"),
        ({"seed": 2**64}, "seed"),
    ],
)
def test_generation_config_rejects_invalid_settings(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        GenerationConfig(**kwargs)


def test_generation_rejects_invalid_prompt_and_top_k() -> None:
    model, tokenizer = _greedy_model()

    with pytest.raises(ValueError, match="prompt"):
        generate_text(model, tokenizer, "")
    with pytest.raises(TypeError, match="prompt"):
        generate_text(model, tokenizer, None)
    with pytest.raises(ValueError, match="top_k"):
        generate_text(
            model,
            tokenizer,
            "a",
            GenerationConfig(max_new_tokens=1, temperature=1, top_k=5),
        )


def test_generation_restores_mode_if_forward_raises() -> None:
    model, tokenizer = _greedy_model()
    model.train()

    def fail_forward(module, inputs):
        raise RuntimeError("forward failed")

    hook = model.register_forward_pre_hook(fail_forward)
    with pytest.raises(RuntimeError, match="forward failed"):
        generate_text(
            model,
            tokenizer,
            "a",
            GenerationConfig(max_new_tokens=1, temperature=0),
        )
    hook.remove()

    assert model.training
