import torch
import pytest

from myllm_model import TinyGPT, TransformerConfig


def test_default_config_and_forward_shape() -> None:
    config = TransformerConfig(vocab_size=12)
    model = TinyGPT(config).eval()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids)

    assert config.context_length == 64
    assert config.num_layers == 2
    assert config.num_heads == 2
    assert config.embedding_dim == 64
    assert config.batch_size == 4
    assert logits.shape == (2, 3, 12)
    assert torch.isfinite(logits).all()


def test_future_tokens_do_not_change_earlier_logits() -> None:
    model = TinyGPT(
        TransformerConfig(
            vocab_size=10,
            context_length=8,
            dropout=0,
        )
    ).eval()
    original = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    changed_future = original.clone()
    changed_future[0, 3] = 8

    with torch.no_grad():
        original_logits = model(original)
        changed_logits = model(changed_future)

    torch.testing.assert_close(
        original_logits[:, :3],
        changed_logits[:, :3],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("input_ids", "error", "message"),
    [
        (torch.tensor([1, 2]), ValueError, "shape"),
        (torch.tensor([[1.0, 2.0]]), TypeError, "torch.long"),
        (torch.empty((1, 0), dtype=torch.long), ValueError, "non-empty"),
        (torch.tensor([[1, 2, 3]]), ValueError, "context length"),
        (torch.tensor([[-1]]), ValueError, "outside the vocabulary"),
        (torch.tensor([[8]]), ValueError, "outside the vocabulary"),
    ],
)
def test_forward_rejects_invalid_inputs(
    input_ids: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    model = TinyGPT(TransformerConfig(vocab_size=8, context_length=2))

    with pytest.raises(error, match=message):
        model(input_ids)


@pytest.mark.parametrize(
    "overrides",
    [
        {"vocab_size": 0},
        {"context_length": 0},
        {"num_layers": 0},
        {"num_heads": 0},
        {"embedding_dim": 63},
        {"dropout": 1.0},
    ],
)
def test_config_rejects_invalid_values(overrides: dict[str, object]) -> None:
    values: dict[str, object] = {"vocab_size": 16}
    values.update(overrides)

    with pytest.raises(ValueError):
        TransformerConfig(**values)
