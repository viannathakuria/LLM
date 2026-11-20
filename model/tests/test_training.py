import itertools

import pytest
import torch
import torch.nn.functional as functional

from myllm_model import TinyGPT, TrainingConfig, TransformerConfig
from myllm_model.training import (
    IGNORE_INDEX,
    _shuffled_window_batches,
    build_next_token_windows,
    evaluate_loss,
    make_next_token_batch,
    next_token_loss,
    train_model,
)
from myllm_tokenizer.dataset import (
    PreparedDataset,
    load_prepared_dataset,
    prepare_text_records,
    save_prepared_dataset,
)


def _saved_sample_dataset(tmp_path) -> PreparedDataset:
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
    output_dir = tmp_path / "prepared"
    save_prepared_dataset(prepared, output_dir)
    return load_prepared_dataset(output_dir)


def test_windows_use_context_stride_and_shift_next_token_targets() -> None:
    windows, skipped = build_next_token_windows(
        [(1, 2, 3, 4, 5)],
        context_length=2,
        vocab_size=6,
        split_name="training",
    )

    assert windows == ((1, 2, 3), (3, 4, 5))
    assert skipped == 0
    inputs, targets = make_next_token_batch(windows, vocab_size=6)
    assert inputs.tolist() == [[1, 2], [3, 4]]
    assert targets.tolist() == [[2, 3], [4, 5]]


def test_short_records_are_counted_without_inventing_token_pairs() -> None:
    windows, skipped = build_next_token_windows(
        [(1,), (2, 3)],
        context_length=2,
        vocab_size=4,
        split_name="training",
    )

    assert windows == ((2, 3),)
    assert skipped == 1


def test_batch_padding_uses_valid_input_id_and_masks_target_loss() -> None:
    inputs, targets = make_next_token_batch(
        [(1, 2, 3), (4, 5)],
        vocab_size=6,
    )

    assert inputs.shape == targets.shape == (2, 2)
    assert inputs.tolist() == [[1, 2], [4, 0]]
    assert targets.tolist() == [[2, 3], [5, IGNORE_INDEX]]


@pytest.mark.parametrize(
    ("sequence", "error"),
    [
        ((-1, 2), ValueError),
        ((1, 6), ValueError),
        ((1, True), TypeError),
        ((1, 2.5), TypeError),
    ],
)
def test_batch_rejects_invalid_token_ids(
    sequence: tuple[object, ...],
    error: type[Exception],
) -> None:
    with pytest.raises(error, match="token"):
        make_next_token_batch([sequence], vocab_size=6)


def test_next_token_loss_ignores_padded_targets() -> None:
    logits = torch.randn(2, 2, 6, requires_grad=True)
    targets = torch.tensor([[2, 3], [5, IGNORE_INDEX]], dtype=torch.long)

    loss = next_token_loss(logits, targets)
    expected = functional.cross_entropy(
        logits[targets != IGNORE_INDEX],
        targets[targets != IGNORE_INDEX],
    )

    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad[1, 1]) == 0


def test_evaluation_uses_token_weighted_loss_without_updating_model(
    tmp_path,
) -> None:
    prepared = _saved_sample_dataset(tmp_path)
    model = TinyGPT(
        TransformerConfig(
            vocab_size=prepared.tokenizer.vocabulary_size,
            context_length=4,
            dropout=0,
        )
    ).train()
    windows, _ = build_next_token_windows(
        prepared.validation_sequences,
        context_length=model.config.context_length,
        vocab_size=model.config.vocab_size,
        split_name="validation",
    )
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    loss = evaluate_loss(
        model,
        windows,
        batch_size=2,
        device=torch.device("cpu"),
    )
    loss_with_smaller_batches = evaluate_loss(
        model,
        windows,
        batch_size=1,
        device=torch.device("cpu"),
    )

    assert torch.isfinite(torch.tensor(loss))
    assert loss == pytest.approx(loss_with_smaller_batches, abs=1e-6)
    assert model.training
    assert all(
        torch.equal(before[name], tensor)
        for name, tensor in model.state_dict().items()
    )


def test_training_updates_weights_and_reports_both_losses(tmp_path) -> None:
    prepared = _saved_sample_dataset(tmp_path)
    model = TinyGPT(
        TransformerConfig(
            vocab_size=prepared.tokenizer.vocabulary_size,
            context_length=4,
            dropout=0,
        )
    )
    before = {
        name: tensor.clone()
        for name, tensor in model.state_dict().items()
    }

    result = train_model(
        model,
        prepared,
        TrainingConfig(
            learning_rate=0.01,
            batch_size=2,
            num_steps=3,
            eval_frequency=2,
            seed=19,
        ),
        device=torch.device("cpu"),
    )

    assert result.steps_completed == 3
    assert [item.step for item in result.evaluations] == [2, 3]
    assert all(
        torch.isfinite(torch.tensor(item.training_loss))
        and torch.isfinite(torch.tensor(item.validation_loss))
        for item in result.evaluations
    )
    assert any(
        not torch.equal(before[name], tensor)
        for name, tensor in model.state_dict().items()
    )


def test_training_validates_all_ids_before_moving_model(monkeypatch, tmp_path) -> None:
    prepared = _saved_sample_dataset(tmp_path)
    invalid_dataset = PreparedDataset(
        train_sequences=((0, prepared.tokenizer.vocabulary_size),),
        validation_sequences=prepared.validation_sequences,
        tokenizer=prepared.tokenizer,
        metadata=prepared.metadata,
    )
    model = TinyGPT(TransformerConfig(vocab_size=prepared.tokenizer.vocabulary_size))

    def unexpected_move(*args, **kwargs):
        pytest.fail("model moved before input IDs were validated")

    monkeypatch.setattr(model, "to", unexpected_move)
    with pytest.raises(ValueError, match=r"outside \[0,"):
        train_model(
            model,
            invalid_dataset,
            TrainingConfig(num_steps=1, eval_frequency=1),
            device=torch.device("cpu"),
        )


def test_training_window_shuffle_is_deterministic(tmp_path) -> None:
    prepared = _saved_sample_dataset(tmp_path)
    windows, _ = build_next_token_windows(
        prepared.train_sequences,
        context_length=4,
        vocab_size=prepared.tokenizer.vocabulary_size,
        split_name="training",
    )

    first = list(
        itertools.islice(
            _shuffled_window_batches(windows, batch_size=2, seed=23),
            4,
        )
    )
    second = list(
        itertools.islice(
            _shuffled_window_batches(windows, batch_size=2, seed=23),
            4,
        )
    )

    assert first == second


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Apple MPS is unavailable",
)
def test_tiny_training_step_runs_on_mps(tmp_path) -> None:
    prepared = _saved_sample_dataset(tmp_path)
    model = TinyGPT(
        TransformerConfig(
            vocab_size=prepared.tokenizer.vocabulary_size,
            context_length=4,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0,
        )
    )

    result = train_model(
        model,
        prepared,
        TrainingConfig(batch_size=2, num_steps=1, eval_frequency=1),
        device=torch.device("mps"),
    )

    assert result.steps_completed == 1
    assert torch.isfinite(torch.tensor(result.final_validation_loss))
