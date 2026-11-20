import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import torch
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from myllm_model import (
    CheckpointFormatError,
    EvaluationMetrics,
    TinyGPT,
    TrainingConfig,
    TrainingResult,
    TransformerConfig,
    save_checkpoint,
)
from myllm_tokenizer import CharacterTokenizer


def _save_test_checkpoint(tmp_path):
    tokenizer = CharacterTokenizer.from_text("abc")
    model = TinyGPT(
        TransformerConfig(
            vocab_size=tokenizer.vocabulary_size,
            context_length=4,
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

    training_config = TrainingConfig(num_steps=1, eval_frequency=1)
    training_result = TrainingResult(
        steps_completed=1,
        evaluations=(EvaluationMetrics(1, 1.0, 1.0),),
        skipped_training_sequences=0,
        skipped_validation_sequences=0,
    )
    checkpoint_dir = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint_dir,
        model,
        tokenizer,
        training_config,
        training_result,
    )
    return checkpoint_dir


def test_generate_loads_checkpoint_once_and_returns_generated_text(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint_dir = _save_test_checkpoint(tmp_path)
    monkeypatch.setenv("MYLLM_CHECKPOINT_DIR", str(checkpoint_dir))
    load_count = 0
    real_load_checkpoint = main_module.load_checkpoint

    def count_loads(path):
        nonlocal load_count
        load_count += 1
        return real_load_checkpoint(path, device=torch.device("cpu"))

    monkeypatch.setattr(main_module, "load_checkpoint", count_loads)

    with TestClient(app) as client:
        first = client.post(
            "/generate",
            json={
                "prompt": "ab",
                "max_new_tokens": 2,
                "temperature": 0,
            },
        )
        second = client.post(
            "/generate",
            json={
                "prompt": "x",
                "max_new_tokens": 1,
                "temperature": 0,
            },
        )

    assert first.status_code == 200
    assert first.json() == {"generated_text": "abcc"}
    assert second.status_code == 200
    assert second.json() == {"generated_text": "xc"}
    assert load_count == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": ""},
        {"prompt": "abc", "max_new_tokens": -1},
        {"prompt": "abc", "max_new_tokens": 257},
        {"prompt": "abc", "max_new_tokens": 1.5},
        {"prompt": "abc", "temperature": -0.1},
        {"prompt": "abc", "temperature": "NaN"},
        {"prompt": "abc", "temperature": "Infinity"},
        {"prompt": "abc", "temperature": "-Infinity"},
    ],
)
def test_generate_rejects_invalid_request_values(
    tmp_path,
    monkeypatch,
    payload,
) -> None:
    checkpoint_dir = _save_test_checkpoint(tmp_path)
    monkeypatch.setenv("MYLLM_CHECKPOINT_DIR", str(checkpoint_dir))

    with TestClient(app) as client:
        response = client.post("/generate", json=payload)

    assert response.status_code == 422


def test_generate_accepts_zero_and_maximum_token_limits(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint_dir = _save_test_checkpoint(tmp_path)
    monkeypatch.setenv("MYLLM_CHECKPOINT_DIR", str(checkpoint_dir))
    requested_limits = []

    def return_prompt(_model, _tokenizer, prompt, config):
        requested_limits.append(config.max_new_tokens)
        return prompt

    monkeypatch.setattr(main_module, "generate_text", return_prompt)

    with TestClient(app) as client:
        zero_response = client.post(
            "/generate",
            json={"prompt": "hello", "max_new_tokens": 0},
        )
        maximum_response = client.post(
            "/generate",
            json={"prompt": "hello", "max_new_tokens": 256},
        )

    assert zero_response.status_code == 200
    assert maximum_response.status_code == 200
    assert requested_limits == [0, 256]


def test_missing_checkpoint_keeps_health_available_and_returns_503(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MYLLM_CHECKPOINT_DIR", "")

    with TestClient(app) as client:
        health_response = client.get("/health")
        generation_response = client.post(
            "/generate",
            json={"prompt": "hello"},
        )

    assert health_response.status_code == 200
    assert health_response.json() == {
        "status": "ok",
        "service": "myllm-backend",
    }
    assert generation_response.status_code == 503
    assert generation_response.json() == {
        "detail": "Generation model is unavailable",
    }


def test_invalid_checkpoint_is_logged_and_returns_sanitized_503(
    monkeypatch,
    caplog,
) -> None:
    private_path = "/private/checkpoints/bad-model"
    monkeypatch.setenv("MYLLM_CHECKPOINT_DIR", private_path)
    caplog.set_level(logging.ERROR)

    with TestClient(app) as client:
        response = client.post("/generate", json={"prompt": "hello"})

    assert response.status_code == 503
    assert private_path not in response.text
    assert "MyLLM checkpoint could not be loaded" in caplog.text


def test_checkpoint_loader_errors_are_logged_and_hidden(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setenv("MYLLM_CHECKPOINT_DIR", "/private/model")

    def fail_to_load(_path):
        raise CheckpointFormatError("sensitive checkpoint detail")

    monkeypatch.setattr(main_module, "load_checkpoint", fail_to_load)
    caplog.set_level(logging.ERROR)

    with TestClient(app) as client:
        response = client.post("/generate", json={"prompt": "hello"})

    assert response.status_code == 503
    assert "sensitive checkpoint detail" not in response.text
    assert "sensitive checkpoint detail" not in caplog.text


def test_generation_errors_are_logged_and_hidden(
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    checkpoint_dir = _save_test_checkpoint(tmp_path)
    monkeypatch.setenv("MYLLM_CHECKPOINT_DIR", str(checkpoint_dir))

    def fail_to_generate(*_args, **_kwargs):
        raise RuntimeError("sensitive generation detail")

    monkeypatch.setattr(main_module, "generate_text", fail_to_generate)
    caplog.set_level(logging.ERROR)

    with TestClient(app) as client:
        response = client.post("/generate", json={"prompt": "hello"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Text generation failed"}
    assert "sensitive generation detail" not in response.text
    assert "MyLLM text generation failed" in caplog.text


def test_generation_requests_are_serialized(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint_dir = _save_test_checkpoint(tmp_path)
    monkeypatch.setenv("MYLLM_CHECKPOINT_DIR", str(checkpoint_dir))
    active_requests = 0
    maximum_active_requests = 0
    tracker_lock = threading.Lock()

    def blocking_generate(_model, _tokenizer, prompt, _config):
        nonlocal active_requests, maximum_active_requests
        with tracker_lock:
            active_requests += 1
            maximum_active_requests = max(
                maximum_active_requests,
                active_requests,
            )
        time.sleep(0.05)
        with tracker_lock:
            active_requests -= 1
        return prompt

    monkeypatch.setattr(main_module, "generate_text", blocking_generate)
    request_barrier = Barrier(2)

    with TestClient(app) as client:
        def send_request(prompt):
            request_barrier.wait()
            return client.post(
                "/generate",
                json={"prompt": prompt},
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(
                executor.map(
                    send_request,
                    ("first", "second"),
                )
            )

    assert all(response.status_code == 200 for response in responses)
    assert maximum_active_requests == 1


def test_generate_preflight_allows_post(monkeypatch, tmp_path) -> None:
    checkpoint_dir = _save_test_checkpoint(tmp_path)
    monkeypatch.setenv("MYLLM_CHECKPOINT_DIR", str(checkpoint_dir))

    with TestClient(app) as client:
        response = client.options(
            "/generate",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "POST" in response.headers["access-control-allow-methods"]
