import torch

from myllm_model import select_device


def test_select_device_uses_mps_when_built_and_available(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert select_device() == torch.device("mps")


def test_select_device_falls_back_to_cpu_when_mps_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert select_device() == torch.device("cpu")


def test_select_device_falls_back_to_cpu_when_mps_is_not_built(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert select_device() == torch.device("cpu")
