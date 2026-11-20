import torch


def select_device() -> torch.device:
    """Use Apple MPS when built and available; otherwise use the CPU."""
    mps_backend = getattr(torch.backends, "mps", None)
    if (
        mps_backend is not None
        and mps_backend.is_built()
        and mps_backend.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")
