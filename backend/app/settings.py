import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def get_cors_origins() -> list[str]:
    configured_origins = os.getenv(
        "MYLLM_CORS_ORIGINS",
        "http://localhost:3000",
    )
    origins = [
        origin.strip()
        for origin in configured_origins.split(",")
        if origin.strip()
    ]

    if not origins:
        raise ValueError("MYLLM_CORS_ORIGINS must contain at least one origin")

    return origins


def get_checkpoint_dir() -> Path | None:
    configured_directory = os.getenv("MYLLM_CHECKPOINT_DIR", "").strip()
    if not configured_directory:
        return None
    return Path(configured_directory).expanduser()
