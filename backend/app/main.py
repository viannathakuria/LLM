from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.settings import get_cors_origins

app = FastAPI(
    title="MyLLM API",
    description="Backend foundation for the MyLLM project.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "myllm-backend"}
