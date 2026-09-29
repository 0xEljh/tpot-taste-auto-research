"""Loopback HTTP worker for the Python/PEFT TPOT scorer."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from tpot_taste.adapters import DEFAULT_BASE, resolve_adapter_base
from tpot_taste.scoring.model import load_trained_scorer, score_texts


class ScoreRequest(BaseModel):
    texts: list[str] = Field(min_length=1)


@dataclass
class ScorerWorkerConfig:
    adapter: Path = Path("outputs/scorer/qwen3b-bt-taste-LOCKED")
    fallback_base: str = DEFAULT_BASE
    require_cuda: bool = True
    batch_size: int = 32

    @classmethod
    def from_env(cls) -> "ScorerWorkerConfig":
        return cls(
            adapter=Path(os.environ.get("TPOT_SCORER_ADAPTER", "outputs/scorer/qwen3b-bt-taste-LOCKED")),
            fallback_base=os.environ.get("TPOT_SCORER_FALLBACK_BASE", DEFAULT_BASE),
            require_cuda=os.environ.get("TPOT_SCORER_REQUIRE_CUDA", "1") != "0",
            batch_size=int(os.environ.get("TPOT_SCORER_BATCH_SIZE", "32")),
        )


class ScorerWorker:
    def __init__(self, config: ScorerWorkerConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.base_model_id: str | None = None
        self.last_error: dict[str, Any] | None = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None and self.last_error is None

    def load(self) -> None:
        try:
            self.base_model_id = resolve_adapter_base(self.config.adapter, self.config.fallback_base)
            self.model, self.tokenizer = load_trained_scorer(
                str(self.config.adapter),
                self.base_model_id,
                four_bit=True,
                require_cuda=self.config.require_cuda,
            )
            self.last_error = None
        except RuntimeError as exc:
            self.model = None
            self.tokenizer = None
            code = "gpu_unavailable" if "cuda" in str(exc).lower() or "offload" in str(exc).lower() else "model_load_failed"
            self.last_error = {"code": code, "message": str(exc)}
        except Exception as exc:  # noqa: BLE001 - expose worker readiness errors cleanly.
            self.model = None
            self.tokenizer = None
            self.last_error = {"code": "model_load_failed", "message": str(exc)}

    def health(self) -> dict[str, Any]:
        if not self.ready:
            return {
                "status": "loading_failed" if self.last_error else "loading",
                "ready": False,
                "last_error": self.last_error,
                "base_model_id": self.base_model_id,
            }
        return {"status": "ok", "ready": True, "last_error": None, "base_model_id": self.base_model_id}

    def score(self, texts: list[str]) -> list[float]:
        if not self.ready:
            raise RuntimeError("scorer worker is not ready")
        values = score_texts(self.model, self.tokenizer, texts, batch_size=self.config.batch_size)
        return [float(x) for x in values.tolist()]


def create_app(worker: ScorerWorker | None = None) -> FastAPI:
    app = FastAPI(title="TPOT Scorer Worker")
    app.state.worker = worker or ScorerWorker(ScorerWorkerConfig.from_env())

    @app.on_event("startup")
    def startup() -> None:
        app.state.worker.load()

    @app.get("/health")
    def health():
        payload = app.state.worker.health()
        if not payload["ready"]:
            raise HTTPException(status_code=503, detail=payload)
        return payload

    @app.post("/score")
    def score(req: ScoreRequest):
        try:
            return {"scores": app.state.worker.score(req.texts)}
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail={"code": "model_load_failed", "message": str(exc)}) from exc

    return app


app = create_app()
