"""Tailnet-only TPOT inference API."""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tpot_taste.deployment.artifacts import sha256_file
from tpot_taste.inference import (
    DEFAULT_GENERATION,
    build_prompt,
    build_writer_completion_payload,
    normalize_generated_text,
    rank_topk,
)


RETRYABLE_CODES = {"gpu_busy", "gpu_unavailable", "deadline_exceeded"}
SERVICE_UNAVAILABLE_CODES = {"gpu_busy", "gpu_unavailable", "models_not_exported", "model_load_failed"}


class BackendRequirement(str, Enum):
    SCORER = "scorer"
    WRITER_AND_SCORER = "writer_and_scorer"


class ServiceError(RuntimeError):
    def __init__(self, code: str, message: str, *, retry_after_s: int | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retry_after_s = retry_after_s

    def payload(self) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.retry_after_s is not None:
            error["retry_after_s"] = self.retry_after_s
        return {"error": error}


class ScoreRequest(BaseModel):
    texts: list[str] = Field(min_length=1)


class IdeateRequest(BaseModel):
    topic: str | None = None
    k: int = DEFAULT_GENERATION.k
    best_of: int = DEFAULT_GENERATION.best_of
    temperature: float = DEFAULT_GENERATION.temperature
    top_p: float = DEFAULT_GENERATION.top_p
    max_new_tokens: int = DEFAULT_GENERATION.max_new_tokens


class ImproveRequest(BaseModel):
    draft: str
    k: int = DEFAULT_GENERATION.k
    best_of: int = DEFAULT_GENERATION.best_of
    temperature: float = DEFAULT_GENERATION.temperature
    top_p: float = DEFAULT_GENERATION.top_p
    max_new_tokens: int = DEFAULT_GENERATION.max_new_tokens


class BatchRequest(BaseModel):
    requests: list[dict[str, Any]] = Field(default_factory=list)
    timeout_s: float = 180


class BatchItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    op: str


@dataclass
class ServiceConfig:
    registry_path: Path = Path("models/serving/registry.json")
    scorer_version: str = "v8-qwen3-8b"
    writer_url: str = "http://127.0.0.1:18181"
    scorer_url: str = "http://127.0.0.1:18182"
    writer_unit: str = "tpot-writer-llama.service"
    scorer_unit: str = "tpot-scorer.service"
    worker_startup_timeout_s: float = 180
    backend_call_timeout_s: float = 180
    idle_unload_timeout_s: float = 60
    lock_path: Path | None = None
    scorer_min_free_mb: int = 0
    writer_scorer_min_free_mb: int = 0

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "tpot-inference"
        return cls(
            registry_path=Path(os.environ.get("TPOT_REGISTRY_PATH", "models/serving/registry.json")),
            scorer_version=os.environ.get("TPOT_SCORER_VERSION", "v8-qwen3-8b"),
            writer_url=os.environ.get("TPOT_WRITER_URL", "http://127.0.0.1:18181"),
            scorer_url=os.environ.get("TPOT_SCORER_URL", "http://127.0.0.1:18182"),
            writer_unit=os.environ.get("TPOT_WRITER_UNIT", "tpot-writer-llama.service"),
            scorer_unit=os.environ.get("TPOT_SCORER_UNIT", "tpot-scorer.service"),
            worker_startup_timeout_s=float(os.environ.get("TPOT_WORKER_STARTUP_TIMEOUT_S", "180")),
            backend_call_timeout_s=float(os.environ.get("TPOT_BACKEND_CALL_TIMEOUT_S", "180")),
            idle_unload_timeout_s=float(os.environ.get("TPOT_IDLE_UNLOAD_TIMEOUT_S", "60")),
            lock_path=Path(os.environ.get("TPOT_GPU_LOCK", runtime_dir / "gpu.lock")),
            scorer_min_free_mb=int(os.environ.get("TPOT_SCORER_MIN_FREE_MB", "0")),
            writer_scorer_min_free_mb=int(os.environ.get("TPOT_WRITER_SCORER_MIN_FREE_MB", "0")),
        )


class ModelRegistry:
    def __init__(self, path: Path, *, scorer_version: str):
        self.path = path
        self.scorer_version = scorer_version
        self.last_error: ServiceError | None = None

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            self.last_error = ServiceError("models_not_exported", "Writer serving registry does not exist.")
            return None
        try:
            data = json.loads(self.path.read_text())
            self._validate_writer_hashes(data)
        except ServiceError as exc:
            self.last_error = exc
            return None
        except Exception as exc:  # noqa: BLE001 - health should expose config corruption cleanly.
            self.last_error = ServiceError("model_load_failed", f"Could not read serving registry: {exc}")
            return None
        self.last_error = None
        return data

    def _validate_writer_hashes(self, data: dict[str, Any]) -> None:
        writer = data.get("models", {}).get("writer")
        if not writer:
            raise ServiceError("models_not_exported", "Registry does not contain a writer model.")
        for path_key, hash_key in (("base_gguf", "base_gguf_sha256"), ("lora_gguf", "lora_gguf_sha256")):
            path = Path(writer[path_key])
            if not path.exists():
                raise ServiceError("models_not_exported", f"Writer artifact is missing: {path}")
            expected = writer.get(hash_key)
            if expected and sha256_file(path) != expected:
                raise ServiceError("model_load_failed", f"Writer artifact hash mismatch: {path}")

    @property
    def models_exported(self) -> bool:
        return self.load() is not None

    def model_versions(self) -> dict[str, str]:
        data = self.load()
        writer_version = "not_exported"
        if data:
            writer_version = data.get("models", {}).get("writer", {}).get("version", "unknown")
        return {"writer": writer_version, "scorer": self.scorer_version}

    def require_writer_exported(self) -> None:
        if self.load() is None:
            raise self.last_error or ServiceError("models_not_exported", "Writer artifacts have not been exported.")


class FileGpuAdmission:
    def __init__(self, *, lock_path: Path, scorer_min_free_mb: int = 0, writer_scorer_min_free_mb: int = 0):
        self.lock_path = lock_path
        self.scorer_min_free_mb = scorer_min_free_mb
        self.writer_scorer_min_free_mb = writer_scorer_min_free_mb
        self._held = False

    @property
    def held(self) -> bool:
        return self._held

    @contextlib.contextmanager
    def acquire(self, requirement: BackendRequirement) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("w") as lock_file:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ServiceError("gpu_busy", "TPOT GPU lock is already held.", retry_after_s=300) from exc
            self._held = True
            try:
                self.preflight(requirement)
                yield
            finally:
                self._held = False
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def preflight(self, requirement: BackendRequirement) -> None:
        floor = self.writer_scorer_min_free_mb if requirement is BackendRequirement.WRITER_AND_SCORER else self.scorer_min_free_mb
        if floor <= 0:
            return
        free = self._free_memory_mb()
        if free < floor:
            raise ServiceError(
                "gpu_unavailable",
                f"Insufficient GPU free memory for {requirement.value}: {free} MiB available, {floor} MiB required.",
                retry_after_s=300,
            )

    def _free_memory_mb(self) -> int:
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception as exc:  # noqa: BLE001 - maps host preflight failures into service semantics.
            raise ServiceError("gpu_unavailable", f"Could not query GPU free memory: {exc}", retry_after_s=300) from exc
        values = [int(line.strip()) for line in proc.stdout.splitlines() if line.strip()]
        if not values:
            raise ServiceError("gpu_unavailable", "nvidia-smi returned no GPU memory rows.", retry_after_s=300)
        return max(values)


class SystemdWorker:
    def __init__(self, *, unit: str, health_url: str, startup_timeout_s: float):
        self.unit = unit
        self.health_url = health_url
        self.startup_timeout_s = startup_timeout_s

    def is_ready(self) -> bool:
        try:
            return httpx.get(self.health_url, timeout=2).status_code == 200
        except httpx.HTTPError:
            return False

    def ensure_ready(self) -> None:
        if self.is_ready():
            return
        proc = subprocess.run(["systemctl", "--user", "start", self.unit], capture_output=True, text=True, timeout=20)
        if proc.returncode != 0:
            raise ServiceError("model_load_failed", f"Failed to start {self.unit}: {proc.stderr.strip() or proc.stdout.strip()}")
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self.is_ready():
                return
            time.sleep(1)
        raise ServiceError("model_load_failed", f"{self.unit} did not become ready before timeout")

    def stop(self) -> None:
        subprocess.run(["systemctl", "--user", "stop", self.unit], capture_output=True, text=True, timeout=20)


class BackendManager:
    def __init__(self, *, writer: SystemdWorker, scorer: SystemdWorker, idle_unload_timeout_s: float = 60):
        self.writer = writer
        self.scorer = scorer
        self.idle_unload_timeout_s = idle_unload_timeout_s
        self._timers: list[threading.Timer] = []

    def writer_ready(self) -> bool:
        return self.writer.is_ready()

    def scorer_ready(self) -> bool:
        return self.scorer.is_ready()

    def ensure_writer(self) -> None:
        self.writer.ensure_ready()

    def ensure_scorer(self) -> None:
        self.scorer.ensure_ready()

    def stop_writer(self) -> None:
        self.writer.stop()

    def stop_scorer(self) -> None:
        self.scorer.stop()

    def schedule_idle_unload(self, requirement: BackendRequirement) -> None:
        for timer in self._timers:
            timer.cancel()
        self._timers.clear()
        if self.idle_unload_timeout_s <= 0:
            return

        def unload() -> None:
            if requirement is BackendRequirement.WRITER_AND_SCORER:
                self.stop_writer()
            self.stop_scorer()

        timer = threading.Timer(self.idle_unload_timeout_s, unload)
        timer.daemon = True
        timer.start()
        self._timers.append(timer)


class LlamaWriterClient:
    def __init__(self, *, base_url: str, registry: ModelRegistry, timeout_s: float):
        self.base_url = base_url.rstrip("/")
        self.registry = registry
        self.timeout_s = timeout_s
        self._tokenizer = None
        self._generation_config = None

    def _load_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        data = self.registry.load()
        if data is None:
            raise self.registry.last_error or ServiceError("models_not_exported", "Writer artifacts have not been exported.")
        model_id = data["models"]["writer"]["base_model_id"]
        from transformers import AutoTokenizer, GenerationConfig

        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self._tokenizer = tok
        try:
            self._generation_config = GenerationConfig.from_pretrained(model_id)
        except Exception:  # noqa: BLE001 - HF default top_k is safe and tested.
            self._generation_config = None
        return self._tokenizer

    def generate(self, user: str, *, n: int, temperature: float, top_p: float, max_new_tokens: int) -> list[str]:
        tok = self._load_tokenizer()
        outs: list[str] = []
        with httpx.Client(timeout=self.timeout_s) as client:
            for _ in range(n):
                payload = build_writer_completion_payload(
                    tok,
                    user,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    generation_config=self._generation_config,
                )
                response = client.post(f"{self.base_url}/completion", json=payload)
                response.raise_for_status()
                data = response.json()
                text = data.get("content")
                if text is None and data.get("choices"):
                    text = data["choices"][0].get("text")
                outs.append(normalize_generated_text(text or ""))
        return outs


class HttpScorerClient:
    def __init__(self, *, base_url: str, timeout_s: float):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def score(self, texts: list[str]) -> list[float]:
        response = httpx.post(f"{self.base_url}/score", json={"texts": texts}, timeout=self.timeout_s)
        response.raise_for_status()
        return [float(x) for x in response.json()["scores"]]


class TpotOrchestrator:
    def __init__(
        self,
        *,
        registry: ModelRegistry,
        admission: FileGpuAdmission,
        backends: BackendManager,
        writer: LlamaWriterClient,
        scorer: HttpScorerClient,
    ):
        self.registry = registry
        self.admission = admission
        self.backends = backends
        self.writer = writer
        self.scorer = scorer
        self.runtime_state = "IDLE"
        self.last_error: dict[str, Any] | None = None

    def _set_error(self, exc: ServiceError) -> None:
        self.last_error = exc.payload()["error"]

    def health(self) -> dict[str, Any]:
        exported = self.registry.models_exported
        last_error = self.last_error or (self.registry.last_error.payload()["error"] if self.registry.last_error else None)
        return {
            "status": "ok",
            "runtime_state": self.runtime_state,
            "last_error": last_error,
            "writer_loaded": self.backends.writer_ready(),
            "scorer_loaded": self.backends.scorer_ready(),
            "gpu_lock_held": self.admission.held,
            "models_exported": exported,
        }

    def models(self) -> dict[str, str]:
        return self.registry.model_versions()

    def _ensure_backends(self, requirement: BackendRequirement) -> None:
        if requirement is BackendRequirement.WRITER_AND_SCORER:
            self.registry.require_writer_exported()
            self.runtime_state = "LOADING_WRITER"
            self.backends.ensure_writer()
            self.runtime_state = "WRITER_READY"
            try:
                self.admission.preflight(BackendRequirement.SCORER)
                self.runtime_state = "LOADING_SCORER"
                self.backends.ensure_scorer()
            except ServiceError:
                self.runtime_state = "UNLOADING"
                self.backends.stop_writer()
                raise
            self.runtime_state = "READY"
            return
        self.runtime_state = "LOADING_SCORER"
        self.backends.ensure_scorer()
        self.runtime_state = "READY"

    def _run(self, requirement: BackendRequirement, fn):
        try:
            with self.admission.acquire(requirement):
                self._ensure_backends(requirement)
                result = fn()
                self.backends.schedule_idle_unload(requirement)
            self.runtime_state = "IDLE"
            self.last_error = None
            return result
        except ServiceError as exc:
            self._set_error(exc)
            self.runtime_state = "IDLE"
            raise

    def score(self, texts: list[str]) -> dict[str, Any]:
        def work():
            return self._score_loaded(texts)

        return self._run(BackendRequirement.SCORER, work)

    def _score_loaded(self, texts: list[str]) -> dict[str, Any]:
        return {"model_versions": {"scorer": self.models()["scorer"]}, "scores": self.scorer.score(texts)}

    def ideate(
        self,
        *,
        topic: str | None = None,
        k: int = DEFAULT_GENERATION.k,
        best_of: int = DEFAULT_GENERATION.best_of,
        temperature: float = DEFAULT_GENERATION.temperature,
        top_p: float = DEFAULT_GENERATION.top_p,
        max_new_tokens: int = DEFAULT_GENERATION.max_new_tokens,
    ) -> dict[str, Any]:
        def work():
            return self._ideate_loaded(
                topic=topic,
                k=k,
                best_of=best_of,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )

        return self._run(BackendRequirement.WRITER_AND_SCORER, work)

    def _ideate_loaded(
        self,
        *,
        topic: str | None = None,
        k: int = DEFAULT_GENERATION.k,
        best_of: int = DEFAULT_GENERATION.best_of,
        temperature: float = DEFAULT_GENERATION.temperature,
        top_p: float = DEFAULT_GENERATION.top_p,
        max_new_tokens: int = DEFAULT_GENERATION.max_new_tokens,
    ) -> dict[str, Any]:
        user = build_prompt("ideate", topic=topic)
        candidates = self.writer.generate(user, n=best_of, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens)
        ranked = rank_topk(candidates, self.scorer.score(candidates), k)
        return {"model_versions": self.models(), "candidates": _candidate_payload(ranked)}

    def improve(
        self,
        *,
        draft: str,
        k: int = DEFAULT_GENERATION.k,
        best_of: int = DEFAULT_GENERATION.best_of,
        temperature: float = DEFAULT_GENERATION.temperature,
        top_p: float = DEFAULT_GENERATION.top_p,
        max_new_tokens: int = DEFAULT_GENERATION.max_new_tokens,
    ) -> dict[str, Any]:
        def work():
            return self._improve_loaded(
                draft=draft,
                k=k,
                best_of=best_of,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )

        return self._run(BackendRequirement.WRITER_AND_SCORER, work)

    def _improve_loaded(
        self,
        *,
        draft: str,
        k: int = DEFAULT_GENERATION.k,
        best_of: int = DEFAULT_GENERATION.best_of,
        temperature: float = DEFAULT_GENERATION.temperature,
        top_p: float = DEFAULT_GENERATION.top_p,
        max_new_tokens: int = DEFAULT_GENERATION.max_new_tokens,
    ) -> dict[str, Any]:
        user = build_prompt("improve", draft=draft)
        candidates = self.writer.generate(user, n=best_of, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens)
        ranked = rank_topk(candidates, self.scorer.score(candidates), k)
        return {"model_versions": self.models(), "candidates": _candidate_payload(ranked)}

    def batch(self, batch: BatchRequest) -> dict[str, Any]:
        requirement = _batch_requirement(batch.requests)

        def work():
            deadline = time.monotonic() + batch.timeout_s
            results: list[dict[str, Any]] = []
            for raw in batch.requests:
                item_id = str(raw.get("id", ""))
                if time.monotonic() > deadline:
                    results.append(_batch_error(item_id, "deadline_exceeded", "Batch deadline elapsed before this subrequest started."))
                    continue
                results.append(_execute_batch_item(self, raw, loaded=True))
            return {"model_versions": self.models(), "results": results}

        if requirement is None:
            return work()
        return self._run(requirement, work)


def _candidate_payload(ranked: list[tuple[str, float]]) -> list[dict[str, Any]]:
    return [{"text": text, "score": score} for text, score in ranked]


def _batch_error(item_id: str, code: str, message: str) -> dict[str, Any]:
    return {"id": item_id, "status": "error", "code": code, "message": message}


def _batch_ok(item_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"id": item_id, "status": "ok", **{k: v for k, v in payload.items() if k != "model_versions"}}


def _validation_error(item_id: str, exc: Exception) -> dict[str, Any]:
    return _batch_error(item_id, "validation_failed", str(exc))


def _batch_requirement(requests: list[dict[str, Any]]) -> BackendRequirement | None:
    ops = {raw.get("op") for raw in requests}
    if ops & {"ideate", "improve"}:
        return BackendRequirement.WRITER_AND_SCORER
    if "score" in ops:
        return BackendRequirement.SCORER
    return None


def _execute_batch_item(orchestrator, raw: dict[str, Any], *, loaded: bool = False) -> dict[str, Any]:
    try:
        item = BatchItem.model_validate(raw)
    except ValidationError as exc:
        return _validation_error(str(raw.get("id", "")), exc)
    try:
        if item.op == "score":
            req = ScoreRequest.model_validate(raw)
            payload = orchestrator._score_loaded(req.texts) if loaded and hasattr(orchestrator, "_score_loaded") else orchestrator.score(req.texts)
            return _batch_ok(item.id, payload)
        if item.op == "ideate":
            req = IdeateRequest.model_validate(raw)
            payload = orchestrator._ideate_loaded(**req.model_dump()) if loaded and hasattr(orchestrator, "_ideate_loaded") else orchestrator.ideate(**req.model_dump())
            return _batch_ok(item.id, payload)
        if item.op == "improve":
            req = ImproveRequest.model_validate(raw)
            payload = orchestrator._improve_loaded(**req.model_dump()) if loaded and hasattr(orchestrator, "_improve_loaded") else orchestrator.improve(**req.model_dump())
            return _batch_ok(item.id, payload)
        return _batch_error(item.id, "validation_failed", f"unsupported op: {item.op}")
    except ValidationError as exc:
        return _validation_error(item.id, exc)
    except ValueError as exc:
        return _validation_error(item.id, exc)


def _service_error_response(exc: ServiceError) -> JSONResponse:
    status = 503 if exc.code in SERVICE_UNAVAILABLE_CODES else 400
    headers = {"Retry-After": str(exc.retry_after_s)} if exc.retry_after_s is not None else None
    return JSONResponse(status_code=status, content=exc.payload(), headers=headers)


def _default_orchestrator() -> TpotOrchestrator:
    cfg = ServiceConfig.from_env()
    registry = ModelRegistry(cfg.registry_path, scorer_version=cfg.scorer_version)
    admission = FileGpuAdmission(
        lock_path=cfg.lock_path or Path("/tmp/tpot-inference/gpu.lock"),
        scorer_min_free_mb=cfg.scorer_min_free_mb,
        writer_scorer_min_free_mb=cfg.writer_scorer_min_free_mb,
    )
    backends = BackendManager(
        writer=SystemdWorker(unit=cfg.writer_unit, health_url=f"{cfg.writer_url}/health", startup_timeout_s=cfg.worker_startup_timeout_s),
        scorer=SystemdWorker(unit=cfg.scorer_unit, health_url=f"{cfg.scorer_url}/health", startup_timeout_s=cfg.worker_startup_timeout_s),
        idle_unload_timeout_s=cfg.idle_unload_timeout_s,
    )
    return TpotOrchestrator(
        registry=registry,
        admission=admission,
        backends=backends,
        writer=LlamaWriterClient(base_url=cfg.writer_url, registry=registry, timeout_s=cfg.backend_call_timeout_s),
        scorer=HttpScorerClient(base_url=cfg.scorer_url, timeout_s=cfg.backend_call_timeout_s),
    )


def create_app(*, orchestrator=None, token: str | None = None) -> FastAPI:
    app = FastAPI(title="TPOT Inference API")
    app.state.orchestrator = orchestrator or _default_orchestrator()
    app.state.token = token if token is not None else os.environ.get("TPOT_INFERENCE_TOKEN")

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        expected = app.state.token
        if expected is None:
            raise HTTPException(status_code=503, detail="TPOT_INFERENCE_TOKEN is not configured")
        if authorization != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="missing or invalid bearer token")

    @app.exception_handler(ServiceError)
    async def service_error_handler(_request: Request, exc: ServiceError):
        return _service_error_response(exc)

    @app.get("/health")
    def health():
        return app.state.orchestrator.health()

    @app.get("/v1/tpot/models", dependencies=[Depends(require_auth)])
    def models():
        return {"model_versions": app.state.orchestrator.models()}

    @app.post("/v1/tpot/score", dependencies=[Depends(require_auth)])
    def score(req: ScoreRequest):
        return app.state.orchestrator.score(req.texts)

    @app.post("/v1/tpot/ideate", dependencies=[Depends(require_auth)])
    def ideate(req: IdeateRequest):
        return app.state.orchestrator.ideate(**req.model_dump())

    @app.post("/v1/tpot/improve", dependencies=[Depends(require_auth)])
    def improve(req: ImproveRequest):
        return app.state.orchestrator.improve(**req.model_dump())

    @app.post("/v1/tpot/batch", dependencies=[Depends(require_auth)])
    def batch(req: BatchRequest):
        if hasattr(app.state.orchestrator, "batch"):
            return app.state.orchestrator.batch(req)
        deadline = time.monotonic() + req.timeout_s
        results = []
        for raw in req.requests:
            if time.monotonic() > deadline:
                results.append(_batch_error(str(raw.get("id", "")), "deadline_exceeded", "Batch deadline elapsed before this subrequest started."))
            else:
                results.append(_execute_batch_item(app.state.orchestrator, raw))
        return {"model_versions": app.state.orchestrator.models(), "results": results}

    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        @app.get("/metrics")
        def metrics():
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    except Exception:  # noqa: BLE001 - optional dependency until service extra is installed.
        pass

    return app


app = create_app()
