"""Local serving artifact classification and registry generation."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tpot_taste.adapters import read_adapter_config


class ArtifactExportError(ValueError):
    pass


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class AdapterClassification:
    adapter_dir: Path
    base_model_id: str | None
    peft_task_type: str | None
    modules_to_save: Any
    adapter_config_sha256: str

    @property
    def export_error(self) -> str | None:
        if self.peft_task_type != "CAUSAL_LM":
            return "unsupported_task_type"
        if self.modules_to_save is not None:
            return "modules_to_save_not_supported"
        return None

    @property
    def exportable_to_llama_cpp(self) -> bool:
        return self.export_error is None

    def assert_exportable(self) -> None:
        if self.peft_task_type != "CAUSAL_LM":
            raise ArtifactExportError(f"only CAUSAL_LM adapters can be exported to llama.cpp, got {self.peft_task_type!r}")
        if self.modules_to_save is not None:
            raise ArtifactExportError("runtime GGUF LoRA adapters cannot represent modules_to_save")


def classify_adapter(adapter_dir: Path | str) -> AdapterClassification:
    adapter_path = Path(adapter_dir)
    cfg = read_adapter_config(adapter_path)
    return AdapterClassification(
        adapter_dir=adapter_path,
        base_model_id=cfg.get("base_model_name_or_path"),
        peft_task_type=cfg.get("task_type"),
        modules_to_save=cfg.get("modules_to_save"),
        adapter_config_sha256=sha256_file(adapter_path / "adapter_config.json"),
    )


def generate_writer_registry(
    *,
    adapter_dir: Path | str,
    adapter_source_alias: str,
    base_gguf: Path | str,
    lora_gguf: Path | str,
    quantization: str,
    base_source_url: str,
    llama_cpp_image: str,
    llama_cpp_build_sha: str,
    chat_template_hash: str,
    converter_requirements_hash: str,
    acceptance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    info = classify_adapter(adapter_dir)
    info.assert_exportable()
    base_path = Path(base_gguf)
    lora_path = Path(lora_gguf)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "models": {
            "writer": {
                "version": f"{Path(adapter_source_alias).name}.{quantization}",
                "adapter_source_alias": adapter_source_alias,
                "adapter_dir": str(Path(adapter_dir)),
                "base_model_id": info.base_model_id,
                "peft_task_type": info.peft_task_type,
                "modules_to_save": info.modules_to_save,
                "adapter_config_sha256": info.adapter_config_sha256,
                "base_gguf": str(base_path),
                "base_gguf_sha256": sha256_file(base_path),
                "lora_gguf": str(lora_path),
                "lora_gguf_sha256": sha256_file(lora_path),
                "quantization": quantization,
                "base_source_url": base_source_url,
                "llama_cpp_image": llama_cpp_image,
                "llama_cpp_build_sha": llama_cpp_build_sha,
                "chat_template_hash": chat_template_hash,
                "converter_requirements_hash": converter_requirements_hash,
                "acceptance": acceptance or {},
            }
        },
    }


def write_registry(path: Path | str, registry: dict[str, Any]) -> None:
    import json

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
