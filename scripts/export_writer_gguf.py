"""Verify and register the TPOT writer GGUF serving artifacts.

This script intentionally treats conversion tooling and model downloads as pinned
operator inputs. It verifies the PEFT adapter can be represented as a runtime
GGUF LoRA, optionally runs a caller-provided conversion command, then records the
serving registry and export metadata with SHA256s.
"""
from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import typer

from tpot_taste.deployment.artifacts import classify_adapter, generate_writer_registry, write_registry

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    adapter: Path = Path("outputs/writer/qwen3b-writer-LOCKED"),
    adapter_source_alias: str = "outputs/writer/qwen3b-writer-LOCKED",
    serving_dir: Path = Path("models/serving/writer-qwen3-4b-sft-v3"),
    base_gguf: Path = Path("models/serving/writer-qwen3-4b-sft-v3/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
    lora_gguf: Path = Path("models/serving/writer-qwen3-4b-sft-v3/tpot-writer-qwen3-4b-sft-v3-lora.gguf"),
    registry_out: Path = Path("models/serving/registry.json"),
    quantization: str = "Q4_K_M",
    base_source_url: str = "https://huggingface.co/ggml-org/Qwen3-4B-Instruct-2507-GGUF",
    llama_cpp_image: str = typer.Option(..., help="Pinned llama.cpp image, including digest."),
    llama_cpp_build_sha: str = typer.Option(..., help="llama.cpp build SHA verified by smoke test."),
    chat_template_hash: str = typer.Option(..., help="sha256:<hash> of the HF chat template used for serving."),
    converter_requirements_hash: str = typer.Option(..., help="sha256:<hash> for llama.cpp converter requirements."),
    convert_command: str | None = typer.Option(
        None,
        help="Optional command template to create the LoRA GGUF. {adapter} and {out} are substituted before shlex splitting.",
    ),
) -> None:
    info = classify_adapter(adapter)
    info.assert_exportable()
    serving_dir.mkdir(parents=True, exist_ok=True)

    if convert_command:
        command = convert_command.format(adapter=str(adapter), out=str(lora_gguf))
        subprocess.run(shlex.split(command), check=True)

    if not base_gguf.exists():
        raise typer.BadParameter(f"base GGUF does not exist: {base_gguf}")
    if not lora_gguf.exists():
        raise typer.BadParameter(f"LoRA GGUF does not exist: {lora_gguf}")

    registry = generate_writer_registry(
        adapter_dir=adapter,
        adapter_source_alias=adapter_source_alias,
        base_gguf=base_gguf,
        lora_gguf=lora_gguf,
        quantization=quantization,
        base_source_url=base_source_url,
        llama_cpp_image=llama_cpp_image,
        llama_cpp_build_sha=llama_cpp_build_sha,
        chat_template_hash=chat_template_hash,
        converter_requirements_hash=converter_requirements_hash,
    )
    write_registry(registry_out, registry)
    export_metadata = registry["models"]["writer"]
    (serving_dir / "export-metadata.json").write_text(json.dumps(export_metadata, indent=2, sort_keys=True) + "\n")
    print(f"[ok] wrote {registry_out} and {serving_dir / 'export-metadata.json'}")


if __name__ == "__main__":
    app()
