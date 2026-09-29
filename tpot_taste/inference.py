"""Shared TPOT inference helpers used by the CLI and HTTP service."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from tpot_taste.writer.sft_data import IDEATE_PROMPTS, IMPROVE_PROMPTS, SYS_PROMPT


@dataclass(frozen=True)
class GenerationDefaults:
    k: int = 3
    best_of: int = 8
    temperature: float = 0.9
    top_p: float = 0.95
    max_new_tokens: int = 64
    prompt_max_length: int = 320


DEFAULT_GENERATION = GenerationDefaults()
_PLEASANTRY_RE = re.compile(r"^(sure|here(\'s| is)|okay|ok|certainly)[,:]?\s*", re.I)


def build_prompt(kind: str, *, topic: str | None = None, draft: str | None = None) -> str:
    """The user-turn text for a TPOT generation task."""
    if kind == "ideate":
        return f"Write a tpot post about {topic.strip()}." if topic else IDEATE_PROMPTS[0]
    if kind == "improve":
        if not draft:
            raise ValueError("improve needs a draft")
        return IMPROVE_PROMPTS[0].format(draft=draft.strip())
    raise ValueError(f"unknown kind: {kind}")


def chat_messages(user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user}]


def render_chat_prompt(tokenizer: Any, user: str) -> str:
    """Render exactly through the HF tokenizer chat template used by TasteEngine."""
    return tokenizer.apply_chat_template(chat_messages(user), tokenize=False, add_generation_prompt=True)


def tokenize_prompt_ids(tokenizer: Any, rendered_prompt: str, *, max_length: int = DEFAULT_GENERATION.prompt_max_length) -> list[int]:
    """Tokenize the rendered prompt with the HF tokenizer under the shared truncation limit."""
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(rendered_prompt, add_special_tokens=False, truncation=True, max_length=max_length))
    encoded = tokenizer(rendered_prompt, truncation=True, max_length=max_length, add_special_tokens=False)
    return list(encoded["input_ids"])


def _resolved_top_k(generation_config: Any | None) -> int:
    if generation_config is None:
        return 50
    top_k = getattr(generation_config, "top_k", None)
    return int(top_k) if top_k is not None else 50


def sampler_params(*, temperature: float, top_p: float, generation_config: Any | None) -> dict[str, float | int]:
    """llama.cpp sampler params pinned to match the effective HF generate call."""
    return {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": _resolved_top_k(generation_config),
        "min_p": 0,
        "repeat_penalty": 1.0,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }


def build_writer_completion_payload(
    tokenizer: Any,
    user: str,
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    generation_config: Any | None = None,
) -> dict[str, Any]:
    rendered = render_chat_prompt(tokenizer, user)
    payload: dict[str, Any] = {
        "prompt": tokenize_prompt_ids(tokenizer, rendered),
        "n_predict": int(max_new_tokens),
        "cache_prompt": False,
    }
    payload.update(sampler_params(temperature=temperature, top_p=top_p, generation_config=generation_config or getattr(tokenizer, "generation_config", None)))
    return payload


def normalize_generated_text(text: str) -> str:
    t = str(text).strip()
    t = _PLEASANTRY_RE.sub("", t).strip().strip('"').strip("'").strip()
    t = _PLEASANTRY_RE.sub("", t).strip().strip('"').strip("'").strip()
    return " ".join(t.split())


def rank_topk(candidates: list[str], scores: list[float], k: int) -> list[tuple[str, float]]:
    """Top-k (text, score) by score desc, dropping blanks and exact-duplicate texts."""
    seen: set[str] = set()
    ranked: list[tuple[str, float]] = []
    for text, score in sorted(zip(candidates, scores), key=lambda item: -item[1]):
        key = " ".join(text.split()).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        ranked.append((text, float(score)))
    return ranked[:k]
