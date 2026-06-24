"""TasteEngine — the integration of the two locked models (Phase 5).

Writer (CausalLM) generates; Scorer (SequenceClassification) judges. ideate/improve do best-of-N:
generate N candidates, rank by the Scorer, return the top. Each adapter loads lazily on its OWN
4-bit base (resolve_adapter_base reads each adapter's recorded base — they need not match, e.g. an
8B scorer + 4B writer), so `score` doesn't pay for the Writer, and vice versa.

Pure helpers (build_prompt, rank_topk) are unit-tested; the model paths are smoke-tested via the CLI.
"""
from __future__ import annotations

from pathlib import Path

from tpot_taste.writer.sft_data import IDEATE_PROMPTS, IMPROVE_PROMPTS, SYS_PROMPT


def build_prompt(kind: str, *, topic: str | None = None, draft: str | None = None) -> str:
    """The user-turn text for a given task. ideate optionally conditions on a topic; improve on a draft."""
    if kind == "ideate":
        return f"Write a tpot post about {topic.strip()}." if topic else IDEATE_PROMPTS[0]
    if kind == "improve":
        if not draft:
            raise ValueError("improve needs a draft")
        return IMPROVE_PROMPTS[0].format(draft=draft.strip())
    raise ValueError(f"unknown kind: {kind}")


def resolve_adapter_base(adapter_path: Path | str, fallback: str) -> str:
    """An adapter's OWN base model id, read from its PEFT `adapter_config.json`.

    The Scorer and Writer no longer share a base (D43/§10: an 8B scorer alongside a 3B/4B writer).
    PEFT records each adapter's base, so the engine loads every adapter on the base it was trained
    on instead of assuming one. Falls back to `fallback` for legacy adapters that don't record one."""
    import json

    cfg = Path(adapter_path) / "adapter_config.json"
    if cfg.exists():
        base = json.loads(cfg.read_text()).get("base_model_name_or_path")
        if base:
            return base
    return fallback


# back-compat alias: the resolver is generic (used for both Writer and Scorer adapters)
resolve_scorer_base = resolve_adapter_base


def rank_topk(candidates: list[str], scores: list[float], k: int) -> list[tuple[str, float]]:
    """Top-k (text, score) by score desc, dropping blanks and exact-duplicate texts."""
    seen: set[str] = set()
    ranked: list[tuple[str, float]] = []
    for t, s in sorted(zip(candidates, scores), key=lambda x: -x[1]):
        key = " ".join(t.split()).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        ranked.append((t, float(s)))
    return ranked[:k]


class TasteEngine:
    def __init__(
        self,
        writer: Path | str = "outputs/writer/qwen3b-writer-LOCKED",
        scorer: Path | str = "outputs/scorer/qwen3b-bt-taste-LOCKED",
        base: str = "Qwen/Qwen2.5-3B-Instruct",
    ) -> None:
        self.writer_path, self.scorer_path, self.base = str(writer), str(scorer), base
        self._w = self._wtok = self._s = self._stok = None

    # ---- lazy loaders ----
    def _ensure_writer(self):
        if self._w is None:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            writer_base = resolve_adapter_base(self.writer_path, self.base)
            tok = AutoTokenizer.from_pretrained(writer_base)
            tok.padding_side = "left"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            m = AutoModelForCausalLM.from_pretrained(writer_base, quantization_config=bnb, dtype=torch.bfloat16)
            self._w = PeftModel.from_pretrained(m, self.writer_path).eval()
            self._wtok = tok
        return self._w, self._wtok

    def _ensure_scorer(self):
        if self._s is None:
            from tpot_taste.scoring.model import load_trained_scorer
            scorer_base = resolve_scorer_base(self.scorer_path, self.base)
            self._s, self._stok = load_trained_scorer(self.scorer_path, scorer_base)
        return self._s, self._stok

    # ---- capabilities ----
    def score(self, texts: list[str]) -> list[float]:
        from tpot_taste.scoring.model import score_texts
        m, tok = self._ensure_scorer()
        return score_texts(m, tok, texts, batch_size=32)

    def _generate(self, user: str, n: int, temperature: float, max_new_tokens: int) -> list[str]:
        import re

        import torch
        m, tok = self._ensure_writer()
        rendered = tok.apply_chat_template(
            [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )
        enc = tok([rendered] * n, return_tensors="pt", padding=True, truncation=True, max_length=320).to(m.device)
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                           top_p=0.95, pad_token_id=tok.pad_token_id)
        pre = re.compile(r"^(sure|here(\'s| is)|okay|ok|certainly)[,:]?\s*", re.I)
        outs = []
        for j in range(n):
            t = tok.decode(g[j][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().strip('"').strip()
            outs.append(" ".join(pre.sub("", t).split()))
        return outs

    def ideate(self, topic: str | None = None, *, k: int = 3, best_of: int = 8,
               temperature: float = 0.9, max_new_tokens: int = 64) -> list[tuple[str, float]]:
        cands = self._generate(build_prompt("ideate", topic=topic), best_of, temperature, max_new_tokens)
        return rank_topk(cands, self.score(cands), k)

    def improve(self, draft: str, *, k: int = 3, best_of: int = 8,
                temperature: float = 0.9, max_new_tokens: int = 64) -> list[tuple[str, float]]:
        cands = self._generate(build_prompt("improve", draft=draft), best_of, temperature, max_new_tokens)
        ranked = rank_topk(cands, self.score(cands), k)
        return ranked
