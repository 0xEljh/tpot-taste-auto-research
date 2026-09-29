"""TasteEngine — the integration of the two locked models (Phase 5).

Writer (CausalLM) generates; Scorer (SequenceClassification) judges. ideate/improve do best-of-N:
generate N candidates, rank by the Scorer, return the top. Each adapter loads lazily on its OWN
4-bit base (resolve_adapter_base reads each adapter's recorded base — they need not match, e.g. an
8B scorer + 4B writer), so `score` doesn't pay for the Writer, and vice versa.

Pure helpers (build_prompt, rank_topk) are unit-tested; the model paths are smoke-tested via the CLI.
"""
from __future__ import annotations

from pathlib import Path

from tpot_taste.adapters import resolve_adapter_base
from tpot_taste.inference import DEFAULT_GENERATION, build_prompt, normalize_generated_text, rank_topk, render_chat_prompt


# back-compat alias: the resolver is generic (used for both Writer and Scorer adapters)
resolve_scorer_base = resolve_adapter_base


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

    def _generate(self, user: str, n: int, temperature: float, max_new_tokens: int, top_p: float = DEFAULT_GENERATION.top_p) -> list[str]:
        import torch
        m, tok = self._ensure_writer()
        rendered = render_chat_prompt(tok, user)
        enc = tok([rendered] * n, return_tensors="pt", padding=True, truncation=True, max_length=320).to(m.device)
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                           top_p=top_p, pad_token_id=tok.pad_token_id)
        outs: list[str] = []
        for j in range(n):
            text = tok.decode(g[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            outs.append(normalize_generated_text(text))
        return outs

    def ideate(self, topic: str | None = None, *, k: int = 3, best_of: int = 8,
               temperature: float = 0.9, top_p: float = 0.95, max_new_tokens: int = 64) -> list[tuple[str, float]]:
        cands = self._generate(build_prompt("ideate", topic=topic), best_of, temperature, max_new_tokens, top_p)
        return rank_topk(cands, list(self.score(cands)), k)

    def improve(self, draft: str, *, k: int = 3, best_of: int = 8,
                temperature: float = 0.9, top_p: float = 0.95, max_new_tokens: int = 64) -> list[tuple[str, float]]:
        cands = self._generate(build_prompt("improve", draft=draft), best_of, temperature, max_new_tokens, top_p)
        ranked = rank_topk(cands, list(self.score(cands)), k)
        return ranked
