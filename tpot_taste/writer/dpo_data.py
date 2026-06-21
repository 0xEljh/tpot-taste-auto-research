"""Form DPO preference pairs from best-of-N scored candidates (Phase 4, D5).

Pure, testable core: given per-prompt candidate generations + their v6-scorer scores, emit
conversational DPO records (chosen = highest, rejected = lowest) for prompts whose score spread
clears a margin (drops scorer-noise mislabels). The generate+score glue lives in
scripts/build_dpo_pairs.py.
"""
from __future__ import annotations

from collections.abc import Iterable


def form_dpo_pairs(
    groups: Iterable[dict],
    *,
    sys_prompt: str,
    min_margin: float = 1.0,
) -> list[dict]:
    """groups: dicts {"user": str, "task": str, "candidates": [(text, score), ...]}.

    Returns DPO records {task, margin, prompt:[system,user], chosen:[assistant], rejected:[assistant]}
    where (max_score - min_score) >= min_margin and chosen text != rejected text.
    """
    out: list[dict] = []
    for g in groups:
        cands = [(t, float(s)) for t, s in g["candidates"] if t and t.strip()]
        if len(cands) < 2:
            continue
        cands.sort(key=lambda x: x[1])
        worst_t, worst_s = cands[0]
        best_t, best_s = cands[-1]
        if best_s - worst_s < min_margin:
            continue
        if best_t.strip().lower() == worst_t.strip().lower():
            continue
        out.append(
            {
                "task": g.get("task", ""),
                "margin": best_s - worst_s,
                "prompt": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": g["user"]},
                ],
                "chosen": [{"role": "assistant", "content": best_t}],
                "rejected": [{"role": "assistant", "content": worst_t}],
            }
        )
    return out
