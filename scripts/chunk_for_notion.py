"""Split long ``` code fences in a Markdown file so each fits Notion's 2000-char/block limit.

notion-cat (and the Notion API) reject any single code block whose text exceeds 2000 chars. This rewrites
a report's fenced blocks into multiple smaller fences (at line boundaries), leaving prose untouched.

  uv run python scripts/chunk_for_notion.py outputs/demo_eval.md > outputs/demo_eval_notion.md
"""
from __future__ import annotations

import sys
from pathlib import Path

MAX_CHARS = 1500
MAX_LINES = 22


def _split(body: list[str]) -> list[list[str]]:
    chunks, cur, n = [], [], 0
    for ln in body:
        if cur and (len(cur) >= MAX_LINES or n + len(ln) + 1 > MAX_CHARS):
            chunks.append(cur)
            cur, n = [], 0
        cur.append(ln)
        n += len(ln) + 1
    if cur:
        chunks.append(cur)
    return chunks


def chunk_md(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            fence = ln  # preserve language, e.g. ```python
            j = i + 1
            body: list[str] = []
            while j < len(lines) and not lines[j].startswith("```"):
                body.append(lines[j])
                j += 1
            for k, chunk in enumerate(_split(body)):
                if k:
                    out.append("")  # blank line between split blocks
                out.append(fence)
                out.extend(chunk)
                out.append("```")
            i = j + 1
        else:
            out.append(ln)
            i += 1
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    src = Path(sys.argv[1])
    sys.stdout.write(chunk_md(src.read_text()))
