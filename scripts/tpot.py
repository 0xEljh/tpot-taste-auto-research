"""tpot — the integrated CLI (Phase 5): score / ideate / improve / repl.

Writer (DPO v2) generates, Scorer (v6) judges. ideate/improve do best-of-N and show the
scorer's ranking, so you see the two locked models working together.

  uv run python scripts/tpot.py score "is this worth posting?"
  uv run python scripts/tpot.py ideate --topic "recursion"
  uv run python scripts/tpot.py improve "lit a fake cig"
  uv run python scripts/tpot.py repl
"""
from __future__ import annotations

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False, help="tpot taste CLI")


def _engine():
    from tpot_taste.engine import TasteEngine
    return TasteEngine()


def _show(ranked: list[tuple[str, float]]) -> None:
    for i, (t, s) in enumerate(ranked, 1):
        print(f"  {i}. {s:+6.2f}  {t}")


@app.command(help="Score one or more posts with the locked v6 taste scorer (is it worth posting?).")
def score(texts: list[str] = typer.Argument(..., help="post text(s) to score")) -> None:
    eng = _engine()
    for t, s in zip(texts, eng.score(list(texts))):
        print(f"{s:+6.2f}  {t}")


@app.command(help="Generate tpot post ideas (best-of-N, scorer-ranked).")
def ideate(
    topic: str = typer.Option(None, help="optional topic to write about"),
    k: int = typer.Option(3, help="how many to show"),
    best_of: int = typer.Option(8, help="candidates to sample and rank"),
) -> None:
    print(f"== ideate{f' on “{topic}”' if topic else ''} (best of {best_of}) ==")
    _show(_engine().ideate(topic, k=k, best_of=best_of))


@app.command(help="Improve a draft for tpot (best-of-N, scorer-ranked).")
def improve(
    draft: str = typer.Argument(..., help="the weak draft to improve"),
    k: int = typer.Option(3, help="how many to show"),
    best_of: int = typer.Option(8, help="candidates to sample and rank"),
) -> None:
    eng = _engine()
    print(f"draft scores {eng.score([draft])[0]:+.2f}:  {draft}\n== improvements (best of {best_of}) ==")
    _show(eng.improve(draft, k=k, best_of=best_of))


@app.command(help="Interactive loop (loads the models once). Commands: score/ideate/improve/quit.")
def repl(best_of: int = typer.Option(8)) -> None:
    eng = _engine()
    print("tpot repl — try:  score <text>   |   ideate [topic]   |   improve <draft>   |   quit")
    while True:
        try:
            line = input("tpot> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        cmd, rest = cmd.lower(), rest.strip()
        if cmd in ("quit", "exit", "q"):
            break
        if cmd == "score" and rest:
            print(f"  {eng.score([rest])[0]:+6.2f}  {rest}")
        elif cmd == "ideate":
            _show(eng.ideate(rest or None, best_of=best_of))
        elif cmd == "improve" and rest:
            _show(eng.improve(rest, best_of=best_of))
        else:
            print("  ? usage: score <text> | ideate [topic] | improve <draft> | quit")


if __name__ == "__main__":
    app()
