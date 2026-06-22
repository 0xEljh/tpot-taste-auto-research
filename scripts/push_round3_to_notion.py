"""Push the round-3 active-labeling batch to the calibration Notion DB (doc 05 §8).

CONTAMINATION-SAFETY: pull_calibration_labels.py reads any *Select* prop as the human label.
So this script sets ONLY the Number (id) + Title (text) properties and writes MY proposal /
confidence / reason into a rich_text prop (or the page body) — it NEVER sets the Taste Select.
The Select stays empty until the human verifies, so pull ingests nothing of mine.

  uv run python scripts/push_round3_to_notion.py --inspect          # print DB schema + a sample row
  uv run python scripts/push_round3_to_notion.py --dry-run          # show the payload for row 1, POST nothing
  uv run python scripts/push_round3_to_notion.py                    # create all rows
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

ENV = Path("/home/elijah/dotfiles/scripts/.env")
DS_ID = "fae195b7-551b-4e01-a70c-3e10b5f8a76f"
VER = "2025-09-03"


def _token() -> str:
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if line.startswith("NOTION_TOKEN") and "=" in line:
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("NOTION_TOKEN not found")


def _req(url: str, token: str, method: str = "GET", body: dict | None = None) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode() if body is not None else None, method=method,
        headers={"Authorization": f"Bearer {token}", "Notion-Version": VER,
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} on {method} {url.split('/v1/')[1]}: {e.read().decode()[:300]}")


def _schema(token: str):
    ds = _req(f"https://api.notion.com/v1/data_sources/{DS_ID}", token)
    props = ds.get("properties", {})
    title = number = note = None
    selects = []
    for name, spec in props.items():
        t = spec.get("type")
        if t == "title":
            title = name
        elif t == "number":
            number = name
        elif t == "rich_text" and note is None:
            note = name
        elif t == "select":
            selects.append(name)
    return props, title, number, note, selects


@app.command()
def main(
    candidates: Path = Path("data/splits/round3_candidates.parquet"),
    prelabels: Path = Path("outputs/round3_prelabels.json"),
    inspect: bool = False,
    dry_run: bool = False,
    limit: int = 0,
) -> None:
    import polars as pl

    token = _token()
    props, title, number, note, selects = _schema(token)
    print(f"[schema] title={title!r} number={number!r} rich_text-note={note!r} selects={selects}")
    for name, spec in props.items():
        opts = ""
        if spec.get("type") == "select":
            opts = " options=" + str([o["name"] for o in spec["select"].get("options", [])])
        print(f"    - {name!r}: {spec.get('type')}{opts}")

    if inspect:
        sample = _req(f"https://api.notion.com/v1/data_sources/{DS_ID}/query", token, "POST",
                      {"page_size": 1})
        rows = sample.get("results", [])
        if rows:
            print("\n[sample row properties]")
            for name, p in rows[0].get("properties", {}).items():
                print(f"    - {name!r}: {p.get('type')} = {str(p.get(p.get('type')))[:90]}")
        return

    pre = json.loads(prelabels.read_text())
    rows = pl.read_parquet(candidates).to_dicts()
    if limit:
        rows = rows[:limit]
    create_url = "https://api.notion.com/v1/pages"
    made = 0
    for r in rows:
        pl_ = pre.get(str(r["id"]), {})
        note_txt = (f"PROPOSED {pl_.get('p','?')} ({pl_.get('c','?')} conf) — {pl_.get('r','')}"
                    f"  [bucket={r['bucket']}, ridge={r['ridge']}, v72={r['v72']}]")
        properties = {
            title: {"title": [{"type": "text", "text": {"content": r["text"][:1900]}}]},
            number: {"number": r["id"]},
        }
        body: dict = {"parent": {"data_source_id": DS_ID}, "properties": properties}
        if note:
            properties[note] = {"rich_text": [{"type": "text", "text": {"content": note_txt[:1900]}}]}
        else:
            body["children"] = [{"object": "block", "type": "callout", "callout": {
                "rich_text": [{"type": "text", "text": {"content": note_txt[:1900]}}],
                "icon": {"emoji": "🤖"}}}]
        if dry_run:
            print("\n[dry-run] would POST:")
            print(json.dumps(body, ensure_ascii=False, indent=2)[:1200])
            print(f"  (Taste select left UNSET — verified: selects={selects})")
            return
        _req(create_url, token, "POST", body)
        made += 1
        time.sleep(0.34)  # ~3 req/s
    print(f"\n[done] created {made} round-3 rows (ids {rows[0]['id']}–{rows[-1]['id']}); Taste left empty for you to verify")


if __name__ == "__main__":
    app()
