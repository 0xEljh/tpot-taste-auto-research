"""Pull the human's calibration labels from the Notion DB via the standard API (Phase 6e, D36).

The MCP query tool is Enterprise-gated; the plain Notion API isn't. Uses NOTION_TOKEN (the notion-cat
integration, which has access to the "Bread" tree where the calibration DB lives). Stdlib only.

  uv run python scripts/pull_calibration_labels.py
Writes outputs/calibration_labels.json: {"<id>": "<Taste>"} for labeled rows.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

ENV = Path("/home/elijah/dotfiles/scripts/.env")
DB_ID = "8d281b6e213c4f83b2a824abba0e073f"
DS_ID = "fae195b7-551b-4e01-a70c-3e10b5f8a76f"


def _token() -> str:
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if line.startswith("NOTION_TOKEN") and "=" in line:
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("NOTION_TOKEN not found")


def _post(url: str, token: str, version: str, body: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {token}", "Notion-Version": version,
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def main() -> None:
    token = _token()
    # try the new data-source endpoint first, fall back to the classic databases endpoint
    endpoints = [(f"https://api.notion.com/v1/data_sources/{DS_ID}/query", "2025-09-03"),
                 (f"https://api.notion.com/v1/databases/{DB_ID}/query", "2022-06-28")]
    results, used = [], None
    for url, ver in endpoints:
        try:
            cursor = None
            results = []
            while True:
                body = {"page_size": 100}
                if cursor:
                    body["start_cursor"] = cursor
                r = _post(url, token, ver, body)
                results += r["results"]
                if r.get("has_more"):
                    cursor = r["next_cursor"]
                else:
                    break
            used = (url, ver)
            break
        except Exception as e:  # noqa: BLE001
            print(f"  [{ver}] {url.split('/v1/')[1]} failed: {str(e)[:80]}")
    if used is None:
        raise SystemExit("both query endpoints failed")
    print(f"[ok] {used[1]} {used[0].split('/v1/')[1]} -> {len(results)} rows")

    labels = {}
    for pg in results:
        props = pg.get("properties", {})
        idv = taste = None
        for p in props.values():
            if p.get("type") == "number" and p.get("number") is not None:
                idv = int(p["number"])
            if p.get("type") == "select":
                taste = p["select"]["name"] if p.get("select") else None
        if idv is not None:
            labels[str(idv)] = taste
    out = Path("outputs/calibration_labels.json")
    out.write_text(json.dumps(labels, ensure_ascii=False, indent=0))
    n_lab = sum(1 for v in labels.values() if v)
    print(f"[write] {out}: {len(labels)} rows, {n_lab} labeled, {len(labels)-n_lab} blank")


if __name__ == "__main__":
    main()
