"""IO helpers for JSONL/JSON/CSV outputs."""

from __future__ import annotations

from pathlib import Path
import csv
import json
from typing import Iterable, Mapping, Any


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ensure_parent_dir(path)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    ensure_parent_dir(path)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    ensure_parent_dir(path)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
