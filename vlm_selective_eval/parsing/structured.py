"""Structured JSON parsing with regex fallback."""

from __future__ import annotations

import json
import re
from typing import Any

from vlm_selective_eval.constants import VALID_STATUSES
from vlm_selective_eval.schemas import ParsedOutput


EXPECTED_KEYS = {"answer", "status", "confidence"}


def _coerce_confidence(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        conf = float(value)
    elif isinstance(value, str):
        raw = value.strip()
        if raw.endswith("%"):
            try:
                conf = float(raw[:-1]) / 100.0
            except ValueError:
                return None
        else:
            try:
                conf = float(raw)
            except ValueError:
                return None
    else:
        return None
    if 0.0 <= conf <= 1.0:
        return conf
    return None


def parse_strict_json_output(raw_output: str) -> ParsedOutput:
    """Strictly parse an output expected to be a JSON object."""
    notes: list[str] = []
    text = raw_output.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return ParsedOutput(
            answer=None,
            status=None,
            confidence=None,
            parse_ok=False,
            format_ok=False,
            parser_name="strict_json",
            notes=["json_decode_error"],
        )
    if not isinstance(data, dict):
        return ParsedOutput(
            answer=None,
            status=None,
            confidence=None,
            parse_ok=False,
            format_ok=False,
            parser_name="strict_json",
            notes=["json_root_not_object"],
        )

    has_all_fields = EXPECTED_KEYS.issubset(data.keys())
    answer = data.get("answer")
    status = data.get("status")
    confidence = _coerce_confidence(data.get("confidence"))

    if not has_all_fields:
        notes.append("missing_expected_keys")
    if set(data.keys()) != EXPECTED_KEYS:
        notes.append("keys_not_exact")
    if not isinstance(answer, str):
        notes.append("answer_not_string")
    if not isinstance(status, str) or status not in VALID_STATUSES:
        notes.append("invalid_status")
    if confidence is None:
        notes.append("invalid_confidence")

    parse_ok = (
        has_all_fields
        and isinstance(answer, str)
        and isinstance(status, str)
        and confidence is not None
    )
    format_ok = parse_ok and set(data.keys()) == EXPECTED_KEYS and status in VALID_STATUSES

    return ParsedOutput(
        answer=answer if isinstance(answer, str) else None,
        status=status if isinstance(status, str) else None,
        confidence=confidence,
        parse_ok=parse_ok,
        format_ok=format_ok,
        parser_name="strict_json",
        status_reliable=isinstance(status, str) and status in VALID_STATUSES,
        confidence_reliable=confidence is not None,
        notes=notes,
    )


def parse_regex_fallback_output(raw_output: str) -> ParsedOutput:
    """Best-effort parser for malformed structured outputs."""
    text = raw_output.strip()
    notes: list[str] = []

    status_match = re.search(r"\b(UNANSWERABLE|ANSWERABLE)\b", text, flags=re.IGNORECASE)
    status = status_match.group(1).upper() if status_match else None
    status_reliable = status in VALID_STATUSES
    if not status_reliable:
        notes.append("status_not_found")

    conf_match = re.search(
        r"confidence\s*[:=]\s*['\"]?\s*([0-9]+(?:\.[0-9]+)?%?)",
        text,
        flags=re.IGNORECASE,
    )
    confidence = _coerce_confidence(conf_match.group(1)) if conf_match else None
    confidence_reliable = confidence is not None
    if not confidence_reliable:
        notes.append("confidence_not_found")

    answer: str | None = None
    answer_patterns = [
        r'"answer"\s*:\s*"([^"]*)"',
        r"'answer'\s*:\s*'([^']*)'",
        r"answer\s*[:=]\s*([^;\n]+)",
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            answer = match.group(1).strip().strip('"').strip("'")
            break

    if answer is None and status == "UNANSWERABLE":
        answer = ""
    if answer is None:
        notes.append("answer_not_found")

    parse_ok = status_reliable and confidence_reliable and answer is not None
    return ParsedOutput(
        answer=answer,
        status=status if status_reliable else None,
        confidence=confidence,
        parse_ok=parse_ok,
        format_ok=False,
        parser_name="regex_fallback",
        status_reliable=status_reliable,
        confidence_reliable=confidence_reliable,
        notes=notes,
    )


def parse_structured_output(raw_output: str) -> ParsedOutput:
    """Run strict JSON parsing first, then fallback regex parsing."""
    strict = parse_strict_json_output(raw_output)
    if strict.parse_ok:
        return strict
    fallback = parse_regex_fallback_output(raw_output)
    if fallback.parse_ok:
        return fallback
    notes = [*strict.notes, *fallback.notes]
    return ParsedOutput(
        answer=fallback.answer,
        status=fallback.status,
        confidence=fallback.confidence,
        parse_ok=False,
        format_ok=False,
        parser_name="parse_failed",
        status_reliable=fallback.status_reliable,
        confidence_reliable=fallback.confidence_reliable,
        notes=notes,
    )
