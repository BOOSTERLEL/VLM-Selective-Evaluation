"""Heuristic extraction for free-text baseline outputs."""

from __future__ import annotations

import re

from vlm_selective_eval.constants import VALID_STATUSES
from vlm_selective_eval.schemas import ParsedOutput

_STATUS_LABEL_PATTERN = re.compile(
    r"\b(?:status|answerability|decision)\s*[:=\-]\s*(UNANSWERABLE|ANSWERABLE)\b",
    flags=re.IGNORECASE,
)
_STATUS_TOKEN_PATTERN = re.compile(r"\b(UNANSWERABLE|ANSWERABLE)\b", flags=re.IGNORECASE)
_NEGATIVE_STATUS_HINTS = (
    "cannot answer",
    "can't answer",
    "unable to answer",
    "insufficient information",
    "not enough information",
    "cannot determine",
    "can't determine",
    "unknown from image",
)

_CONFIDENCE_LABEL_PATTERNS = (
    re.compile(
        r"\b(?:confidence|conf|probability|likelihood|certainty|surety)\s*[:=\-]?\s*"
        r"([0-9]+(?:\.[0-9]+)?%?)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:confidence|conf)\s*(?:is|=|:)\s*(?:about|around)?\s*"
        r"([0-9]+(?:\.[0-9]+)?%?)",
        flags=re.IGNORECASE,
    ),
)
_CONFIDENCE_PERCENT_SURE_PATTERNS = (
    re.compile(r"\b([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:sure|confident)\b", flags=re.IGNORECASE),
    re.compile(
        r"\b(?:sure|confident)\s*(?:at|about|around)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        flags=re.IGNORECASE,
    ),
)
_CONFIDENCE_CONTEXTUAL_PATTERNS = (
    re.compile(
        r"\b(?:confidence|probability|likelihood)\b[^0-9]{0,20}([01](?:\.[0-9]+)?)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"([01](?:\.[0-9]+)?)\s*(?:confidence|probability|likelihood)\b",
        flags=re.IGNORECASE,
    ),
)

_ANSWER_LABEL_PATTERNS = (
    re.compile(
        r"(?:^|\n)\s*(?:final\s+)?answer\s*[:=\-]\s*(.*?)"
        r"(?=\n\s*(?:status|confidence|answerability|decision)\s*[:=\-]|$)",
        flags=re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?:^|\n)\s*(?:prediction|response)\s*[:=\-]\s*(.+?)"
        r"(?=\n\s*(?:status|confidence|answerability|decision)\s*[:=\-]|$)",
        flags=re.IGNORECASE | re.DOTALL,
    ),
)


def _clean_answer(raw: str) -> str:
    answer = raw.strip().strip("`").strip()
    if len(answer) >= 2 and answer[0] == answer[-1] and answer[0] in {'"', "'"}:
        answer = answer[1:-1].strip()
    return answer


def _coerce_confidence(raw: str, allow_percent_fallback: bool = False) -> float | None:
    text = raw.strip().rstrip(".,;")
    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1].strip()
    try:
        value = float(text)
    except ValueError:
        return None

    if is_percent:
        value /= 100.0
    elif allow_percent_fallback and value > 1.0 and value <= 100.0:
        value /= 100.0

    if 0.0 <= value <= 1.0:
        return value
    return None


def _extract_status(text: str) -> tuple[str | None, bool]:
    labeled = _STATUS_LABEL_PATTERN.findall(text)
    if labeled:
        status = labeled[-1].upper()
        return status, status in VALID_STATUSES

    low = text.lower()
    if any(phrase in low for phrase in _NEGATIVE_STATUS_HINTS):
        return "UNANSWERABLE", True

    tokens = [token.upper() for token in _STATUS_TOKEN_PATTERN.findall(text)]
    token_set = set(tokens)
    if len(token_set) == 1:
        status = next(iter(token_set))
        return status, status in VALID_STATUSES
    return None, False


def _extract_confidence(text: str) -> tuple[float | None, bool]:
    for pattern in _CONFIDENCE_LABEL_PATTERNS:
        matches = pattern.findall(text)
        if not matches:
            continue
        conf = _coerce_confidence(matches[-1], allow_percent_fallback=True)
        if conf is not None:
            return conf, True

    for pattern in _CONFIDENCE_PERCENT_SURE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        conf = _coerce_confidence(f"{match.group(1)}%")
        if conf is not None:
            return conf, True

    for pattern in _CONFIDENCE_CONTEXTUAL_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        conf = _coerce_confidence(match.group(1))
        if conf is not None:
            return conf, True

    return None, False


def _extract_answer(text: str, status: str | None) -> str | None:
    for pattern in _ANSWER_LABEL_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        answer = _clean_answer(match.group(1))
        if answer:
            return answer
        if status == "UNANSWERABLE":
            return ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()
        if re.fullmatch(r"(?:final\s+)?answer\s*[:=\-]\s*", line, flags=re.IGNORECASE):
            if status == "UNANSWERABLE":
                return ""
            continue
        if low.startswith(("status:", "status=", "confidence:", "confidence=")):
            continue
        if low.startswith(("answerability:", "answerability=", "decision:", "decision=")):
            continue
        inline_answer = re.match(r"^(?:final\s+)?answer\s*[:=\-]\s*(.*)$", line, flags=re.IGNORECASE)
        if inline_answer:
            answer = _clean_answer(inline_answer.group(1))
            if answer:
                return answer
            if status == "UNANSWERABLE":
                return ""
            continue
        if re.fullmatch(r"(?:answerable|unanswerable)\.?", line, flags=re.IGNORECASE):
            continue
        if "confidence" in low and re.search(r"[0-9]", line):
            continue
        if "status" in low and ("answerable" in low or "unanswerable" in low):
            continue
        answer = _clean_answer(line)
        if answer:
            return answer

    if status == "UNANSWERABLE":
        return ""

    first_sentence = re.split(r"[.!?\n]", text, maxsplit=1)[0].strip()
    if first_sentence and not re.search(r"\b(status|confidence)\b", first_sentence, flags=re.IGNORECASE):
        return _clean_answer(first_sentence)
    return None


def parse_free_text_output(raw_output: str) -> ParsedOutput:
    """Extract answer/status/confidence from unconstrained text."""
    text = raw_output.strip()
    notes: list[str] = []

    status, status_reliable = _extract_status(text)
    if not status_reliable:
        notes.append("status_not_reliably_extracted")

    confidence, confidence_reliable = _extract_confidence(text)
    if not confidence_reliable:
        notes.append("confidence_not_reliably_extracted")

    answer = _extract_answer(text, status=status)
    if answer is None:
        notes.append("answer_not_extracted")

    parse_ok = status_reliable and confidence_reliable and answer is not None
    format_ok = parse_ok
    return ParsedOutput(
        answer=answer,
        status=status if status in VALID_STATUSES else None,
        confidence=confidence,
        parse_ok=parse_ok,
        format_ok=format_ok,
        parser_name="free_text_extract_v2",
        status_reliable=status_reliable,
        confidence_reliable=confidence_reliable,
        notes=notes,
    )
