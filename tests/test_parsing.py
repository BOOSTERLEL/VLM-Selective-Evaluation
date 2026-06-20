from vlm_selective_eval.parsing.structured import (
    parse_regex_fallback_output,
    parse_strict_json_output,
)


def test_strict_json_parsing_valid():
    raw = '{"answer": "cat", "status": "ANSWERABLE", "confidence": 0.91}'
    parsed = parse_strict_json_output(raw)
    assert parsed.parse_ok is True
    assert parsed.format_ok is True
    assert parsed.answer == "cat"
    assert parsed.status == "ANSWERABLE"
    assert parsed.confidence == 0.91


def test_regex_fallback_parsing_malformed():
    raw = "answer=cat; status=ANSWERABLE; confidence=0.84"
    parsed = parse_regex_fallback_output(raw)
    assert parsed.parse_ok is True
    assert parsed.format_ok is False
    assert parsed.answer == "cat"
    assert parsed.status == "ANSWERABLE"
    assert parsed.confidence == 0.84
