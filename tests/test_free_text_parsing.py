from vlm_selective_eval.parsing.free_text import parse_free_text_output


def test_free_text_tagged_lines_parse_ok():
    raw = "Answer: 42\nStatus: ANSWERABLE\nConfidence: 0.87"
    parsed = parse_free_text_output(raw)
    assert parsed.parse_ok is True
    assert parsed.format_ok is True
    assert parsed.answer == "42"
    assert parsed.status == "ANSWERABLE"
    assert parsed.confidence == 0.87


def test_free_text_extract_percent_confidence():
    raw = "The sign says OPEN.\nDecision: ANSWERABLE\nI am 82% sure."
    parsed = parse_free_text_output(raw)
    assert parsed.parse_ok is True
    assert parsed.status == "ANSWERABLE"
    assert parsed.answer == "The sign says OPEN."
    assert parsed.confidence == 0.82


def test_free_text_unanswerable_allows_empty_answer():
    raw = "Status: UNANSWERABLE\nConfidence: 0.35\nAnswer:"
    parsed = parse_free_text_output(raw)
    assert parsed.parse_ok is True
    assert parsed.status == "UNANSWERABLE"
    assert parsed.answer == ""
    assert parsed.confidence == 0.35


def test_free_text_ambiguous_status_not_reliable():
    raw = "Answer: cat\nConfidence: 0.8\nEither ANSWERABLE or UNANSWERABLE."
    parsed = parse_free_text_output(raw)
    assert parsed.parse_ok is False
    assert parsed.status is None
    assert parsed.confidence == 0.8
