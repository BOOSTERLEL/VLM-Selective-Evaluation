from vlm_selective_eval.config import ModelConfig, PromptConfig
from vlm_selective_eval.models.mock_vlm import MockVLMAdapter
from vlm_selective_eval.parsing.free_text import parse_free_text_output
from vlm_selective_eval.prompting import build_prompt


def _new_mock(seed: int = 7) -> MockVLMAdapter:
    cfg = ModelConfig(
        adapter="mock",
        model_name="mock-vlm",
        free_text_missing_field_rate=0.4,
        answer_error_rate=0.1,
        hallucination_rate=0.1,
        malformed_rate=0.0,
    )
    return MockVLMAdapter(config=cfg, seed=seed)


def test_mock_tagged_output_has_three_lines():
    model = _new_mock(seed=13)
    prompt = build_prompt("What is shown?", PromptConfig(mode="free_text_tagged"))
    output = model.generate(image_path="img__answerable__cat.png", prompt=prompt)
    lines = output.splitlines()
    assert len(lines) >= 3
    assert lines[0].startswith("Answer:")
    assert lines[1].startswith("Status:")
    assert lines[2].startswith("Confidence:")


def test_mock_tagged_parse_success_not_worse_than_free_text():
    n = 120
    image = "img__answerable__cat.png"

    free_model = _new_mock(seed=101)
    tagged_model = _new_mock(seed=101)

    free_prompt = build_prompt("What animal?", PromptConfig(mode="free_text"))
    tagged_prompt = build_prompt("What animal?", PromptConfig(mode="free_text_tagged"))

    free_ok = 0
    tagged_ok = 0
    for _ in range(n):
        free_out = free_model.generate(image_path=image, prompt=free_prompt)
        tagged_out = tagged_model.generate(image_path=image, prompt=tagged_prompt)
        free_ok += int(parse_free_text_output(free_out).parse_ok)
        tagged_ok += int(parse_free_text_output(tagged_out).parse_ok)

    assert tagged_ok >= free_ok
