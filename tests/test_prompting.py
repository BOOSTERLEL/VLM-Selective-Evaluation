from vlm_selective_eval.config import PromptConfig
from vlm_selective_eval.prompting import build_prompt, normalize_prompt_mode


def test_prompt_mode_aliases():
    assert normalize_prompt_mode("structured_strict") == "structured"
    assert normalize_prompt_mode("freetext") == "free_text"
    assert normalize_prompt_mode("free-text-tagged") == "free_text_tagged"


def test_build_prompt_free_text_tagged_instructions():
    cfg = PromptConfig(mode="free_text_tagged", include_system=False, system_text="")
    prompt = build_prompt("What is shown?", cfg)
    assert "Answer: <short answer or empty string>" in prompt
    assert "Status: ANSWERABLE or UNANSWERABLE" in prompt
    assert "Confidence: <float in [0,1]>" in prompt
