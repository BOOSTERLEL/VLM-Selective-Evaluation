"""Prompt builders for structured and free-text modes."""

from __future__ import annotations

from vlm_selective_eval.config import PromptConfig


STRUCTURED_INSTRUCTIONS = (
    "Return ONLY valid JSON using exactly these keys: "
    '{"answer": "<string>", "status": "ANSWERABLE|UNANSWERABLE", "confidence": <float_0_to_1>}. '
    "Do not include markdown fences, extra keys, or additional text. "
    "If status is UNANSWERABLE, use an empty answer string when needed."
)


FREE_TEXT_INSTRUCTIONS = (
    "Answer the question about the image. Include your answerability decision "
    "(ANSWERABLE or UNANSWERABLE) and a confidence score in [0,1]."
)

FREE_TEXT_TAGGED_INSTRUCTIONS = (
    "Answer in plain text using exactly three lines:\n"
    "Answer: <short answer or empty string>\n"
    "Status: ANSWERABLE or UNANSWERABLE\n"
    "Confidence: <float in [0,1]>\n"
    "Do not output JSON or markdown code fences."
)


def normalize_prompt_mode(mode: str) -> str:
    key = mode.lower().strip().replace("-", "_")
    if key in {"structured", "structured_json", "structured_strict"}:
        return "structured"
    if key in {"free_text", "freetext", "baseline", "free_text_loose"}:
        return "free_text"
    if key in {"free_text_tagged", "free_text_labeled", "free_text_robust"}:
        return "free_text_tagged"
    raise ValueError(f"Unsupported prompt mode: {mode}")


def is_structured_prompt_mode(mode: str) -> bool:
    return normalize_prompt_mode(mode) == "structured"


def build_prompt(question: str, prompt_config: PromptConfig) -> str:
    """Build prompt text for a single image-question input."""
    mode = normalize_prompt_mode(prompt_config.mode)
    parts: list[str] = []
    if prompt_config.include_system and prompt_config.system_text:
        parts.append(f"System: {prompt_config.system_text}")
    if mode == "structured":
        parts.append(f"Instruction: {STRUCTURED_INSTRUCTIONS}")
    elif mode == "free_text":
        parts.append(f"Instruction: {FREE_TEXT_INSTRUCTIONS}")
    elif mode == "free_text_tagged":
        parts.append(f"Instruction: {FREE_TEXT_TAGGED_INSTRUCTIONS}")
    else:
        raise ValueError(f"Unsupported prompt mode: {prompt_config.mode}")
    parts.append(f"Question: {question}")
    return "\n".join(parts).strip()
