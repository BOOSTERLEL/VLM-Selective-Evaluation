"""TextVQA official-style scoring utilities (multi-answer soft accuracy)."""

from __future__ import annotations

import re
from typing import Any

_ARTICLES = {"a", "an", "the"}
_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "im": "i'm",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    "lets": "let's",
    "mightve": "might've",
    "mightnt": "mightn't",
    "mustve": "must've",
    "mustnt": "mustn't",
    "shant": "shan't",
    "shed": "she'd",
    "shell": "she'll",
    "shes": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "somebodyd": "somebody'd",
    "somebodyd've": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "wed": "we'd",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whod": "who'd",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}
_NUMBER_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def normalize_vqa_answer(text: str | None) -> str:
    """Normalize answer using VQA-style punctuation/article/number rules."""
    if text is None:
        return ""
    out = text.lower().strip().replace("\n", " ").replace("\t", " ")
    out = re.sub(r"(?<=\d),(?=\d)", "", out)
    out = re.sub(r"[^\w\s']", " ", out)
    out = re.sub(r"\s+", " ", out).strip()

    tokens = []
    for token in out.split(" "):
        mapped = _NUMBER_MAP.get(token, token)
        mapped = _CONTRACTIONS.get(mapped, mapped)
        if mapped in _ARTICLES:
            continue
        tokens.append(mapped)
    return " ".join(tokens).strip()


def textvqa_soft_accuracy(prediction: str | None, answers: list[str]) -> float:
    """Official-like TextVQA soft score: min(1, count(match)/3)."""
    if not answers:
        return 0.0
    pred = normalize_vqa_answer(prediction)
    matches = sum(1 for ans in answers if normalize_vqa_answer(ans) == pred)
    return float(min(1.0, matches / 3.0))


def evaluate_textvqa_official_from_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Re-evaluate existing prediction rows using TextVQA multi-answer scoring.

    Expects answerable rows to include one of:
    - metadata["official_answers"]: list[str]
    - metadata["answers"]: list[str]
    Falls back to single `ground_truth_answer` when multi-answer annotations are absent.
    """
    if not rows:
        raise ValueError("No prediction rows provided.")

    answerable_rows = [
        row for row in rows if str(row.get("ground_truth_answerability", "")).upper() == "ANSWERABLE"
    ]
    if not answerable_rows:
        raise ValueError("No answerable rows found in predictions.")

    scores: list[float] = []
    missing_multi_answers = 0
    total_reference_answers = 0
    n_with_exactly_ten = 0
    n_truncated_to_ten = 0
    for row in answerable_rows:
        metadata = row.get("metadata") or {}
        answers = metadata.get("official_answers") or metadata.get("answers") or []
        if not isinstance(answers, list):
            answers = []
        answers = [str(x).strip() for x in answers if str(x).strip()]
        if not answers:
            missing_multi_answers += 1
            answers = [str(row.get("ground_truth_answer", ""))]
        if len(answers) == 10:
            n_with_exactly_ten += 1
        if len(answers) > 10:
            n_truncated_to_ten += 1
            answers = answers[:10]
        total_reference_answers += len(answers)
        score = textvqa_soft_accuracy(row.get("parsed_answer"), answers)
        scores.append(score)

    mean_score = float(sum(scores) / len(scores))
    return {
        "official_metric_name": "soft_accuracy",
        "official_score_mean": mean_score,
        "n_answerable": len(answerable_rows),
        "official_soft_score_mean": mean_score,
        "multi_answer_available_rate": float((len(answerable_rows) - missing_multi_answers) / len(answerable_rows)),
        "used_fallback_single_answer_count": int(missing_multi_answers),
        "avg_reference_answer_count": float(total_reference_answers / len(answerable_rows)),
        "exactly_ten_reference_rate": float(n_with_exactly_ten / len(answerable_rows)),
        "truncated_to_first_10_count": int(n_truncated_to_ten),
    }




def evaluate_textvqa_official_metrics_with_standard_schema(
    rows: list[dict[str, Any]],
    n_calibration_bins: int = 10,
) -> dict[str, Any]:
    """Compute TextVQA official-scored metrics using the standard metrics schema."""
    from .metrics import evaluate_predictions

    return evaluate_predictions(rows, n_calibration_bins=n_calibration_bins)
