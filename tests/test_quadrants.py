from vlm_selective_eval.evaluation.quadrants import map_quadrant
from vlm_selective_eval.constants import (
    ANSWERABLE,
    QUADRANT_ANSWERABLE_CORRECT,
    QUADRANT_ANSWERABLE_WRONG,
    QUADRANT_UNANSWERABLE_ABSTAIN,
    QUADRANT_UNANSWERABLE_ASSERT,
    UNANSWERABLE,
)


def test_quadrant_mapping_all_cases():
    assert (
        map_quadrant(
            gt_answerability=ANSWERABLE,
            gt_answer="cat",
            pred_status=ANSWERABLE,
            pred_answer="cat",
        )
        == QUADRANT_ANSWERABLE_CORRECT
    )
    assert (
        map_quadrant(
            gt_answerability=ANSWERABLE,
            gt_answer="cat",
            pred_status=ANSWERABLE,
            pred_answer="dog",
        )
        == QUADRANT_ANSWERABLE_WRONG
    )
    assert (
        map_quadrant(
            gt_answerability=UNANSWERABLE,
            gt_answer="",
            pred_status=UNANSWERABLE,
            pred_answer="",
        )
        == QUADRANT_UNANSWERABLE_ABSTAIN
    )
    assert (
        map_quadrant(
            gt_answerability=UNANSWERABLE,
            gt_answer="",
            pred_status=ANSWERABLE,
            pred_answer="anything",
        )
        == QUADRANT_UNANSWERABLE_ASSERT
    )
