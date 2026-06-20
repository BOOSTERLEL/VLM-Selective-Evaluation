from vlm_selective_eval.evaluation.normalize import normalize_answer


def test_answer_normalization_basic():
    assert normalize_answer(" The, Cat! ") == "cat"
    assert normalize_answer("A BLUE car") == "blue car"
    assert normalize_answer("") == ""
