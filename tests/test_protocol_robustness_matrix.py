import pytest

from scripts.run_protocol_robustness_matrix import _ensure_positive_temperatures


def test_positive_temperature_list_passes():
    temperatures = _ensure_positive_temperatures([0.2, "0.6", 1])
    assert temperatures == [0.2, 0.6, 1.0]


def test_zero_temperature_rejected():
    with pytest.raises(ValueError, match="must be > 0"):
        _ensure_positive_temperatures([0.2, 0.0])
