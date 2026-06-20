import pytest

from scripts.run_experiment_matrix import _build_run_config as build_experiment_run_config
from scripts.run_protocol_robustness_matrix import _build_run_config as build_protocol_run_config
from vlm_selective_eval.models.hf_adapter import (
    build_runtime_generation_defaults,
    normalize_vram_tier,
    resolve_vram_tier,
)


class _FakeCuda:
    def __init__(self, total_gb: float) -> None:
        self._total_memory = int(total_gb * (1024**3))

    def is_available(self) -> bool:
        return True

    def get_device_properties(self, index: int):
        assert index == 0
        return type("Props", (), {"total_memory": self._total_memory})()


class _FakeTorch:
    def __init__(self, total_gb: float) -> None:
        self.cuda = _FakeCuda(total_gb=total_gb)


def test_normalize_vram_tier_buckets_sizes():
    assert normalize_vram_tier(None) is None
    assert normalize_vram_tier(12) == 16
    assert normalize_vram_tier(16) == 16
    assert normalize_vram_tier(20) == 24
    assert normalize_vram_tier(24) == 24


def test_normalize_vram_tier_rejects_non_positive_values():
    with pytest.raises(ValueError, match="must be positive"):
        normalize_vram_tier(0)


def test_resolve_vram_tier_uses_detected_gpu_memory_when_config_missing():
    assert resolve_vram_tier(None, device="cuda", torch_module=_FakeTorch(15.2)) == 16
    assert resolve_vram_tier(None, device="cuda", torch_module=_FakeTorch(23.1)) == 24


def test_resolve_vram_tier_prefers_safer_detected_tier_over_larger_config():
    assert resolve_vram_tier(24, device="cuda", torch_module=_FakeTorch(14.6)) == 16
    assert resolve_vram_tier(16, device="cuda", torch_module=_FakeTorch(23.1)) == 16


def test_build_runtime_generation_defaults_for_16gb_devices():
    assert build_runtime_generation_defaults("qwen2_5_vl", 16) == {"use_cache": False}
    assert build_runtime_generation_defaults("internvl", 16) == {
        "use_cache": False,
        "max_num": 4,
    }
    assert build_runtime_generation_defaults("internvl", 24) == {}


def test_experiment_matrix_run_config_keeps_vram_gb(tmp_path):
    config = build_experiment_run_config(
        run_id="demo",
        run_output_dir=tmp_path,
        seed=7,
        n_bins=10,
        dataset_cfg={"mode": "synthetic"},
        model_cfg={
            "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
            "vram_gb": 16,
            "generation_kwargs": {"temperature": 0.2},
        },
        prompt_mode="structured",
        prompt_system_text="system",
    )
    assert config["model"]["vram_gb"] == 16


def test_protocol_matrix_run_config_keeps_vram_gb(tmp_path):
    config = build_protocol_run_config(
        run_id="demo",
        run_output_dir=tmp_path,
        seed=7,
        n_bins=10,
        dataset_cfg={"mode": "textvqa"},
        model_cfg={
            "model_name": "OpenGVLab/InternVL2_5-8B",
            "vram_gb": 24,
            "generation_kwargs": {},
        },
        prompt_cfg={"name": "structured", "mode": "structured"},
        default_system_text="system",
        temperature=0.2,
    )
    assert config["model"]["vram_gb"] == 24
