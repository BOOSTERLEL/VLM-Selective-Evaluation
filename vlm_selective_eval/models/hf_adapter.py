"""Hugging Face-compatible VLM adapter."""

from __future__ import annotations

import math
from typing import Any

from PIL import Image

from vlm_selective_eval.config import ModelConfig
from .base import BaseVLMAdapter


def normalize_vram_tier(vram_gb: int | float | None) -> int | None:
    """Normalize arbitrary VRAM values into the supported 16GB / 24GB tiers."""
    if vram_gb is None:
        return None
    try:
        numeric = float(vram_gb)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid VRAM size: {vram_gb!r}") from exc
    if numeric <= 0:
        raise ValueError(f"VRAM size must be positive, got {vram_gb!r}.")
    if numeric <= 16:
        return 16
    return 24


def resolve_vram_tier(
    configured_vram_gb: int | float | None,
    *,
    device: str,
    torch_module: Any | None = None,
) -> int | None:
    """Resolve the active VRAM tier from config and the current CUDA device.

    If both a config value and detected GPU memory are available, always use the
    smaller tier so a 24GB config does not force a 16GB device onto the
    high-memory path.
    """
    configured_tier = normalize_vram_tier(configured_vram_gb)
    detected_tier: int | None = None
    if device == "cuda" and torch_module is not None and torch_module.cuda.is_available():
        detected_bytes = torch_module.cuda.get_device_properties(0).total_memory
        detected_gb = detected_bytes / (1024**3)
        detected_tier = normalize_vram_tier(math.ceil(detected_gb))

    if configured_tier is None:
        return detected_tier
    if detected_tier is None:
        return configured_tier
    return min(configured_tier, detected_tier)


def build_runtime_generation_defaults(model_family: str, vram_tier: int | None) -> dict[str, Any]:
    """Return generation overrides for the active VRAM tier."""
    if vram_tier != 16:
        return {}
    defaults: dict[str, Any] = {"use_cache": False}
    if model_family == "internvl":
        defaults["max_num"] = 4
    return defaults


class HuggingFaceVLMAdapter(BaseVLMAdapter):
    """Thin wrapper around Transformers vision-language generation.

    Supports:
    - Qwen/Qwen2.5-VL-7B-Instruct
    - Qwen/Qwen2.5-VL-3B-Instruct
    - llava-hf/llava-onevision-qwen2-0.5b-ov-hf
    - OpenGVLab/InternVL2_5-8B
    """

    def __init__(self, config: ModelConfig) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor, AutoTokenizer
            try:
                from transformers import AutoModelForImageTextToText as HFVLMModel
            except ImportError:
                from transformers import AutoModelForVision2Seq as HFVLMModel
        except Exception as exc:
            raise ImportError(
                "HuggingFace adapter requires `transformers`, `torch`, and for InternVL also `torchvision`."
            ) from exc

        self._torch = torch
        self._AutoModel = AutoModel
        self._AutoProcessor = AutoProcessor
        self._AutoTokenizer = AutoTokenizer
        self._HFVLMModel = HFVLMModel

        self.config = config
        self.model_name = config.model_name
        self.model_family = self._detect_model_family(self.model_name)
        self.max_new_tokens = config.max_new_tokens

        if config.device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.vram_tier = resolve_vram_tier(
            config.vram_gb,
            device=self.device,
            torch_module=torch,
        )
        self.low_vram_mode = self.device == "cuda" and self.vram_tier == 16
        self.dtype = self._pick_dtype()
        self.processor = None
        self.tokenizer = None
        self.default_generation_kwargs = build_runtime_generation_defaults(
            model_family=self.model_family,
            vram_tier=self.vram_tier,
        )

        self.model = self._load_model()
        self.input_device = self._resolve_input_device()

    def _detect_model_family(self, model_name: str) -> str:
        name = model_name.lower()
        if "internvl2_5" in name or "internvl2.5" in name or "opengvlab/internvl" in name:
            return "internvl"
        if "llava-onevision" in name:
            return "llava_onevision"
        if "qwen2.5-vl" in name:
            return "qwen2_5_vl"
        if "qwen2-vl" in name:
            return "qwen2_vl"
        return "generic"

    def _pick_dtype(self):
        if self.device != "cuda":
            return self._torch.float32
        if self._torch.cuda.is_bf16_supported():
            return self._torch.bfloat16
        return self._torch.float16

    def _base_model_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "low_cpu_mem_usage": True,
            "torch_dtype": self.dtype,
        }
        if self.low_vram_mode:
            kwargs["device_map"] = "auto"
        return kwargs

    def _should_move_model_to_device(self, model: Any) -> bool:
        return not bool(getattr(model, "hf_device_map", None))

    def _coerce_device(self, raw_device: Any):
        if isinstance(raw_device, self._torch.device):
            return raw_device
        if isinstance(raw_device, int):
            return self._torch.device(f"cuda:{raw_device}")
        if isinstance(raw_device, str):
            return self._torch.device(raw_device)
        return self._torch.device(self.device)

    def _resolve_input_device(self):
        hf_device_map = getattr(self.model, "hf_device_map", None)
        if hf_device_map:
            for target in hf_device_map.values():
                if target not in {"cpu", "disk"}:
                    return self._coerce_device(target)
            return self._torch.device("cpu")
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            return self._coerce_device(model_device)
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self._torch.device(self.device)

    def _load_model(self):
        if self.model_family == "internvl":
            return self._load_internvl_model()
        return self._load_standard_vlm_model()

    def _load_standard_vlm_model(self):
        self.processor = self._AutoProcessor.from_pretrained(self.model_name)

        # LLaVA-OneVision 官方建议 batch generation 使用 left padding
        if (
            self.model_family == "llava_onevision"
            and hasattr(self.processor, "tokenizer")
            and self.processor.tokenizer is not None
        ):
            self.processor.tokenizer.padding_side = "left"

        model = self._HFVLMModel.from_pretrained(
            self.model_name,
            **self._base_model_kwargs(),
        )
        if self._should_move_model_to_device(model):
            model.to(self.device)
        model.eval()
        return model

    def _load_internvl_model(self):
        self.tokenizer = self._AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=False,
        )

        model = self._AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            **self._base_model_kwargs(),
        )
        if self._should_move_model_to_device(model):
            model.to(self.device)
        model.eval()
        return model

    def _move_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                if self._torch.is_tensor(value) and self._torch.is_floating_point(value):
                    moved[key] = value.to(self.input_device, dtype=self.dtype)
                else:
                    moved[key] = value.to(self.input_device)
            else:
                moved[key] = value
        return moved

    # ---------------------------
    # InternVL image preprocessing
    # ---------------------------
    def _build_internvl_transform(self, input_size: int = 448):
        try:
            import torchvision.transforms as T
            from torchvision.transforms.functional import InterpolationMode
        except Exception as exc:
            raise ImportError(
                "InternVL2_5-8B requires torchvision for image preprocessing."
            ) from exc

        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)

        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )
        return transform

    def _find_closest_aspect_ratio(
        self,
        aspect_ratio: float,
        target_ratios,
        width: int,
        height: int,
        image_size: int,
    ):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height

        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(
        self,
        image: Image.Image,
        min_num: int = 1,
        max_num: int = 12,
        image_size: int = 448,
        use_thumbnail: bool = True,
    ):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []

        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)

        return processed_images

    def _load_internvl_image(
        self,
        image_path: str,
        input_size: int = 448,
        max_num: int = 12,
    ):
        image = Image.open(image_path).convert("RGB")
        transform = self._build_internvl_transform(input_size=input_size)
        images = self._dynamic_preprocess(
            image,
            image_size=input_size,
            use_thumbnail=True,
            max_num=max_num,
        )
        pixel_values = [transform(img) for img in images]
        pixel_values = self._torch.stack(pixel_values)
        pixel_values = pixel_values.to(self.input_device, dtype=self.dtype)
        return pixel_values

    # ---------------------------
    # Public generation API
    # ---------------------------
    def generate(self, image_path: str, prompt: str, **kwargs):
        if self.model_family == "internvl":
            return self._generate_internvl(image_path=image_path, prompt=prompt, **kwargs)
        return self._generate_standard(image_path=image_path, prompt=prompt, **kwargs)

    def _generate_standard(self, image_path: str, prompt: str, **kwargs):
        generation_kwargs = {"max_new_tokens": self.max_new_tokens}
        generation_kwargs.update(self.default_generation_kwargs)
        generation_kwargs.update(kwargs)

        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )
        moved = self._move_inputs(inputs)

        with self._torch.no_grad():
            output_ids = self.model.generate(**moved, **generation_kwargs)

        if "input_ids" in moved:
            generated_ids = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(moved["input_ids"], output_ids)
            ]
        else:
            generated_ids = output_ids

        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return text.strip()

    def _generate_internvl(self, image_path: str, prompt: str, **kwargs):
        runtime_kwargs = dict(self.default_generation_kwargs)
        runtime_kwargs.update(kwargs)
        max_num = int(runtime_kwargs.pop("max_num", 12))
        generation_config = {"max_new_tokens": self.max_new_tokens}
        generation_config.update(runtime_kwargs)

        pixel_values = self._load_internvl_image(image_path=image_path, max_num=max_num)
        question = f"<image>\n{prompt}"

        with self._torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
            )

        return response.strip()
