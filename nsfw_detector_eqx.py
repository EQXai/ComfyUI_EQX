import os
import logging
import tempfile
from typing import List, Tuple, Dict, Union

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    CLIPImageProcessor,
    CLIPConfig,
    CLIPVisionModel,
    PreTrainedModel,
)

from nudenet import NudeDetector
from folder_paths import models_dir


class _ColorFormatter(logging.Formatter):
    RED = "\033[91m"
    RESET = "\033[0m"

    def format(self, record):
        msg = super().format(record)
        if record.levelno >= logging.WARNING:
            msg = f"{self.RED}{msg}{self.RESET}"
        return msg


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(_ColorFormatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class _ClipSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig
    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(
            config.vision_config.hidden_size, config.projection_dim, bias=False
        )

        self.concept_embeds = nn.Parameter(
            torch.empty(17, config.projection_dim), requires_grad=False
        )
        self.special_care_embeds = nn.Parameter(
            torch.empty(3, config.projection_dim), requires_grad=False
        )
        self.concept_embeds_weights = nn.Parameter(torch.empty(17), requires_grad=False)
        self.special_care_embeds_weights = nn.Parameter(torch.empty(3), requires_grad=False)

        self.eval()

    @staticmethod
    def _cosine_similarity(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        src_n = F.normalize(src.unsqueeze(1), dim=-1)
        tgt_n = F.normalize(target, dim=-1)
        return (src_n * tgt_n).sum(-1)

    def forward(self, clip_input: torch.Tensor, *, sensitivity: float) -> torch.Tensor:
        image_features = self.vision_model(clip_input).pooler_output
        image_embeds = self.visual_projection(image_features)

        special_sim = self._cosine_similarity(image_embeds, self.special_care_embeds)
        concept_sim = self._cosine_similarity(image_embeds, self.concept_embeds)

        offset = -0.1 + 0.14 * sensitivity
        special_scores = special_sim - self.special_care_embeds_weights + offset
        concept_scores = concept_sim - self.concept_embeds_weights + offset

        return (special_scores > 0).any(dim=-1) | (concept_scores > 0).any(dim=-1)


THRESHOLDS: Dict[str, float] = {
    "EXPOSED_ANUS": 0.99,
    "BUTTOCKS_EXPOSED": 0.40,
    "EXPOSED_BREAST_F": 0.35,
    "EXPOSED_GENITALIA_F": 0.35,
    "EXPOSED_GENITALIA_M": 0.99,
    "BUTTOCKS_COVERED": 0.40,
    "FEMALE_BREAST_EXPOSED": 0.35,
    "FEMALE_GENITALIA_EXPOSED": 0.33,
    "NOT_DETECTED": 2.0,
    "OTHER": 2.0,
}

SAFE_CLASSES = {"NOT_DETECTED", "OTHER"}


class NSFW_Detector_EQX:
    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("IMAGE", "nsfw")
    FUNCTION = "detect"
    CATEGORY = "EQX"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "sensitivity": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            }
        }

    def __init__(self):
        model_path = os.path.join(models_dir, "safety_checker")

        self.feature_extractor = CLIPImageProcessor.from_pretrained(model_path)
        self.safety_checker: _ClipSafetyChecker = _ClipSafetyChecker.from_pretrained(
            model_path
        ).eval()

        self.nude_detector = NudeDetector()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.safety_checker.to(self.device)

    @staticmethod
    def _to_pil(batch: Union[np.ndarray, torch.Tensor]) -> List[Image.Image]:
        if torch.is_tensor(batch):
            batch = batch.detach().cpu().numpy()

        if batch.ndim == 3:
            batch = batch[None, ...]

        batch = np.clip(batch * 255, 0, 255).astype(np.uint8)
        return [Image.fromarray(img) for img in batch]

    def detect(self, images: Union[np.ndarray, torch.Tensor], sensitivity: float) -> Tuple[Union[np.ndarray, torch.Tensor], bool]:
        pil_images = self._to_pil(images)
        clip_inputs = self.feature_extractor(pil_images, return_tensors="pt").pixel_values.to(
            self.device
        )

        with torch.no_grad():
            nsfw_tensor = self.safety_checker(clip_input=clip_inputs, sensitivity=sensitivity)
        nsfw_flags = nsfw_tensor.cpu().tolist()

        if not any(nsfw_flags):
            return images, False

        for idx, (img, flagged) in enumerate(zip(pil_images, nsfw_flags)):
            if not flagged:
                continue

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img.save(tmp.name)
                detections = self.nude_detector.detect(tmp.name)

            try:
                os.remove(tmp.name)
            except OSError as e:
                logger.warning(f"Temp cleanup failed: {e}")

            for det in detections:
                label = det.get("class") or det.get("label")
                score = det.get("score", 0.0)

                threshold = THRESHOLDS.get(label, THRESHOLDS["OTHER"])
                if label not in SAFE_CLASSES and score >= threshold:
                    logger.warning(
                        "Safety Checker: NudeNet confirmed NSFW "
                        f"(idx={idx}, class={label}, score={score:.2f})"
                    )
                    return images, True

        return images, False


NODE_CLASS_MAPPINGS = {"NSFW Detector EQX": NSFW_Detector_EQX}
NODE_DISPLAY_NAME_MAPPINGS = {"NSFW Detector EQX": "NSFW Detector EQX"}
