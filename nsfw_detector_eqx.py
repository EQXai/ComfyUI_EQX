import os
import logging
import tempfile
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPConfig, CLIPVisionModel, PreTrainedModel
from nudenet import NudeDetector
from folder_paths import models_dir

# ANSI codes for colored warnings
RED = "\033[91m"
RESET = "\033[0m"

class CustomFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno >= logging.WARNING:
            return f"{RED}{super().format(record)}{RESET}"
        return super().format(record)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
for handler in logger.handlers:
    handler.setFormatter(CustomFormatter())

class ClipSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig
    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)
        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)
        self.concept_embeds = nn.Parameter(torch.ones(17, config.projection_dim), requires_grad=False)
        self.special_care_embeds = nn.Parameter(torch.ones(3, config.projection_dim), requires_grad=False)
        self.concept_embeds_weights = nn.Parameter(torch.ones(17), requires_grad=False)
        self.special_care_embeds_weights = nn.Parameter(torch.ones(3), requires_grad=False)

    @staticmethod
    def compute_cosine_similarity(embeds, target_embeds):
        if len(embeds.shape) == 1:
            embeds = embeds.unsqueeze(0)
        if len(target_embeds.shape) == 1:
            target_embeds = target_embeds.unsqueeze(0)
        if embeds.dim() == 2 and target_embeds.dim() == 2:
            embeds = embeds.unsqueeze(1)
        return F.cosine_similarity(embeds, target_embeds, dim=-1)

    @staticmethod
    def process_image_scores_batch(special_scores, concept_scores):
        batch_results = []
        for i in range(special_scores.shape[0]):
            result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []}
            special_attention_indices = torch.where(special_scores[i] > 0)[0]
            concept_attention_indices = torch.where(concept_scores[i] > 0)[0]
            for idx in special_attention_indices:
                score = round(special_scores[i][idx].item(), 3)
                result_img["special_scores"][idx.item()] = score
                result_img["special_care"].append((idx.item(), score))
            for idx in concept_attention_indices:
                score = round(concept_scores[i][idx].item(), 3)
                result_img["concept_scores"][idx.item()] = score
                result_img["bad_concepts"].append((idx.item(), score))
            batch_results.append(result_img)
        return batch_results

    def forward(self, clip_input, images, sensitivity):
        with torch.no_grad():
            image_batch = self.vision_model(clip_input)[1]
            image_embeds = self.visual_projection(image_batch)
            special_cos_dist = self.compute_cosine_similarity(image_embeds, self.special_care_embeds)
            cos_dist = self.compute_cosine_similarity(image_embeds, self.concept_embeds)
            adjusted_sensitivity = calculate_sensitivity(sensitivity)
            special_scores = special_cos_dist - self.special_care_embeds_weights.unsqueeze(0) + adjusted_sensitivity
            concept_scores = cos_dist - self.concept_embeds_weights.unsqueeze(0) + adjusted_sensitivity
            results = self.process_image_scores_batch(special_scores, concept_scores)
            nsfw_concepts = [len(res["bad_concepts"]) > 0 for res in results]
            if any(nsfw_concepts):
                logging.warning("Safety Checker: NSFW content detected.")
            return images, nsfw_concepts[0] if len(nsfw_concepts) == 1 else nsfw_concepts

def calculate_sensitivity(input_sensitivity):
    return -0.1 + 0.14 * input_sensitivity

# Threshold configuration for NudeDetector verification
thresholds = {
    'EXPOSED_ANUS': 0.99,
    'BUTTOCKS_EXPOSED': 0.40,
    'EXPOSED_BREAST_F': 0.35,
    'EXPOSED_GENITALIA_F': 0.35,
    'EXPOSED_GENITALIA_M': 0.99,
    'BELLY_EXPOSED': 0.99,
    'FEMALE_BREAST_COVERED': 0.99,
    'MALE_BREAST_EXPOSED': 0.99,
    'FEMALE_GENITALIA_COVERED': 0.99,
    'MALE_GENITALIA_COVERED': 0.99,
    'BUTTOCKS_COVERED': 0.40,
    'COVERED_FEET': 0.01,
    'EXPOSED_FEET': 0.99,
    'ARMPITS_EXPOSED': 0.99,
    'FACE_FEMALE': 0.6,
    'FACE_MALE': 0.99,
    'FEMALE_BREAST_EXPOSED': 0.35,
    'FEMALE_GENITALIA_EXPOSED': 0.33,
    'OTHER': 0.6
}

allowed_categories = {
    "EXPOSED_GENITALIA_F", "EXPOSED_BREAST_F", "EXPOSED_ANUS",
    "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED",
    "BUTTOCKS_COVERED", "BUTTOCKS_EXPOSED", "NOT_DETECTED", "OTHER"
}

class NSFW_Detector_EQX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "sensitivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.10}),
            }
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("IMAGE", "nsfw")
    FUNCTION = "detect"
    CATEGORY = "EQX"

    def __init__(self):
        safety_checker_model = os.path.join(models_dir, "safety_checker")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(safety_checker_model)
        self.safety_checker = ClipSafetyChecker.from_pretrained(safety_checker_model)
        self.nude_detector = NudeDetector()

    def numpy_to_pil(self, images):
        if torch.is_tensor(images):
            images = images.cpu().numpy()
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        return [Image.fromarray(image) for image in images]

    def detect(self, images, sensitivity):
        pil_images = self.numpy_to_pil(images)
        safety_input = self.feature_extractor(pil_images, return_tensors="pt")
        images_for_check = images.clone() if torch.is_tensor(images) else np.copy(images)
        _, nsfw_clip = self.safety_checker(images=images_for_check, clip_input=safety_input.pixel_values, sensitivity=sensitivity)
        nsfw_flag = bool(nsfw_clip) if isinstance(nsfw_clip, bool) else any(nsfw_clip)

        for img in pil_images:
            fd, tmp_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            try:
                img.save(tmp_path)
                try:
                    detections = self.nude_detector.detect(tmp_path)
                except Exception as e:
                    logger.error(f"NudeDetector error: {e}")
                    detections = []
                for detection in detections:
                    label = detection.get('class') or detection.get('label')
                    score = detection.get('score', 0)
                    if label in allowed_categories:
                        threshold = thresholds.get(label, thresholds['OTHER'])
                        if score >= threshold:
                            nsfw_flag = True
                            break
            finally:
                try:
                    os.remove(tmp_path)
                except OSError as e:
                    logger.error(f"Temp file cleanup error: {e}")
            if nsfw_flag:
                break
        return images, nsfw_flag

NODE_CLASS_MAPPINGS = {"NSFW Detector EQX": NSFW_Detector_EQX}
NODE_DISPLAY_NAME_MAPPINGS = {"NSFW Detector EQX": "NSFW Detector EQX"}
