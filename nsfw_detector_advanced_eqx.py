import os
import shutil
import numpy as np
import torch
from PIL import Image
import tempfile
import time
from nudenet import NudeDetector

ALL_CATEGORIES = [
    'EXPOSED_ANUS', 'BUTTOCKS_EXPOSED', 'EXPOSED_BREAST_F', 'EXPOSED_GENITALIA_F',
    'BELLY_EXPOSED', 'FEMALE_BREAST_COVERED',
    'FEMALE_GENITALIA_COVERED', 'BUTTOCKS_COVERED', 'COVERED_FEET',
    'EXPOSED_FEET', 'ARMPITS_EXPOSED', 'FACE_FEMALE',
    'FEMALE_BREAST_EXPOSED', 'FEMALE_GENITALIA_EXPOSED', 'OTHER', 'NOT_DETECTED'
]

CATEGORY_DISPLAY_MAPPINGS = {
    'EXPOSED_ANUS': 'Exposed Anus',
    'BUTTOCKS_EXPOSED': 'Buttocks Exposed',
    'EXPOSED_BREAST_F': 'Exposed Breast Female',
    'EXPOSED_GENITALIA_F': 'Exposed Genitalia Female',
    'BELLY_EXPOSED': 'Belly Exposed',
    'FEMALE_BREAST_COVERED': 'Female Breast Covered',
    'FEMALE_GENITALIA_COVERED': 'Female Genitalia Covered',
    'BUTTOCKS_COVERED': 'Buttocks Covered',
    'COVERED_FEET': 'Covered Feet',
    'EXPOSED_FEET': 'Exposed Feet',
    'ARMPITS_EXPOSED': 'Armpits Exposed',
    'FACE_FEMALE': 'Face',
    'FEMALE_BREAST_EXPOSED': 'Female Breast Exposed',
    'FEMALE_GENITALIA_EXPOSED': 'Female Genitalia Exposed',
    'OTHER': 'Other',
}

DEFAULT_THRESHOLDS = {
    'EXPOSED_ANUS': 0.99, 'BUTTOCKS_EXPOSED': 0.40, 'EXPOSED_BREAST_F': 0.35,
    'EXPOSED_GENITALIA_F': 0.35, 'BELLY_EXPOSED': 0.99,
    'FEMALE_BREAST_COVERED': 0.99, 'FEMALE_GENITALIA_COVERED': 0.99,
    'BUTTOCKS_COVERED': 0.40, 'COVERED_FEET': 0.01,
    'EXPOSED_FEET': 0.99, 'ARMPITS_EXPOSED': 0.99, 'FACE_FEMALE': 0.6,
    'FEMALE_BREAST_EXPOSED': 0.35, 'FEMALE_GENITALIA_EXPOSED': 0.33, 'OTHER': 0.6,
    'NOT_DETECTED': 1.0
}

DEFAULT_ALLOWED_CATEGORIES = {
    "EXPOSED_GENITALIA_F", "EXPOSED_BREAST_F", "EXPOSED_ANUS",
    "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED",
    "BUTTOCKS_COVERED", "BUTTOCKS_EXPOSED", "NOT_DETECTED", "OTHER"
}

def create_dummy_image():
    return torch.zeros((1, 1, 1, 3), dtype=torch.float32)

class NSFWDetectorAdvancedEQX:
    _INTERNAL_TO_WIDGET_NAME = {
        internal: display.replace(' ', '_')
        for internal, display in CATEGORY_DISPLAY_MAPPINGS.items()
    }
    _WIDGET_TO_INTERNAL_NAME = {v: k for k, v in _INTERNAL_TO_WIDGET_NAME.items()}

    def __init__(self):
        self.detector = NudeDetector()
        self.image_index = 0
        self.seed = 0

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "directory_path": ("STRING", {"default": ""}),
                "reset_iterator": ("BOOLEAN", {"default": False}),
                "enable_action": ("BOOLEAN", {"default": False}),
                "target_directory": ("STRING", {"default": "C:\\output\\classified"}),
            }
        }
        for internal_name, widget_name in cls._INTERNAL_TO_WIDGET_NAME.items():
            inputs["optional"][widget_name] = ("BOOLEAN", {"default": internal_name in DEFAULT_ALLOWED_CATEGORIES})
        return inputs

    RETURN_TYPES = ("IMAGE", "STRING", "BOOLEAN", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("image", "best_category", "is_nsfw", "filename", "summary", "image_index", "current_seed")
    FUNCTION = "execute"
    CATEGORY = "EQX"

    def execute(self, seed, image=None, directory_path="", reset_iterator=False, enable_action=False, target_directory="", **kwargs):
        if reset_iterator:
            self.image_index = 0
            self.seed = 0

        current_index = self.image_index
        current_seed_val = self.seed
        ui_info = {"text": [f"Index: {current_index}", f"Seed: {current_seed_val}"]}
        
        allowed_categories = self._get_allowed_categories(kwargs)

        if image is not None:
            img_tensor, best_category, is_nsfw, filename, summary = self.process_tensor_image(image, enable_action, target_directory, allowed_categories)
            result = (img_tensor, best_category, is_nsfw, filename, summary, current_index, current_seed_val)
            return {"result": result, "ui": ui_info}
        
        if directory_path.strip():
            img_tensor, best_category, is_nsfw, filename, summary = self.process_directory_image(directory_path, current_index, enable_action, target_directory, allowed_categories)
            self.image_index += 1
            self.seed += 1
            result = (img_tensor, best_category, is_nsfw, filename, summary, current_index, current_seed_val)
            return {"result": result, "ui": ui_info}
        
        result = (create_dummy_image(), "NOT_DETECTED", False, "", "Error: No image or directory provided.", current_index, current_seed_val)
        return {"result": result, "ui": ui_info}

    def _get_allowed_categories(self, kwargs):
        allowed = set()
        for widget_name, internal_name in self._WIDGET_TO_INTERNAL_NAME.items():
            is_default_enabled = internal_name in DEFAULT_ALLOWED_CATEGORIES
            if kwargs.get(widget_name, is_default_enabled):
                allowed.add(internal_name)
        
        if 'NOT_DETECTED' in DEFAULT_ALLOWED_CATEGORIES:
             allowed.add('NOT_DETECTED')

        return allowed

    def _perform_detection(self, image_path, allowed_categories):
        best_category = "NOT_DETECTED"
        summary = ""
        max_score = -1
        try:
            results = self.detector.detect(image_path)
            for detection in results:
                label = detection.get('class') or detection.get('label')
                if label not in allowed_categories: continue
                score = detection.get('score', 0)
                if score > max_score and score >= DEFAULT_THRESHOLDS.get(label, 0.99):
                    max_score = score
                    best_category = label
            summary = f"Best category: {best_category} (Score: {max_score:.2f} if detected)"
        except Exception as e:
            summary = f"Error during detection: {e}"
        return best_category, summary

    def process_tensor_image(self, image_tensor, enable_saving, save_directory, allowed_categories):
        img_pil = Image.fromarray((image_tensor[0].cpu().numpy() * 255).astype(np.uint8)).convert("RGB")
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img_pil.save(tmp.name)
            tmp_path = tmp.name

        best_category, summary = self._perform_detection(tmp_path, allowed_categories)
        os.remove(tmp_path)
        
        filename = ""
        if enable_saving:
            if not os.path.isdir(save_directory):
                summary += f"\nWarning: Save directory '{save_directory}' is invalid."
            else:
                try:
                    target_dir = os.path.join(save_directory, best_category)
                    os.makedirs(target_dir, exist_ok=True)
                    filename = f"img_{int(time.time() * 1000)}.png"
                    destination_path = os.path.join(target_dir, filename)
                    img_pil.save(destination_path)
                    summary += f"\nSaved to: {destination_path}"
                except Exception as e:
                    summary += f"\nError saving file: {e}"

        is_nsfw = best_category != "NOT_DETECTED"
        return (image_tensor, best_category, is_nsfw, filename, summary)

    def process_directory_image(self, directory_path, image_index, enable_move, move_directory, allowed_categories):
        if not os.path.isdir(directory_path):
            return (create_dummy_image(), "NOT_DETECTED", False, "", f"Error: Directory not found: {directory_path}")

        image_files = sorted([f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])
        if not image_files:
            return (create_dummy_image(), "NOT_DETECTED", False, "", "Error: No images found in directory.")

        filename = image_files[image_index % len(image_files)]
        image_path = os.path.join(directory_path, filename)
        
        try:
            img_pil = Image.open(image_path).convert("RGB")
            img_tensor = torch.from_numpy(np.array(img_pil, dtype=np.float32) / 255.0).unsqueeze(0)
        except Exception as e:
            return (create_dummy_image(), "NOT_DETECTED", False, filename, f"Error loading image: {e}")

        best_category, summary = self._perform_detection(image_path, allowed_categories)
        summary = f"File: {filename}\n{summary}"

        if enable_move:
            if not os.path.isdir(move_directory):
                summary += f"\nWarning: Move directory '{move_directory}' is invalid."
            else:
                try:
                    target_dir = os.path.join(move_directory, best_category)
                    os.makedirs(target_dir, exist_ok=True)
                    destination_path = os.path.join(target_dir, filename)
                    shutil.move(image_path, destination_path)
                    summary += f"\nMoved to: {destination_path}"
                except Exception as e:
                    summary += f"\nError moving file: {e}"
        
        is_nsfw = best_category != "NOT_DETECTED"
        return (img_tensor, best_category, is_nsfw, filename, summary)