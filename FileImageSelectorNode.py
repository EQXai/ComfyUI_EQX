import os
import random
import numpy as np
import torch
from PIL import Image

class FileImageSelector:
    """Selects images from a folder, with modes for random, incremental, or single selection."""

    CATEGORY = "Custom"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Path to the folder containing images
                "folder_path": ("STRING",),
                # Mode of selection: random, incremental, or single_image
                "mode": (["random", "incremental", "single_image"],),
                # For single_image mode, choose a specific index
                "index": ("INT", {"default": 0, "min": 0}),
            },
            "hidden": {
                # Internal fields for ComfyUI
                "unique_id": ("UNIQUE_ID",),
                "extra_pnginfo": ("EXTRA_PNGINFO",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "select_image"
    DEFAULTS = {"mode": "random"}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Only recalc for single_image if the explicit index changes
        if kwargs.get("mode") == "single_image":
            return hash(f"{kwargs['folder_path']}:{kwargs['index']}")
        # For other modes, always refresh
        return float("nan")

    def __init__(self):
        # Counter for incremental mode
        self._counter = 0

    def _gather_images(self, folder_path: str):
        """Collect valid image file paths in the folder."""
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}
        try:
            files = sorted(os.listdir(folder_path))
        except Exception as exc:
            raise RuntimeError(f"Error reading '{folder_path}': {exc}") from exc
        paths = [os.path.join(folder_path, f)
                 for f in files
                 if os.path.splitext(f)[1].lower() in exts]
        if not paths:
            raise ValueError(f"No images found in '{folder_path}'")
        return paths

    def select_image(self, folder_path: str, mode: str, index: int,
                     unique_id=None, extra_pnginfo=None):
        """Main execution: choose an index based on mode, load image, output tensor and filename."""
        paths = self._gather_images(folder_path)
        num_imgs = len(paths)

        if mode == "random":
            # Choose a random index each execution
            idx = random.randint(0, num_imgs - 1)

        elif mode == "incremental":
            # Each run increments the counter by 1
            idx = self._counter % num_imgs
            self._counter += 1

        else:  # single_image
            # Wrap-around single image index
            idx = index % num_imgs

        # Load and convert image to tensor
        img = Image.open(paths[idx]).convert("RGB")
        tensor = (torch.from_numpy(np.array(img, dtype=np.uint8))
                  .float().div(255.0).unsqueeze(0))

        filename_no_ext = os.path.splitext(os.path.basename(paths[idx]))[0]
        # Return index for UI display, along with image tensor and filename
        return {"ui": {"index": [idx]},
                "result": (tensor, filename_no_ext)}

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {"File Image Selector": FileImageSelector}
