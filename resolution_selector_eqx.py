import os
import json
import torch
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Any, Optional

class ResolutionSelectorEQX:
    """
    A ComfyUI node for selecting image resolutions with manual input or preset selection.
    Can also resize and center-crop images to the selected resolution.
    Supports loading presets from a text file and switching between manual and preset modes.
    """

    NODE_NAME = "Resolution Selector EQX"
    CATEGORY = "EQX/Utils"

    def __init__(self):
        self.presets_file = os.path.join(os.path.dirname(__file__), "resolutions.txt")
        self.presets = self.load_presets()

    def load_presets(self) -> Dict[str, Tuple[int, int]]:
        """Load resolution presets from the text file."""
        presets = {}

        # Create default file if it doesn't exist
        if not os.path.exists(self.presets_file):
            self.create_default_presets_file()

        try:
            with open(self.presets_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue

                    # Parse format: "name:width,height" or "name:widthxheight"
                    if ':' in line:
                        # Split only on the last colon to handle names with colons
                        last_colon = line.rfind(':')
                        name = line[:last_colon].strip()
                        resolution = line[last_colon + 1:].strip()

                        # Support both comma and 'x' as separators
                        if ',' in resolution:
                            width, height = resolution.split(',')
                        elif 'x' in resolution.lower():
                            width, height = resolution.lower().split('x')
                        else:
                            continue

                        try:
                            width = int(width.strip())
                            height = int(height.strip())
                            presets[name] = (width, height)
                        except ValueError:
                            print(f"[ResolutionSelectorEQX] Invalid resolution format: {line}")
                            continue
        except FileNotFoundError:
            print(f"[ResolutionSelectorEQX] Presets file not found, using defaults")
            self.create_default_presets_file()
            return self.load_presets()
        except Exception as e:
            print(f"[ResolutionSelectorEQX] Error loading presets: {e}")

        # Add a default if no presets loaded
        if not presets:
            presets["HD 1920x1080"] = (1920, 1080)

        return presets

    def create_default_presets_file(self):
        """Create a default resolutions.txt file with common presets."""
        default_content = """# Resolution Presets for Social Media
# Format: name:width,height or name:widthxheight
# Updated for 2024/2025 standards

Instagram (Square post) - 1080x1080 px (1:1):1080x1080
Instagram (Vertical post) - 1080x1350 px (4:5):1080x1350
Instagram (Vertical post - TALL) - 1440x1888 px (3:4):1440x1888
Instagram (Horizontal post) - 1080x566 px (1.91:1):1080x566
Instagram (Stories/Reels) - 1080x1920 px (9:16):1080x1920
TikTok (Video, stories, carousel image) - 1080x1920 px (9:16):1080x1920
"""

        try:
            with open(self.presets_file, 'w', encoding='utf-8') as f:
                f.write(default_content)
            print(f"[ResolutionSelectorEQX] Created default presets file: {self.presets_file}")
        except Exception as e:
            print(f"[ResolutionSelectorEQX] Error creating default presets file: {e}")

    @classmethod
    def INPUT_TYPES(cls):
        # Create instance to load presets
        instance = cls()
        preset_names = list(instance.presets.keys())

        # Ensure at least one preset exists
        if not preset_names:
            preset_names = ["HD 1920x1080"]

        return {
            "required": {
                "mode": (["manual", "preset"], {"default": "preset"}),
                "preset": (preset_names, {"default": preset_names[0]}),
                "manual_width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "display": "number"
                }),
                "manual_height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "display": "number"
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "swap_dimensions": ("BOOLEAN", {"default": False}),
                "resize_mode": (["center_crop", "fit", "stretch"], {"default": "center_crop"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "aspect_ratio", "resolution_name")
    FUNCTION = "select_resolution"
    OUTPUT_NODE = False

    def resize_and_crop_image(self, image_tensor: torch.Tensor, target_width: int, target_height: int, mode: str = "center_crop") -> torch.Tensor:
        """
        Resize and crop an image tensor to the target dimensions.

        Args:
            image_tensor: Input image tensor in ComfyUI format (B, H, W, C)
            target_width: Target width
            target_height: Target height
            mode: Resize mode - "center_crop", "fit", or "stretch"

        Returns:
            Resized image tensor
        """
        batch_size = image_tensor.shape[0]
        results = []

        for i in range(batch_size):
            # Convert from tensor to PIL Image
            # ComfyUI uses (B, H, W, C) format with values 0-1
            img_np = (image_tensor[i].cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np, mode='RGB' if img_np.shape[-1] == 3 else 'RGBA')

            if mode == "center_crop":
                # Calculate scaling factor to fill the target area
                scale_x = target_width / pil_image.width
                scale_y = target_height / pil_image.height
                scale = max(scale_x, scale_y)  # Use max to ensure full coverage

                # Resize image
                new_width = int(pil_image.width * scale)
                new_height = int(pil_image.height * scale)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Center crop to target dimensions
                left = (new_width - target_width) // 2
                top = (new_height - target_height) // 2
                right = left + target_width
                bottom = top + target_height
                pil_image = pil_image.crop((left, top, right, bottom))

            elif mode == "fit":
                # Scale to fit within target dimensions, maintaining aspect ratio
                pil_image.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)

                # Create a new image with target dimensions and paste centered
                new_image = Image.new(pil_image.mode, (target_width, target_height),
                                    (0, 0, 0, 0) if pil_image.mode == 'RGBA' else (0, 0, 0))
                paste_x = (target_width - pil_image.width) // 2
                paste_y = (target_height - pil_image.height) // 2
                new_image.paste(pil_image, (paste_x, paste_y))
                pil_image = new_image

            elif mode == "stretch":
                # Simple resize without maintaining aspect ratio
                pil_image = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

            # Convert back to tensor
            img_np = np.array(pil_image).astype(np.float32) / 255.0
            if len(img_np.shape) == 2:  # Grayscale
                img_np = np.expand_dims(img_np, axis=-1)
            results.append(torch.from_numpy(img_np))

        # Stack all images back into a batch
        return torch.stack(results)

    def select_resolution(self, mode: str, preset: str, manual_width: int, manual_height: int,
                        image: Optional[torch.Tensor] = None, swap_dimensions: bool = False,
                        resize_mode: str = "center_crop") -> Tuple:
        """
        Select resolution based on mode (manual or preset) and optionally resize an image.

        Args:
            mode: Either "manual" or "preset"
            preset: Name of the preset to use (when mode is "preset")
            manual_width: Manual width value (when mode is "manual")
            manual_height: Manual height value (when mode is "manual")
            image: Optional image tensor to resize
            swap_dimensions: Whether to swap width and height
            resize_mode: How to resize the image ("center_crop", "fit", or "stretch")

        Returns:
            Tuple of (image, width, height, aspect_ratio, resolution_name)
        """

        # Reload presets in case the file was updated
        self.presets = self.load_presets()

        # Determine width and height based on mode
        if mode == "manual":
            width = manual_width
            height = manual_height
            resolution_name = f"Manual {width}x{height}"
        else:  # preset mode
            if preset in self.presets:
                width, height = self.presets[preset]
                resolution_name = preset
            else:
                # Fallback to first preset or default
                print(f"[ResolutionSelectorEQX] Preset '{preset}' not found, using default")
                if self.presets:
                    first_preset = list(self.presets.keys())[0]
                    width, height = self.presets[first_preset]
                    resolution_name = first_preset
                else:
                    width, height = 1920, 1080
                    resolution_name = "Default HD 1920x1080"

        # Swap dimensions if requested
        if swap_dimensions:
            width, height = height, width
            resolution_name = f"{resolution_name} (swapped)"

        # Calculate aspect ratio
        aspect_ratio = round(width / height, 4) if height > 0 else 1.0

        # Ensure dimensions are multiples of 8 (common requirement for many AI models)
        width = (width // 8) * 8
        height = (height // 8) * 8

        print(f"[ResolutionSelectorEQX] Selected: {resolution_name} - {width}x{height} (AR: {aspect_ratio})")

        # Process image if provided
        if image is not None:
            # Get original dimensions
            orig_height, orig_width = image.shape[1], image.shape[2]
            print(f"[ResolutionSelectorEQX] Resizing image from {orig_width}x{orig_height} to {width}x{height} using {resize_mode} mode")

            # Resize the image
            output_image = self.resize_and_crop_image(image, width, height, resize_mode)
        else:
            # Create a black dummy image if no input image
            output_image = torch.zeros((1, height, width, 3), dtype=torch.float32)

        return (output_image, width, height, aspect_ratio, resolution_name)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        This method is called to check if the node needs to be re-executed.
        We return a hash of the file modification time so the node updates when the file changes.
        """
        instance = cls()
        if os.path.exists(instance.presets_file):
            return str(os.path.getmtime(instance.presets_file))
        return ""


# Node registration
NODE_CLASS_MAPPINGS = {
    "ResolutionSelectorEQX": ResolutionSelectorEQX
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResolutionSelectorEQX": "Resolution Selector EQX"
}