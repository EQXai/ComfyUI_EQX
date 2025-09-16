import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import math

class AspectRatioCropEQX:
    """
    A ComfyUI node for resizing and cropping images based on aspect ratio and maximum resolution.
    Maintains the specified aspect ratio while respecting maximum resolution limits.
    """

    NODE_NAME = "Aspect Ratio Crop EQX"
    CATEGORY = "EQX/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "aspect_ratio": ("STRING", {
                    "default": "16:9",
                    "multiline": False,
                    "display": "text"
                }),
                "max_resolution": ("INT", {
                    "default": 1920,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "display": "number"
                }),
                "limit_mode": (["width", "height", "longest_side", "shortest_side", "total_pixels"], {
                    "default": "longest_side"
                }),
                "crop_position": (["center", "top", "bottom", "left", "right", "top_left", "top_right", "bottom_left", "bottom_right"], {
                    "default": "center"
                }),
                "resize_quality": (["lanczos", "bilinear", "bicubic", "nearest"], {
                    "default": "lanczos"
                }),
                "ensure_multiple_of": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("image", "width", "height", "scale_factor", "crop_percentage")
    FUNCTION = "process_image"
    OUTPUT_NODE = False

    def parse_aspect_ratio(self, aspect_ratio_str: str) -> float:
        """
        Parse aspect ratio from string format (e.g., "16:9", "1.778", "16/9")
        Returns the aspect ratio as a float (width/height)
        """
        aspect_ratio_str = aspect_ratio_str.strip()

        # Try to parse as a direct float
        try:
            return float(aspect_ratio_str)
        except ValueError:
            pass

        # Try to parse as ratio format (16:9 or 16/9)
        for separator in [':', '/']:
            if separator in aspect_ratio_str:
                try:
                    parts = aspect_ratio_str.split(separator)
                    if len(parts) == 2:
                        width = float(parts[0].strip())
                        height = float(parts[1].strip())
                        if height > 0:
                            return width / height
                except (ValueError, ZeroDivisionError):
                    pass

        # Default to 16:9 if parsing fails
        print(f"[AspectRatioCropEQX] Could not parse aspect ratio '{aspect_ratio_str}', using default 16:9")
        return 16.0 / 9.0

    def calculate_target_dimensions(self,
                                  original_width: int,
                                  original_height: int,
                                  target_aspect_ratio: float,
                                  max_resolution: int,
                                  limit_mode: str,
                                  multiple_of: int) -> Tuple[int, int]:
        """
        Calculate the target dimensions based on aspect ratio and resolution limits
        """
        # Start with dimensions that match the target aspect ratio
        if target_aspect_ratio >= 1.0:  # Landscape or square
            target_width = max_resolution
            target_height = int(max_resolution / target_aspect_ratio)
        else:  # Portrait
            target_height = max_resolution
            target_width = int(max_resolution * target_aspect_ratio)

        # Apply limit mode constraints
        if limit_mode == "width":
            if target_width > max_resolution:
                target_width = max_resolution
                target_height = int(target_width / target_aspect_ratio)
        elif limit_mode == "height":
            if target_height > max_resolution:
                target_height = max_resolution
                target_width = int(target_height * target_aspect_ratio)
        elif limit_mode == "longest_side":
            longest = max(target_width, target_height)
            if longest > max_resolution:
                scale = max_resolution / longest
                target_width = int(target_width * scale)
                target_height = int(target_height * scale)
        elif limit_mode == "shortest_side":
            shortest = min(target_width, target_height)
            if shortest > max_resolution:
                scale = max_resolution / shortest
                target_width = int(target_width * scale)
                target_height = int(target_height * scale)
        elif limit_mode == "total_pixels":
            # Max resolution represents maximum total pixels (in thousands)
            max_pixels = max_resolution * 1000
            current_pixels = target_width * target_height
            if current_pixels > max_pixels:
                scale = math.sqrt(max_pixels / current_pixels)
                target_width = int(target_width * scale)
                target_height = int(target_height * scale)

        # Ensure dimensions are multiples of the specified value
        target_width = (target_width // multiple_of) * multiple_of
        target_height = (target_height // multiple_of) * multiple_of

        # Ensure minimum size
        target_width = max(target_width, multiple_of)
        target_height = max(target_height, multiple_of)

        return target_width, target_height

    def get_crop_coordinates(self,
                           img_width: int,
                           img_height: int,
                           crop_width: int,
                           crop_height: int,
                           position: str) -> Tuple[int, int, int, int]:
        """
        Calculate crop coordinates based on position
        Returns (left, top, right, bottom)
        """
        # Calculate offsets
        x_offset = img_width - crop_width
        y_offset = img_height - crop_height

        if position == "center":
            left = x_offset // 2
            top = y_offset // 2
        elif position == "top":
            left = x_offset // 2
            top = 0
        elif position == "bottom":
            left = x_offset // 2
            top = y_offset
        elif position == "left":
            left = 0
            top = y_offset // 2
        elif position == "right":
            left = x_offset
            top = y_offset // 2
        elif position == "top_left":
            left = 0
            top = 0
        elif position == "top_right":
            left = x_offset
            top = 0
        elif position == "bottom_left":
            left = 0
            top = y_offset
        elif position == "bottom_right":
            left = x_offset
            top = y_offset
        else:  # Default to center
            left = x_offset // 2
            top = y_offset // 2

        right = left + crop_width
        bottom = top + crop_height

        return left, top, right, bottom

    def get_resize_filter(self, quality: str):
        """Get PIL resize filter based on quality setting"""
        filters = {
            "lanczos": Image.Resampling.LANCZOS,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "nearest": Image.Resampling.NEAREST
        }
        return filters.get(quality, Image.Resampling.LANCZOS)

    def process_single_image(self,
                           image_tensor: torch.Tensor,
                           target_width: int,
                           target_height: int,
                           crop_position: str,
                           resize_quality: str) -> Tuple[torch.Tensor, float, float]:
        """
        Process a single image from the batch
        Returns (processed_image, scale_factor, crop_percentage)
        """
        # Convert tensor to PIL Image
        img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np, mode='RGB' if img_np.shape[-1] == 3 else 'RGBA')

        original_width = pil_image.width
        original_height = pil_image.height
        original_area = original_width * original_height

        # Calculate target aspect ratio
        target_aspect = target_width / target_height
        current_aspect = original_width / original_height

        # Determine scaling strategy
        if abs(current_aspect - target_aspect) < 0.01:
            # Aspects are very close, just resize
            scale_factor = min(target_width / original_width, target_height / original_height)
            new_width = target_width
            new_height = target_height
            pil_image = pil_image.resize((new_width, new_height), self.get_resize_filter(resize_quality))
            crop_percentage = 0.0
        else:
            # Need to resize and crop
            # Scale to ensure we can crop to target aspect ratio
            if current_aspect > target_aspect:
                # Image is wider than target - scale by height
                scale_factor = target_height / original_height
                new_height = target_height
                new_width = int(original_width * scale_factor)
            else:
                # Image is taller than target - scale by width
                scale_factor = target_width / original_width
                new_width = target_width
                new_height = int(original_height * scale_factor)

            # Resize image
            pil_image = pil_image.resize((new_width, new_height), self.get_resize_filter(resize_quality))

            # Calculate crop
            if new_width > target_width or new_height > target_height:
                left, top, right, bottom = self.get_crop_coordinates(
                    new_width, new_height, target_width, target_height, crop_position
                )

                # Calculate crop percentage
                cropped_area = (new_width * new_height) - (target_width * target_height)
                crop_percentage = (cropped_area / (new_width * new_height)) * 100

                # Perform crop
                pil_image = pil_image.crop((left, top, right, bottom))
            else:
                crop_percentage = 0.0

        # Convert back to tensor
        img_np = np.array(pil_image).astype(np.float32) / 255.0
        if len(img_np.shape) == 2:
            img_np = np.expand_dims(img_np, axis=-1)

        return torch.from_numpy(img_np), scale_factor, crop_percentage

    def process_image(self,
                     image: torch.Tensor,
                     aspect_ratio: str,
                     max_resolution: int,
                     limit_mode: str,
                     crop_position: str,
                     resize_quality: str,
                     ensure_multiple_of: int) -> Tuple:
        """
        Main processing function
        """
        batch_size = image.shape[0]
        original_height, original_width = image.shape[1], image.shape[2]

        # Parse aspect ratio
        target_aspect_ratio = self.parse_aspect_ratio(aspect_ratio)

        # Calculate target dimensions
        target_width, target_height = self.calculate_target_dimensions(
            original_width, original_height, target_aspect_ratio,
            max_resolution, limit_mode, ensure_multiple_of
        )

        print(f"[AspectRatioCropEQX] Processing {batch_size} image(s)")
        print(f"[AspectRatioCropEQX] Original: {original_width}x{original_height}")
        print(f"[AspectRatioCropEQX] Target: {target_width}x{target_height} (AR: {target_aspect_ratio:.3f})")
        print(f"[AspectRatioCropEQX] Limit mode: {limit_mode}, Max resolution: {max_resolution}")

        # Process each image in the batch
        processed_images = []
        scale_factors = []
        crop_percentages = []

        for i in range(batch_size):
            processed_img, scale_factor, crop_pct = self.process_single_image(
                image[i], target_width, target_height, crop_position, resize_quality
            )
            processed_images.append(processed_img)
            scale_factors.append(scale_factor)
            crop_percentages.append(crop_pct)

        # Stack processed images
        output_image = torch.stack(processed_images)

        # Calculate average metrics
        avg_scale_factor = sum(scale_factors) / len(scale_factors)
        avg_crop_percentage = sum(crop_percentages) / len(crop_percentages)

        print(f"[AspectRatioCropEQX] Average scale factor: {avg_scale_factor:.3f}")
        print(f"[AspectRatioCropEQX] Average crop: {avg_crop_percentage:.1f}%")

        return (output_image, target_width, target_height, avg_scale_factor, avg_crop_percentage)


# Node registration
NODE_CLASS_MAPPINGS = {
    "AspectRatioCropEQX": AspectRatioCropEQX
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AspectRatioCropEQX": "Aspect Ratio Crop EQX"
}