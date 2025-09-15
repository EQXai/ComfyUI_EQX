import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
import tempfile
from typing import Tuple, List, Dict, Optional
import json

try:
    from nudenet import NudeDetector
    NUDENET_AVAILABLE = True
    print("[BodyCropMaskEQX] NudeNet import successful")
except ImportError as e:
    print(f"[BodyCropMaskEQX] ERROR: nudenet not found. Install with: pip install nudenet")
    print(f"[BodyCropMaskEQX] Import error details: {e}")
    NUDENET_AVAILABLE = False


# Body part categories from NudeNet (as they appear in detection results)
# Note: NudeNet returns these in uppercase
BODY_CATEGORIES = [
    'FACE_FEMALE', 'FACE_MALE',
    'ARMPITS_EXPOSED', 'BELLY_EXPOSED', 'BELLY_COVERED',
    'BUTTOCKS_EXPOSED', 'BUTTOCKS_COVERED',
    'FEMALE_BREAST_EXPOSED', 'FEMALE_BREAST_COVERED',
    'FEMALE_GENITALIA_EXPOSED', 'FEMALE_GENITALIA_COVERED',
    'MALE_GENITALIA_EXPOSED', 'MALE_BREAST_EXPOSED',
    'ANUS_EXPOSED', 'ANUS_COVERED',
    'FEET_EXPOSED', 'FEET_COVERED',
    'ARMPITS_COVERED'
]

CATEGORY_DISPLAY_NAMES = {
    'FACE_FEMALE': 'Face (Female)',
    'FACE_MALE': 'Face (Male)',
    'ARMPITS_EXPOSED': 'Armpits (Exposed)',
    'ARMPITS_COVERED': 'Armpits (Covered)',
    'BELLY_EXPOSED': 'Belly (Exposed)',
    'BELLY_COVERED': 'Belly (Covered)',
    'BUTTOCKS_EXPOSED': 'Buttocks (Exposed)',
    'BUTTOCKS_COVERED': 'Buttocks (Covered)',
    'FEMALE_BREAST_EXPOSED': 'Female Breast (Exposed)',
    'FEMALE_BREAST_COVERED': 'Female Breast (Covered)',
    'MALE_BREAST_EXPOSED': 'Male Breast (Exposed)',
    'FEMALE_GENITALIA_EXPOSED': 'Female Genitalia (Exposed)',
    'FEMALE_GENITALIA_COVERED': 'Female Genitalia (Covered)',
    'MALE_GENITALIA_EXPOSED': 'Male Genitalia (Exposed)',
    'ANUS_EXPOSED': 'Anus (Exposed)',
    'ANUS_COVERED': 'Anus (Covered)',
    'FEET_EXPOSED': 'Feet (Exposed)',
    'FEET_COVERED': 'Feet (Covered)'
}


class BodyCropMaskEQX:
    """
    A ComfyUI node that detects body parts using NudeNet and creates crops and masks
    for the detected regions.
    """

    NODE_NAME = "Body Crop & Mask EQX"
    CATEGORY = "EQX/Detection"

    def __init__(self):
        self.detector = None
        print(f"[BodyCropMaskEQX] Initializing... NUDENET_AVAILABLE = {NUDENET_AVAILABLE}")

        if NUDENET_AVAILABLE:
            try:
                print("[BodyCropMaskEQX] Attempting to load NudeNet detector...")
                self.detector = NudeDetector()
                print("[BodyCropMaskEQX] ✓ NudeNet detector loaded successfully")
                print(f"[BodyCropMaskEQX] Detector type: {type(self.detector)}")
            except Exception as e:
                print(f"[BodyCropMaskEQX] ✗ Error loading NudeNet detector: {e}")
                self.detector = None
        else:
            print("[BodyCropMaskEQX] ✗ NudeNet not available - skipping detector initialization")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "body_part": (list(CATEGORY_DISPLAY_NAMES.values()), {
                    "default": "Face (Female)"
                }),
                "padding_pixels": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 200,
                    "step": 5,
                    "display": "number"
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "detection_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "display": "number"
                }),
                "mask_blur": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "display": "slider"
                }),
                "combine_masks": ("BOOLEAN", {
                    "default": False,
                    "display": "checkbox"
                }),
                "square_crop": ("BOOLEAN", {
                    "default": False,
                    "display": "checkbox"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("body_crop", "body_mask", "masked_original", "detections_count", "detected_parts")
    FUNCTION = "detect_and_mask"
    OUTPUT_NODE = False

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image."""
        if tensor.dim() == 4:
            tensor = tensor[0]

        img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)

        if img_np.shape[-1] == 3:
            return Image.fromarray(img_np, mode='RGB')
        elif img_np.shape[-1] == 4:
            return Image.fromarray(img_np, mode='RGBA')
        else:
            return Image.fromarray(img_np.squeeze(), mode='L')

    def pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor."""
        img_np = np.array(pil_image).astype(np.float32) / 255.0

        if len(img_np.shape) == 2:
            img_np = np.expand_dims(img_np, axis=-1)

        return torch.from_numpy(img_np).unsqueeze(0)

    def create_body_mask(self, image_size: Tuple[int, int], detections: List[Dict],
                        blur_radius: int = 8, combine: bool = False) -> np.ndarray:
        """
        Create a mask for detected body parts.

        Args:
            image_size: (width, height) of the image
            detections: List of detection dictionaries with bounding boxes
            blur_radius: Radius for Gaussian blur
            combine: Whether to combine all detections into one mask

        Returns:
            Mask as numpy array
        """
        width, height = image_size
        mask = np.zeros((height, width), dtype=np.uint8)

        print(f"[BodyCropMaskEQX] Creating mask for {len(detections)} detections (combine={combine})")

        for i, detection in enumerate(detections):
            box = detection.get('box', [])
            if len(box) >= 4:
                # NudeNet returns [x, y, width, height] format
                x, y, w, h = box[:4]

                # Convert to x1, y1, x2, y2 format
                x1 = int(x)
                y1 = int(y)
                x2 = int(x + w)
                y2 = int(y + h)

                # Ensure coordinates are within bounds
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                print(f"[BodyCropMaskEQX] Adding mask region {i}: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

                # Draw rectangle for this detection
                mask[y1:y2, x1:x2] = 255

                # Continue to next detection if combining, otherwise stop
                if not combine:
                    break  # Only use first detection if not combining

        # Apply Gaussian blur if requested
        if blur_radius > 0:
            mask_pil = Image.fromarray(mask, mode='L')
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            mask = np.array(mask_pil)

        return mask

    def get_crop_bbox(self, detections: List[Dict], padding: int, image_size: Tuple[int, int],
                     square: bool = False) -> Tuple[int, int, int, int]:
        """
        Calculate bounding box for crop with padding.

        Args:
            detections: List of detection dictionaries
            padding: Padding in pixels
            image_size: (width, height) of image
            square: Whether to make the crop square

        Returns:
            Tuple of (x1, y1, x2, y2) for crop
        """
        if not detections:
            return (0, 0, 64, 64)

        width, height = image_size

        # Get first detection or combine all
        if len(detections) == 1:
            box = detections[0].get('box', [0, 0, 64, 64])
            # NudeNet returns [x, y, width, height] format
            x, y, w, h = box[:4]
            x1 = int(x)
            y1 = int(y)
            x2 = int(x + w)
            y2 = int(y + h)
        else:
            # Combine all detection boxes
            min_x = width
            min_y = height
            max_x = 0
            max_y = 0

            for det in detections:
                box = det.get('box', [])
                if len(box) >= 4:
                    # NudeNet returns [x, y, width, height] format
                    x, y, w, h = box[:4]
                    x1 = int(x)
                    y1 = int(y)
                    x2 = int(x + w)
                    y2 = int(y + h)

                    min_x = min(min_x, x1)
                    min_y = min(min_y, y1)
                    max_x = max(max_x, x2)
                    max_y = max(max_y, y2)

            x1, y1, x2, y2 = min_x, min_y, max_x, max_y

        # Add padding
        x1 = max(0, int(x1 - padding))
        y1 = max(0, int(y1 - padding))
        x2 = min(width, int(x2 + padding))
        y2 = min(height, int(y2 + padding))

        # Make square if requested
        if square:
            crop_width = x2 - x1
            crop_height = y2 - y1

            if crop_width != crop_height:
                # Use larger dimension
                target_size = max(crop_width, crop_height)

                # Center the square crop
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                x1 = max(0, center_x - target_size // 2)
                y1 = max(0, center_y - target_size // 2)
                x2 = min(width, x1 + target_size)
                y2 = min(height, y1 + target_size)

                # Adjust if hitting image boundaries
                if x2 - x1 != y2 - y1:
                    target_size = min(x2 - x1, y2 - y1)
                    x2 = x1 + target_size
                    y2 = y1 + target_size

        return (x1, y1, x2, y2)

    def detect_and_mask(self, image: torch.Tensor, body_part: str, padding_pixels: int,
                       confidence_threshold: float, detection_index: int, mask_blur: int,
                       combine_masks: bool, square_crop: bool) -> Tuple:
        """
        Detect body parts and create crops and masks.

        Returns:
            Tuple of (body_crop, body_mask, masked_original, detections_count, detected_parts)
        """
        print(f"\n[BodyCropMaskEQX] === Starting detection ===")
        print(f"[BodyCropMaskEQX] Target body part: {body_part}")
        print(f"[BodyCropMaskEQX] Confidence threshold: {confidence_threshold}")
        print(f"[BodyCropMaskEQX] Image shape: {image.shape}")

        if not NUDENET_AVAILABLE:
            error_msg = "NudeNet library not installed"
            print(f"[BodyCropMaskEQX] ERROR: {error_msg}")
            empty_crop = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32)
            return (empty_crop, empty_mask, image, 0, error_msg)

        if self.detector is None:
            error_msg = "NudeNet detector not initialized"
            print(f"[BodyCropMaskEQX] ERROR: {error_msg}")
            empty_crop = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32)
            return (empty_crop, empty_mask, image, 0, error_msg)

        # Convert display name back to internal category name
        target_category = None
        for internal, display in CATEGORY_DISPLAY_NAMES.items():
            if display == body_part:
                target_category = internal
                break

        if target_category is None:
            target_category = 'FACE_FEMALE'  # Fallback

        # Convert tensor to PIL for detection
        pil_image = self.tensor_to_pil(image)
        img_np = np.array(pil_image)

        # Save temporary file for NudeNet
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil_image.save(tmp.name)
            temp_path = tmp.name

        try:
            # Perform detection
            print(f"[BodyCropMaskEQX] Running NudeNet detection on: {temp_path}")
            results = self.detector.detect(temp_path)
            print(f"[BodyCropMaskEQX] Raw detection results: {len(results)} detections found")

            # Log all detections for debugging
            for i, detection in enumerate(results):
                label = detection.get('class') or detection.get('label')
                score = detection.get('score', 0)
                box = detection.get('box', [])
                if len(box) >= 4:
                    x, y, w, h = box[:4]
                    print(f"[BodyCropMaskEQX] Detection {i}: {label} (score: {score:.3f}) box: [x={x}, y={y}, w={w}, h={h}]")
                else:
                    print(f"[BodyCropMaskEQX] Detection {i}: {label} (score: {score:.3f}) box: invalid")

            # Filter detections for target body part
            filtered_detections = []
            print(f"[BodyCropMaskEQX] Filtering for: {target_category}")

            for detection in results:
                label = detection.get('class') or detection.get('label')
                score = detection.get('score', 0)

                # Check both uppercase and mixed case versions
                label_upper = label.upper() if label else ""

                if (label == target_category or label_upper == target_category) and score >= confidence_threshold:
                    filtered_detections.append(detection)
                    print(f"[BodyCropMaskEQX] ✓ Matched detection: {label} with score {score:.3f}")

            # Sort by confidence score
            filtered_detections.sort(key=lambda x: x.get('score', 0), reverse=True)
            print(f"[BodyCropMaskEQX] Filtered detections: {len(filtered_detections)} match criteria")

            # Select specific detection if index provided
            if filtered_detections:
                if detection_index < len(filtered_detections):
                    selected_detections = [filtered_detections[detection_index]]
                else:
                    selected_detections = [filtered_detections[0]]

                if combine_masks:
                    selected_detections = filtered_detections  # Use all for combined mask
            else:
                # No detections found
                print(f"[BodyCropMaskEQX] No {body_part} detected")
                empty_crop = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                empty_mask = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32)
                return (empty_crop, empty_mask, image, 0, f"No {body_part} detected")

            # Get crop bounding box
            crop_bbox = self.get_crop_bbox(
                selected_detections,
                padding_pixels,
                (img_np.shape[1], img_np.shape[0]),
                square_crop
            )

            # Create crop
            x1, y1, x2, y2 = crop_bbox
            print(f"[BodyCropMaskEQX] Crop bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            print(f"[BodyCropMaskEQX] Using {len(selected_detections)} detection(s) for crop/mask")

            # Validate coordinates before cropping
            if x2 <= x1 or y2 <= y1:
                print(f"[BodyCropMaskEQX] Invalid crop coordinates, using fallback")
                x1, y1, x2, y2 = 0, 0, min(64, img_np.shape[1]), min(64, img_np.shape[0])

            body_crop = pil_image.crop((x1, y1, x2, y2))
            body_crop_tensor = self.pil_to_tensor(body_crop)

            # Create mask
            mask_np = self.create_body_mask(
                (img_np.shape[1], img_np.shape[0]),
                selected_detections,
                mask_blur,
                combine_masks
            )

            # Convert mask to tensor
            mask_tensor = torch.from_numpy(mask_np.astype(np.float32) / 255.0).unsqueeze(0)

            # Create masked original
            mask_3channel = mask_tensor.unsqueeze(-1).expand(-1, -1, -1, image.shape[-1])
            masked_original = image * mask_3channel

            # Create summary of detected parts
            detected_summary = f"Detected {len(filtered_detections)} {body_part}(s)"
            if filtered_detections:
                scores = [f"{d.get('score', 0):.2f}" for d in filtered_detections[:3]]
                detected_summary += f" (scores: {', '.join(scores)})"

            print(f"[BodyCropMaskEQX] Summary: {detected_summary}")
            print(f"[BodyCropMaskEQX] === Detection complete ===")

            return (body_crop_tensor, mask_tensor, masked_original,
                   len(filtered_detections), detected_summary)

        except Exception as e:
            import traceback
            print(f"[BodyCropMaskEQX] ERROR during detection: {e}")
            print(f"[BodyCropMaskEQX] Traceback: {traceback.format_exc()}")
            empty_crop = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32)
            return (empty_crop, empty_mask, image, 0, f"Error: {str(e)}")

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force re-execution when inputs change."""
        return float("nan")


# Node registration
NODE_CLASS_MAPPINGS = {
    "BodyCropMaskEQX": BodyCropMaskEQX
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BodyCropMaskEQX": "Body Crop & Mask EQX"
}