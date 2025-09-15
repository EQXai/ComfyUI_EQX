import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List
import os
import sys

# Add thirdparty to path for facexlib
third_party_dir = os.path.join(os.path.dirname(__file__), 'thirdparty')
if third_party_dir not in sys.path:
    sys.path.append(third_party_dir)

try:
    from facexlib.detection import RetinaFace
    FACEXLIB_AVAILABLE = True
except ImportError:
    print("[FaceCropMaskEQX] Warning: facexlib not found. The node will not be functional.")
    FACEXLIB_AVAILABLE = False


class FaceCropMaskEQX:
    """
    A ComfyUI node that detects faces in images and creates both a cropped face image
    and a mask showing the face location in the original resolution.
    """

    NODE_NAME = "Face Crop & Mask EQX"
    CATEGORY = "EQX/Detection"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "padding_pixels": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 500,
                    "step": 5,
                    "display": "number"
                }),
                "square_crop": ("BOOLEAN", {
                    "default": False,
                    "display": "checkbox"
                }),
                "face_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "display": "number"
                }),
                "confidence": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "mask_blur": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "display": "slider"
                }),
            },
            "optional": {
                "model": ("RETINAFACE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "INT")
    RETURN_NAMES = ("face_crop", "face_mask", "masked_original", "faces_count")
    FUNCTION = "detect_and_crop"
    OUTPUT_NODE = False

    def __init__(self):
        self.model = None

    def load_model(self):
        """Load the RetinaFace model if not already loaded."""
        if not FACEXLIB_AVAILABLE:
            raise ImportError("facexlib is required for face detection. Please install it.")

        if self.model is None:
            print("[FaceCropMaskEQX] Loading RetinaFace model...")
            # Initialize RetinaFace with default settings
            self.model = RetinaFace(half=False, device='cpu')
            print("[FaceCropMaskEQX] Model loaded successfully")

        return self.model

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image."""
        # Assuming tensor is in [B, H, W, C] format with values 0-1
        if tensor.dim() == 4:
            tensor = tensor[0]  # Take first image from batch

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

        if len(img_np.shape) == 2:  # Grayscale
            img_np = np.expand_dims(img_np, axis=-1)

        return torch.from_numpy(img_np).unsqueeze(0)

    def create_face_mask(self, original_size: Tuple[int, int], bbox: Tuple[int, int, int, int],
                        blur_radius: int = 8) -> np.ndarray:
        """
        Create a mask with white where the face is and black elsewhere.

        Args:
            original_size: (width, height) of the original image
            bbox: (x, y, width, height) of the face bounding box
            blur_radius: Radius for Gaussian blur to soften mask edges

        Returns:
            Mask as numpy array
        """
        width, height = original_size
        mask = np.zeros((height, width), dtype=np.uint8)

        x, y, w, h = bbox
        # Ensure coordinates are within bounds
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        x2 = min(x + w, width)
        y2 = min(y + h, height)

        # Draw white rectangle for face area
        mask[y:y2, x:x2] = 255

        # Apply Gaussian blur if requested
        if blur_radius > 0:
            from PIL import ImageFilter
            mask_pil = Image.fromarray(mask, mode='L')
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            mask = np.array(mask_pil)

        return mask

    def detect_and_crop(self, image: torch.Tensor, padding_pixels: int, square_crop: bool,
                       face_index: int, confidence: float, mask_blur: int,
                       model: Optional[object] = None) -> Tuple:
        """
        Detect faces and create cropped image and mask.

        Args:
            image: Input image tensor
            padding_pixels: Padding around face in pixels
            square_crop: Whether to make the crop square
            face_index: Which face to use if multiple detected (0 = first/largest)
            confidence: Minimum confidence for face detection
            mask_blur: Blur radius for mask edges
            model: Optional pre-loaded RetinaFace model

        Returns:
            Tuple of (face_crop, face_mask, masked_original, faces_count)
        """

        # Use provided model or load our own
        if model is not None:
            detector = model
        else:
            detector = self.load_model()

        # Convert tensor to format suitable for face detection
        pil_image = self.tensor_to_pil(image)
        img_np = np.array(pil_image)

        # Convert RGB to BGR for OpenCV/facexlib
        if len(img_np.shape) == 3 and img_np.shape[-1] == 3:
            img_bgr = img_np[:, :, ::-1].copy()
        else:
            img_bgr = img_np.copy()

        # Detect faces
        faces = detector.detect_faces(img_bgr, confidence)

        if len(faces) == 0:
            print("[FaceCropMaskEQX] No faces detected")
            # Return empty outputs
            empty_crop = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32)
            return (empty_crop, empty_mask, image, 0)

        print(f"[FaceCropMaskEQX] Detected {len(faces)} faces")

        # Sort faces by area (largest first)
        faces_sorted = sorted(faces, key=lambda f: (f[2]-f[0]) * (f[3]-f[1]), reverse=True)

        # Select the requested face
        if face_index >= len(faces_sorted):
            face_index = 0
            print(f"[FaceCropMaskEQX] Face index {face_index} not available, using face 0")

        face = faces_sorted[face_index]
        x1, y1, x2, y2 = face[:4]

        # Calculate face dimensions
        face_width = x2 - x1
        face_height = y2 - y1
        face_center_x = (x1 + x2) / 2
        face_center_y = (y1 + y2) / 2

        # Calculate padded dimensions
        if square_crop:
            # Make it square by using the larger dimension
            max_dim = max(face_width, face_height)
            crop_width = max_dim + (padding_pixels * 2)
            crop_height = max_dim + (padding_pixels * 2)

            # Calculate coordinates from center
            x1_pad = max(0, int(face_center_x - crop_width / 2))
            y1_pad = max(0, int(face_center_y - crop_height / 2))
            x2_pad = min(img_np.shape[1], int(face_center_x + crop_width / 2))
            y2_pad = min(img_np.shape[0], int(face_center_y + crop_height / 2))

            # Ensure it stays square even at image boundaries
            actual_width = x2_pad - x1_pad
            actual_height = y2_pad - y1_pad

            if actual_width != actual_height:
                # Adjust to maintain square aspect ratio
                target_size = min(actual_width, actual_height)

                # Re-center with the target size
                x1_pad = max(0, int(face_center_x - target_size / 2))
                y1_pad = max(0, int(face_center_y - target_size / 2))
                x2_pad = min(img_np.shape[1], x1_pad + target_size)
                y2_pad = min(img_np.shape[0], y1_pad + target_size)
        else:
            # Regular rectangular crop with padding
            x1_pad = max(0, int(x1 - padding_pixels))
            y1_pad = max(0, int(y1 - padding_pixels))
            x2_pad = min(img_np.shape[1], int(x2 + padding_pixels))
            y2_pad = min(img_np.shape[0], int(y2 + padding_pixels))

        # Crop the face from the original image
        face_crop = pil_image.crop((x1_pad, y1_pad, x2_pad, y2_pad))
        face_crop_tensor = self.pil_to_tensor(face_crop)

        # Create mask (white where face is, black elsewhere)
        bbox_with_padding = (x1_pad, y1_pad, x2_pad - x1_pad, y2_pad - y1_pad)
        mask_np = self.create_face_mask(
            (img_np.shape[1], img_np.shape[0]),
            bbox_with_padding,
            mask_blur
        )

        # Convert mask to tensor (H, W) -> (1, H, W)
        mask_tensor = torch.from_numpy(mask_np.astype(np.float32) / 255.0).unsqueeze(0)

        # Create masked original (apply mask to original image)
        mask_3channel = mask_tensor.unsqueeze(-1).expand(-1, -1, -1, image.shape[-1])
        masked_original = image * mask_3channel

        return (face_crop_tensor, mask_tensor, masked_original, len(faces))

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force re-execution when inputs change."""
        return float("nan")


# Node registration
NODE_CLASS_MAPPINGS = {
    "FaceCropMaskEQX": FaceCropMaskEQX
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceCropMaskEQX": "Face Crop & Mask EQX"
}