import torch
import numpy as np
import cv2
from typing import List, Literal, Tuple

# BBox type definition for type hinting
BBox = Tuple[int, int, int, int]

# Dependency check for facexlib
try:
    from facexlib.detection import RetinaFace
except ImportError:
    print("Warning: facexlib not found. The FaceDetectOut node will not be functional.")
    RetinaFace = None

def tensor2cv(image: torch.Tensor) -> np.ndarray:
    """Converts a PyTorch tensor (H, W, C) from RGB [0,1] to an OpenCV image (BGR, uint8)."""
    if image.dim() == 4 and image.shape[0] == 1:
        image = image.squeeze(0)
    image_np = image.cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

def cv2tensor(image: np.ndarray) -> torch.Tensor:
    """Converts an OpenCV image (BGR, uint8) to a PyTorch tensor (1, H, W, C) RGB [0,1]."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    return torch.from_numpy(image).unsqueeze(0)

def hex2bgr(hex_color: str) -> Tuple[int, int, int]:
    """Converts a hex color string to a BGR tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

class FaceDetectOut:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("RETINAFACE", ),
                "image": ("IMAGE", ),
                "confidence": ("FLOAT", {"default": 0.8, "min": 0, "max": 1}),
                "margin": ("FLOAT", {"default": 0.5, "min": 0, "max": 2.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "BBOX", "BOOLEAN")
    RETURN_NAMES = ("face_image", "preview", "bbox", "is_partial_face")
    FUNCTION = "crop"
    CATEGORY = "CFaceSwap"

    def crop(self, model: RetinaFace, image: torch.Tensor, confidence: float, margin: float):
        if RetinaFace is None:
             raise ImportError("facexlib is not installed. Please install it to use this node.")
        
        with torch.no_grad():
            bboxes_with_landmarks = model.detect_faces(tensor2cv(image), confidence)

        if len(bboxes_with_landmarks) == 0:
            print("no face detected")
            dummy_face = torch.zeros((1, 64, 64, 3), dtype=torch.float32, device=image.device)
            return (dummy_face, image, (0, 0, 0, 0), False)

        img_height, img_width = image.shape[1], image.shape[2]
        
        x0, y0, x1, y1, *_ = bboxes_with_landmarks[0]
        original_bbox = (int(min(x0, x1)), int(min(y0, y1)), int(abs(x1 - x0)), int(abs(y1 - y0)))
        is_partial = self.is_bbox_partially_outside(original_bbox, margin, img_width, img_height)

        detection_preview = self.visualize_detection(tensor2cv(image), bboxes_with_landmarks)
        
        processed_bboxes = [
            self.add_margin_and_make_square(
                (int(min(x0, x1)), int(min(y0, y1)), int(abs(x1 - x0)), int(abs(y1 - y0))),
                margin,
                img_width=img_width,
                img_height=img_height
            ) for (x0, y0, x1, y1, *_) in bboxes_with_landmarks
        ]
        
        faces = self.crop_faces(processed_bboxes, image)
        
        return (faces[0].unsqueeze(0), cv2tensor(detection_preview).to(image.device), processed_bboxes[0], is_partial)
    
    def is_bbox_partially_outside(self, bbox: BBox, margin_percent: float, img_width: int, img_height: int) -> bool:
        """Checks if a bbox with a given margin percentage would extend beyond image boundaries."""
        x, y, w, h = bbox
        
        margin_w_half = int(w * margin_percent / 2)
        margin_h_half = int(h * margin_percent / 2)

        is_out = (
            (x - margin_w_half) < 0 or
            (y - margin_h_half) < 0 or
            (x + w + margin_w_half) > img_width or
            (y + h + margin_h_half) > img_height
        )
        return is_out

    def crop_faces(self, bboxes: List[BBox], image: torch.Tensor) -> List[torch.Tensor]:
        """Returns: list of Tensor[h,w,c]"""
        return [image[0, y:y+h, x:x+w, :] for (x, y, w, h) in bboxes]
    
    def scale_faces(self, faces: List[torch.Tensor], size: int, upscaler: Literal["linear"]="linear") -> List[torch.Tensor]:
        """Args: faces: list of Tensor[h,w,c]"""
        scaled_faces: List[torch.Tensor] = []
        for face in faces:
            # Change layout to [batch, channel, height, width] for interpolation
            face_permuted = face.permute(2, 0, 1).unsqueeze(0)
            
            if upscaler == "linear":
                scaled_face = torch.nn.functional.interpolate(face_permuted, size=(size, size), mode="bilinear", align_corners=True)
            elif upscaler == "nearest":
                scaled_face = torch.nn.functional.interpolate(face_permuted, size=(size, size), mode="nearest")
            else:
                raise ValueError(f"Invalid upscaler: {upscaler}")
            
            # Change layout back to [height, width, channel]
            scaled_face = scaled_face.squeeze(0).permute(1, 2, 0)
            scaled_faces.append(scaled_face)
        
        return scaled_faces

    def visualize_margin(self, img: np.ndarray, bboxes: List[BBox]) -> np.ndarray:
        img_copy = np.copy(img)
        for bbox in bboxes:
            x, y, w, h = bbox
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), hex2bgr("#710193"), 2)
        return img_copy

    def visualize_detection(self, img: np.ndarray, bboxes_and_landmarks: list) -> np.ndarray:
        """Args: img (np.ndarray): bgr"""
        img_copy = np.copy(img)
        for b in bboxes_and_landmarks:
            cv2.putText(img_copy, f'{b[4]:.4f}', (int(b[0]), int(b[1] + 12)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            b_int = list(map(int, b))
            cv2.rectangle(img_copy, (b_int[0], b_int[1]), (b_int[2], b_int[3]), (0, 0, 255), 2)
            # Landmarks for RetinaFace
            cv2.circle(img_copy, (b_int[5], b_int[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_copy, (b_int[7], b_int[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_copy, (b_int[9], b_int[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_copy, (b_int[11], b_int[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_copy, (b_int[13], b_int[14]), 1, (255, 0, 0), 4)
        return img_copy
    
    def add_margin_and_make_square(self, bbox: BBox, margin_percent: float, img_width: int, img_height: int) -> BBox:
        x, y, w, h = bbox
        
        # Calculate margin in pixels from percentage
        margin_w = int(w * margin_percent)
        margin_h = int(h * margin_percent)
        
        # Add margin to the bounding box, centered
        x = max(0, x - margin_w // 2)
        y = max(0, y - margin_h // 2)
        w = min(img_width - x, w + margin_w)
        h = min(img_height - y, h + margin_h)
        
        # Make the bounding box square
        cx, cy = x + w // 2, y + h // 2
        max_side = max(w, h)
        x_sq = max(0, cx - max_side // 2)
        y_sq = max(0, cy - max_side // 2)
        side = min(max_side, img_width - x_sq, img_height - y_sq)
        
        return (x_sq, y_sq, side, side)

NODE_CLASS_MAPPINGS = {
    "FaceDetectOut": FaceDetectOut
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDetectOut": "Face Detect Out"
} 