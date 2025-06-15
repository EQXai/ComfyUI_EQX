import os
from facexlib.detection import RetinaFace
import torch

from .face_ct_utils import models_dir, tensor2cv, tensor2pil, pil2hex

class LoadRetinaFace_EQX:
    models_dir = os.path.join(models_dir, 'facexlib')
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{}}
    
    RETURN_TYPES = ("RETINAFACE", )
    RETURN_NAMES = ("MODEL", )
    FUNCTION = "load"
    CATEGORY = "EQX/Face Detection"
    def load(self):
        from facexlib.detection import init_detection_model
        return (init_detection_model("retinaface_resnet50", model_rootpath=self.models_dir), )

class CountFaces_EQX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("RETINAFACE", ),
                "image": ("IMAGE", ),
                "confidence": ("FLOAT", {"default": 0.8, "min": 0, "max": 1}),
            },
            "optional": {
                "save_directory": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("INT", "IMAGE")
    RETURN_NAMES = ("face_count", "image_out")
    FUNCTION = "count"
    CATEGORY = "EQX/Face Detection"

    def count(self, model: RetinaFace, image: torch.Tensor, confidence: float, save_directory: str = ""):
        # 1. Count faces (existing logic)
        with torch.no_grad():
            bboxes = model.detect_faces(tensor2cv(image), confidence)
        face_count = len(bboxes)

        # 2. Save image if directory is provided
        if save_directory and save_directory.strip():
            # Check if the base directory exists
            if not os.path.isdir(save_directory):
                print(f"[CountFaces_EQX] Warning: Save directory '{save_directory}' does not exist. Image will not be saved.")
            else:
                try:
                    # Create subdirectory based on face count
                    target_dir = os.path.join(save_directory, str(face_count))
                    os.makedirs(target_dir, exist_ok=True)

                    # Convert tensor to PIL image to get a hash and save
                    pil_image = tensor2pil(image)
                    
                    # Generate a unique filename using a hash of the image content
                    image_hash = pil2hex(image)
                    filename = f"image_{image_hash[:16]}.png"
                    
                    # Construct the full path and save the image
                    full_path = os.path.join(target_dir, filename)
                    pil_image.save(full_path)
                    print(f"[CountFaces_EQX] Saved image to {full_path}")

                except Exception as e:
                    print(f"[CountFaces_EQX] Error: Could not save image. Reason: {e}")

        # 3. Return face count and the original image tensor
        return (face_count, image)

NODE_CLASS_MAPPINGS = {
    "LoadRetinaFace_EQX": LoadRetinaFace_EQX,
    "CountFaces_EQX": CountFaces_EQX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadRetinaFace_EQX": "Load RetinaFace (FaceCT)",
    "CountFaces_EQX": "Count Faces (FaceCT)",
} 