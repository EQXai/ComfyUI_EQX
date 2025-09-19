import os
import sys
import subprocess
import importlib

# ASCII Art Banner
try:
    print("═══════════════════════════════════════")
    print("═══════════════════════════════════════")
    print("═══════════════════════════════════════")
    banner = """
███████╗ ██████╗ ██╗  ██╗
██╔════╝██╔═══██╗╚██╗██╔╝
█████╗  ██║   ██║ ╚███╔╝
██╔══╝  ██║ ╚╗██║ ██╔██╗
███████╗╚██████╔╝██╔╝ ██╗
╚══════╝     ╚═╝╚═╝  ╚═╝
"""
    print(banner)
except UnicodeEncodeError:
    # Fallback to simple ASCII if Unicode characters fail
    print("=" * 39)
    print("=" * 39)
    print("=" * 39)
    print("[ComfyUI_EQX]")

def _ensure_package(package):
    """Checks if a package is installed, and if not, tries to install it from PyPI."""
    try:
        # First, try to import the package. This will use the bundled `thirdparty` lib if available.
        importlib.import_module(package)
        return True
    except ImportError:
        print(f"[ComfyUI_EQX] Package '{package}' not found. Attempting to install from PyPI...")
        try:
            # If the import fails, try to install it via pip.
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            # Invalidate caches and try importing again to make it available in the current session.
            importlib.invalidate_caches()
            importlib.import_module(package)
            print(f"[ComfyUI_EQX] Successfully installed '{package}'.")
            return True
        except Exception as e:
            # If installation fails, print an error and return False.
            print(f"[ComfyUI_EQX] ERROR: Could not install package '{package}'. FaceCT nodes will be unavailable. Error: {e}")
            return False

# Add the thirdparty directory to the path to allow importing facexlib
# This is checked before falling back to PyPI installation.
third_party_dir = os.path.join(os.path.dirname(__file__), 'thirdparty')
if third_party_dir not in sys.path:
    sys.path.append(third_party_dir)

__version__ = "0.1.0"

import folder_paths

# Node registration tables used by ComfyUI
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# SaveImage_EQX
from .save_image_eqx import SaveImage_EQX
NODE_CLASS_MAPPINGS["SaveImage_EQX"] = SaveImage_EQX
NODE_DISPLAY_NAME_MAPPINGS["SaveImage_EQX"] = SaveImage_EQX.NODE_NAME

# File Image Selector
from .FileImageSelectorNode import FileImageSelector
NODE_CLASS_MAPPINGS["File Image Selector"] = FileImageSelector
NODE_DISPLAY_NAME_MAPPINGS["File Image Selector"] = "File Image Selector"

# Load Prompt From File - EQX
from .comfy_register_nodes import LoadPromptFromFileEQXNode
NODE_CLASS_MAPPINGS["Load Prompt From File - EQX"] = LoadPromptFromFileEQXNode
NODE_DISPLAY_NAME_MAPPINGS["Load Prompt From File - EQX"] = "Load Prompt From File - EQX"

# LoraStackEQX_random
from .LoraStackEQX_random import LoraStackEQX_random
NODE_CLASS_MAPPINGS["LoraStackEQX_random"] = LoraStackEQX_random
NODE_DISPLAY_NAME_MAPPINGS["LoraStackEQX_random"] = "Lora Stack EQX (Random)"

# Extract Filename - EQX
from .extract_filename import ExtractFilename
NODE_CLASS_MAPPINGS["Extract Filename - EQX"] = ExtractFilename
NODE_DISPLAY_NAME_MAPPINGS["Extract Filename - EQX"] = "Extract Filename - EQX"

# Extract LORA name - EQX
from .extract_safetensors_filename import ExtractSafetensorsFilename
NODE_CLASS_MAPPINGS["Extract LORA name - EQX"] = ExtractSafetensorsFilename
NODE_DISPLAY_NAME_MAPPINGS["Extract LORA name - EQX"] = "Extract LORA name - EQX"

# NSFW Detector Nodes
# Try to import NSFW detector nodes, but don't fail if nudenet is missing
try:
    from .nsfw_detector_eqx import NSFW_Detector_EQX
    NODE_CLASS_MAPPINGS["NSFW Detector EQX"] = NSFW_Detector_EQX
    NODE_DISPLAY_NAME_MAPPINGS["NSFW Detector EQX"] = "NSFW Detector EQX"

    from .nsfw_detector_advanced_eqx import NSFWDetectorAdvancedEQX
    NODE_CLASS_MAPPINGS["NSFW Detector Advanced EQX"] = NSFWDetectorAdvancedEQX
    NODE_DISPLAY_NAME_MAPPINGS["NSFW Detector Advanced EQX"] = "NSFW Detector Advanced EQX"

    # Body Crop & Mask EQX
    from .body_crop_mask_eqx import BodyCropMaskEQX
    NODE_CLASS_MAPPINGS["BodyCropMaskEQX"] = BodyCropMaskEQX
    NODE_DISPLAY_NAME_MAPPINGS["BodyCropMaskEQX"] = "Body Crop & Mask EQX"
except ImportError as e:
    print(f"[ComfyUI_EQX] Warning: NSFW/Body detector nodes unavailable - missing dependency 'nudenet'. Install with: pip install nudenet")
    print(f"[ComfyUI_EQX] Error details: {e}")

# Resolution Selector EQX
from .resolution_selector_eqx import ResolutionSelectorEQX
NODE_CLASS_MAPPINGS["ResolutionSelectorEQX"] = ResolutionSelectorEQX
NODE_DISPLAY_NAME_MAPPINGS["ResolutionSelectorEQX"] = "Resolution Selector EQX"

# Uncrop by Mask EQX
from .uncrop_by_mask_eqx import UncropByMaskEQX
NODE_CLASS_MAPPINGS["UncropByMaskEQX"] = UncropByMaskEQX
NODE_DISPLAY_NAME_MAPPINGS["UncropByMaskEQX"] = "Uncrop by Mask EQX"

# Aspect Ratio Crop EQX
from .aspect_ratio_crop_eqx import AspectRatioCropEQX
NODE_CLASS_MAPPINGS["AspectRatioCropEQX"] = AspectRatioCropEQX
NODE_DISPLAY_NAME_MAPPINGS["AspectRatioCropEQX"] = "Aspect Ratio Crop EQX"

# WorkFlow Check
from .workflow_check_node import NODE_CLASS_MAPPINGS as workflow_check_class_mappings
from .workflow_check_node import NODE_DISPLAY_NAME_MAPPINGS as workflow_check_display_name_mappings
NODE_CLASS_MAPPINGS.update(workflow_check_class_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(workflow_check_display_name_mappings)


# Logic Nodes
try:
    from .logic_nodes import NODE_CLASS_MAPPINGS as logic_class_mappings
    from .logic_nodes import NODE_DISPLAY_NAME_MAPPINGS as logic_display_name_mappings
    NODE_CLASS_MAPPINGS.update(logic_class_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(logic_display_name_mappings)
except ImportError:
    # This can happen if the file is deleted or in the middle of an update.
    print("[ComfyUI_EQX] Warning: Could not import 'logic_nodes'.")


# FaceCT Nodes
# Ensure the main dependency is met before trying to load the nodes.
if _ensure_package("facexlib"):
    try:
        from .face_ct_nodes import NODE_CLASS_MAPPINGS as face_ct_class_mappings
        from .face_ct_nodes import NODE_DISPLAY_NAME_MAPPINGS as face_ct_display_name_mappings
        NODE_CLASS_MAPPINGS.update(face_ct_class_mappings)
        NODE_DISPLAY_NAME_MAPPINGS.update(face_ct_display_name_mappings)

        # FaceDetectOut Node
        from .face_detect_out import NODE_CLASS_MAPPINGS as face_detect_out_class_mappings
        from .face_detect_out import NODE_DISPLAY_NAME_MAPPINGS as face_detect_out_display_name_mappings
        NODE_CLASS_MAPPINGS.update(face_detect_out_class_mappings)
        NODE_DISPLAY_NAME_MAPPINGS.update(face_detect_out_display_name_mappings)

        # FaceCropMaskEQX Node
        from .face_crop_mask_eqx import NODE_CLASS_MAPPINGS as face_crop_mask_class_mappings
        from .face_crop_mask_eqx import NODE_DISPLAY_NAME_MAPPINGS as face_crop_mask_display_name_mappings
        NODE_CLASS_MAPPINGS.update(face_crop_mask_class_mappings)
        NODE_DISPLAY_NAME_MAPPINGS.update(face_crop_mask_display_name_mappings)

    except Exception as e:
        print(f"[ComfyUI_EQX] Warning: Could not import FaceCT nodes even after successful dependency check. Error: {e}")

# Video Combine Nodes
try:
    from .video_combine_nodes import NODE_CLASS_MAPPINGS as video_combine_class_mappings
    from .video_combine_nodes import NODE_DISPLAY_NAME_MAPPINGS as video_combine_display_name_mappings
    NODE_CLASS_MAPPINGS.update(video_combine_class_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(video_combine_display_name_mappings)
    print("[ComfyUI_EQX] Video Combine nodes loaded successfully")
except ImportError as e:
    print(f"[ComfyUI_EQX] Warning: Could not import Video Combine nodes. Error: {e}")

# Video Fragments Node
try:
    from .video_fragments_node import NODE_CLASS_MAPPINGS as video_fragments_class_mappings
    from .video_fragments_node import NODE_DISPLAY_NAME_MAPPINGS as video_fragments_display_name_mappings
    NODE_CLASS_MAPPINGS.update(video_fragments_class_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(video_fragments_display_name_mappings)
    print("[ComfyUI_EQX] Video Fragments node loaded successfully")
except ImportError as e:
    print(f"[ComfyUI_EQX] Warning: Could not import Video Fragments node. Error: {e}")

# Image Duplicate Remover Node
try:
    from .image_duplicate_remover import NODE_CLASS_MAPPINGS as duplicate_remover_class_mappings
    from .image_duplicate_remover import NODE_DISPLAY_NAME_MAPPINGS as duplicate_remover_display_name_mappings
    NODE_CLASS_MAPPINGS.update(duplicate_remover_class_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(duplicate_remover_display_name_mappings)
    print("[ComfyUI_EQX] Image Duplicate Remover node loaded successfully")
except ImportError as e:
    print(f"[ComfyUI_EQX] Warning: Could not import Image Duplicate Remover node. Error: {e}")

# Batch Image Trimmer Node
try:
    from .batch_image_trimmer import NODE_CLASS_MAPPINGS as batch_trimmer_class_mappings
    from .batch_image_trimmer import NODE_DISPLAY_NAME_MAPPINGS as batch_trimmer_display_name_mappings
    NODE_CLASS_MAPPINGS.update(batch_trimmer_class_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(batch_trimmer_display_name_mappings)
    print("[ComfyUI_EQX] Batch Image Trimmer node loaded successfully")
except ImportError as e:
    print(f"[ComfyUI_EQX] Warning: Could not import Batch Image Trimmer node. Error: {e}")

# Video Fragments Splitter Node
try:
    from .video_fragments_splitter import NODE_CLASS_MAPPINGS as fragments_splitter_class_mappings
    from .video_fragments_splitter import NODE_DISPLAY_NAME_MAPPINGS as fragments_splitter_display_name_mappings
    NODE_CLASS_MAPPINGS.update(fragments_splitter_class_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(fragments_splitter_display_name_mappings)
    print("[ComfyUI_EQX] Video Fragments Splitter node loaded successfully")
except ImportError as e:
    print(f"[ComfyUI_EQX] Warning: Could not import Video Fragments Splitter node. Error: {e}")

# Batch Image Trimmer Multi Node
try:
    from .batch_image_trimmer_multi import NODE_CLASS_MAPPINGS as batch_trimmer_multi_class_mappings
    from .batch_image_trimmer_multi import NODE_DISPLAY_NAME_MAPPINGS as batch_trimmer_multi_display_name_mappings
    NODE_CLASS_MAPPINGS.update(batch_trimmer_multi_class_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(batch_trimmer_multi_display_name_mappings)
    print("[ComfyUI_EQX] Batch Image Trimmer Multi node loaded successfully")
except ImportError as e:
    print(f"[ComfyUI_EQX] Warning: Could not import Batch Image Trimmer Multi node. Error: {e}")

__all__ = list(NODE_CLASS_MAPPINGS)

print(f"[ComfyUI_EQX] Loaded {len(NODE_CLASS_MAPPINGS)} nodes (v{__version__})")
try:
    print("═══════════════════════════════════════")
    print("═══════════════════════════════════════")
    print("═══════════════════════════════════════")
except UnicodeEncodeError:
    print("=" * 39)
    print("=" * 39)
    print("=" * 39)