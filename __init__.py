import os
import sys
import subprocess
import importlib

# ASCII Art Banner
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

# NSFW Detector
from .nsfw_detector_eqx import NSFW_Detector_EQX
NODE_CLASS_MAPPINGS["NSFW Detector EQX"] = NSFW_Detector_EQX
NODE_DISPLAY_NAME_MAPPINGS["NSFW Detector EQX"] = "NSFW Detector EQX"

# NSFW Detector Advanced EQX
from .nsfw_detector_advanced_eqx import NSFWDetectorAdvancedEQX
NODE_CLASS_MAPPINGS["NSFW Detector Advanced EQX"] = NSFWDetectorAdvancedEQX
NODE_DISPLAY_NAME_MAPPINGS["NSFW Detector Advanced EQX"] = "NSFW Detector Advanced EQX"


# FaceCT Nodes
# Ensure the main dependency is met before trying to load the nodes.
if _ensure_package("facexlib"):
    try:
        from .face_ct_nodes import NODE_CLASS_MAPPINGS as face_ct_class_mappings
        from .face_ct_nodes import NODE_DISPLAY_NAME_MAPPINGS as face_ct_display_name_mappings
        NODE_CLASS_MAPPINGS.update(face_ct_class_mappings)
        NODE_DISPLAY_NAME_MAPPINGS.update(face_ct_display_name_mappings)
    except Exception as e:
        print(f"[ComfyUI_EQX] Warning: Could not import FaceCT nodes even after successful dependency check. Error: {e}")

__all__ = list(NODE_CLASS_MAPPINGS)

print(f"[ComfyUI_EQX] Loaded {len(NODE_CLASS_MAPPINGS)} nodes (v{__version__})")
print("═══════════════════════════════════════")
print("═══════════════════════════════════════")
print("═══════════════════════════════════════")