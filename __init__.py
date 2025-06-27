__version__ = "0.1.0"

import folder_paths
import os
import sys
import subprocess
import importlib

# Debug banner and delimiters (taken from init_externo.py)
print("═══════════════════════════════════════")
print("═══════════════════════════════════════")
print("═══════════════════════════════════════")
_banner_eqx = """
███████╗ ██████╗ ██╗  ██╗
██╔════╝██╔═══██╗╚██╗██╔╝
█████╗  ██║   ██║ ╚███╔╝ 
██╔══╝  ██║ ╚╗██║ ██╔██╗ 
███████╗╚██████╔╝██╔╝ ██╗
╚══════╝     ╚═╝╚═╝  ╚═╝ 
"""
print(_banner_eqx)


def _ensure_package(package: str) -> bool:
    """Ensure that *package* is importable. If not installed, attempt to install it via pip.

    Returns True when the package is successfully imported (either already present or installed);
    returns False otherwise. This mirrors the helper from `init_externo.py` so other nodes can
    utilise it for optional dependencies while providing clear logging output.
    """
    try:
        importlib.import_module(package)
        return True
    except ImportError:
        print(f"[ComfyUI_EQX] Package '{package}' not found. Attempting to install from PyPI...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            importlib.invalidate_caches()
            importlib.import_module(package)
            print(f"[ComfyUI_EQX] Successfully installed '{package}'.")
            return True
        except Exception as e:
            print(f"[ComfyUI_EQX] ERROR: Could not install package '{package}'. Error: {e}")
            return False

# Make bundled third-party libs importable before falling back to PyPI
_thirdparty_dir = os.path.join(os.path.dirname(__file__), "thirdparty")
if os.path.isdir(_thirdparty_dir) and _thirdparty_dir not in sys.path:
    sys.path.append(_thirdparty_dir)

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

# Prompt Concatenate Unified - EQX
from .prompt_nodes import PromptConcatenateUnified
NODE_CLASS_MAPPINGS["Prompt Concatenate Unified - EQX"] = PromptConcatenateUnified
NODE_DISPLAY_NAME_MAPPINGS["Prompt Concatenate Unified - EQX"] = PromptConcatenateUnified.NODE_NAME

__all__ = list(NODE_CLASS_MAPPINGS)

print(f"[ComfyUI_EQX] Loaded {len(NODE_CLASS_MAPPINGS)} nodes (v{__version__})")
print("═══════════════════════════════════════")
print("═══════════════════════════════════════")
print("═══════════════════════════════════════")