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

__all__ = list(NODE_CLASS_MAPPINGS)

print(f"[ComfyUI_EQX] Loaded {len(NODE_CLASS_MAPPINGS)} nodes (v{__version__})")