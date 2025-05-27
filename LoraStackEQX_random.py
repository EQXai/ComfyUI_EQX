import os
import random
import folder_paths
from pathlib import Path

class LoraStackEQX_random:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "subfolder_name": ("STRING", {"default": "Combine"}),
                "num_loras_to_select": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
                "strength_model": ("FLOAT", {"default": 0.3, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("LORA_STACK", "STRING")
    RETURN_NAMES = ("LORA_STACK", "show_text")
    FUNCTION = "stack_loras"
    CATEGORY = "EQX"

    def stack_loras(self, subfolder_name, num_loras_to_select, strength_model, strength_clip, seed):
        lora_dir = folder_paths.get_folder_paths("loras")[0]
        lora_path = os.path.join(lora_dir, subfolder_name)
        if not os.path.isdir(lora_path):
            print(f"Warning: Subfolder '{subfolder_name}' not found in loras directory: {lora_path}")
            return ([], "")

        files = [
            f for f in os.listdir(lora_path)
            if os.path.isfile(os.path.join(lora_path, f)) and f.lower().endswith(('.safetensors', '.pt', '.ckpt'))
        ]
        if not files:
            print(f"Warning: No LoRA files found in {lora_path}")
            return ([], "")

        n = min(num_loras_to_select, len(files))
        if n <= 0:
            print(f"Warning: num_loras_to_select is too low or no files to select")
            return ([], "")

        current_seed = seed or random.randint(0, 2**32 - 1)
        random.seed(current_seed)
        selected = random.sample(files, n)

        stack = []
        lines = []
        for fname in selected:
            # for internal use we keep the subfolder prefix
            internal = os.path.join(subfolder_name, fname)
            stack.append((internal, strength_model, strength_clip))

            # for the tag we strip off any folder, use only the filename
            base = Path(fname).name
            lines.append(f"<lora:{base}:{strength_model:.2f}:{strength_clip:.2f}>")

        if seed == 0:
            print(f"[LoraStackEQX_random] Random seed generated: {current_seed}")

        return (stack, "\n".join(lines))

NODE_CLASS_MAPPINGS = {
    "LoraStackEQX_random": LoraStackEQX_random
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraStackEQX_random": "Lora Stack EQX (Random)"
}
