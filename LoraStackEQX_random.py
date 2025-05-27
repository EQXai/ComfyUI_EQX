import os
import random
import folder_paths
from pathlib import Path

class LoraStackEQX_random:
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "num_loras_to_select": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
                "strength_model": ("FLOAT", {"default": 0.3, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "subfolder_name": ("STRING", {"default": "Combine"}),
            }
        }
        for i in range(1, 25):
            inputs["required"][f"subfolder_name_{i}"] = ("STRING", {"default": ""})
        return inputs

    RETURN_TYPES = ("LORA_STACK", "STRING")
    RETURN_NAMES = ("LORA_STACK", "show_text")
    FUNCTION = "stack_loras"
    CATEGORY = "EQX"

    def stack_loras(self, num_loras_to_select, strength_model, strength_clip, seed, **kwargs):
        lora_dir = folder_paths.get_folder_paths("loras")[0]
        all_files = []

        # Collect subfolder names from kwargs and the original subfolder_name
        subfolder_names = [kwargs[f"subfolder_name_{i}"] for i in range(1, 25) if kwargs.get(f"subfolder_name_{i}")]
        if kwargs.get("subfolder_name"):
             subfolder_names.append(kwargs.get("subfolder_name"))

        for subfolder_name_item in subfolder_names:
            if not subfolder_name_item: # Skip if subfolder name is empty
                continue
            lora_path = os.path.join(lora_dir, subfolder_name_item)
            if not os.path.isdir(lora_path):
                print(f"Warning: Subfolder '{subfolder_name_item}' not found in loras directory: {lora_path}")
                continue

            files_in_subfolder = [
                os.path.join(subfolder_name_item, f) # Keep subfolder prefix for internal use
                for f in os.listdir(lora_path)
                if os.path.isfile(os.path.join(lora_path, f)) and f.lower().endswith(('.safetensors', '.pt', '.ckpt'))
            ]
            all_files.extend(files_in_subfolder)

        if not all_files:
            print(f"Warning: No LoRA files found in any of the specified subfolders.")
            return ([], "")

        n = min(num_loras_to_select, len(all_files))
        if n <= 0:
            print(f"Warning: num_loras_to_select is too low or no files to select")
            return ([], "")

        current_seed = seed or random.randint(0, 2**32 - 1)
        random.seed(current_seed)
        selected_files = random.sample(all_files, n)

        stack = []
        lines = []
        for internal_fpath in selected_files:
            stack.append((internal_fpath, strength_model, strength_clip))

            # for the tag we strip off any folder, use only the filename
            base = Path(internal_fpath).name
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
