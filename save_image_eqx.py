import os
import re
import json
import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from comfy.cli_args import args
import folder_paths

class SaveImage_EQX:
    NODE_NAME = "SaveImage - EQX"

    def __init__(self):
        self.base_output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":          ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "SaveImage_EQX"}),
                "filename_suffix": ("STRING", {"default": ""}),
                "save_folder":     ("STRING", {"default": ""}),
            },
            "hidden": {
                "prompt":        "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION     = "save_images"
    OUTPUT_NODE  = True
    CATEGORY     = "EQX"
    DESCRIPTION  = "Save RGB image as PNG to a custom folder, with optional prefix and suffix."

    def save_images(self, images, filename_prefix="SaveImage_EQX", filename_suffix="", save_folder="", prompt=None, extra_pnginfo=None):
        if save_folder:
            output_dir = os.path.join(self.base_output_dir, save_folder)
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = self.base_output_dir

        full_output_folder, base_filename, _, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir,
            images[0].shape[1], images[0].shape[0]
        )

        pattern = re.compile(rf"{re.escape(base_filename)}_(\d+)(?:_.*)?\.[a-zA-Z0-9]+$")
        existing_files = os.listdir(full_output_folder)
        max_counter = 0
        for fname in existing_files:
            m = pattern.match(fname)
            if m:
                num = int(m.group(1))
                if num > max_counter:
                    max_counter = num

        results = []
        for idx, img_tensor in enumerate(images):
            cnt = max_counter + idx + 1

            img_arr = (255.0 * img_tensor.cpu().numpy()).astype(np.uint8)
            img = Image.fromarray(img_arr)

            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo:
                    for k, v in extra_pnginfo.items():
                        metadata.add_text(k, json.dumps(v))

            if filename_suffix:
                out_name = f"{base_filename}_{cnt:05}_{filename_suffix}.png"
            else:
                out_name = f"{base_filename}_{cnt:05}.png"

            img.save(os.path.join(full_output_folder, out_name), pnginfo=metadata, compress_level=4)
            results.append({"filename": out_name, "subfolder": subfolder, "type": self.type})

        return {"ui": {"images": results}}
