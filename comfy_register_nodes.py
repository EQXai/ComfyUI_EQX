import os
import re

class LoadPromptFromFileEQXNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "multiline": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 10000000}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("id", "prompt", "negative_prompt", "seed")
    FUNCTION = "load_prompt"
    CATEGORY = "Load"

    def load_prompt(self, file_path, seed):
        if not os.path.isfile(file_path):
            return "", "", "", seed
        with open(file_path, 'r') as file:
            data = file.read()
        pattern = r"\{{{(.*?)}}}{{(.*?)}}{([^}]*?)}"
        matches = re.finditer(pattern, data)
        prompt_list = []
        for match in matches:
            prompt_list.append(match.groups())
        if not prompt_list:
            return "", "", "", seed
        index = seed % len(prompt_list)
        identificador, positive_prompt, negative_prompt = prompt_list[index]
        seed += 1
        return (
            identificador.strip(),
            positive_prompt.strip(),
            negative_prompt.strip(),
            seed
        )

NODE_CLASS_MAPPINGS = {
    "Load Prompt From File - EQX": LoadPromptFromFileEQXNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load Prompt From File - EQX": "Load Prompt From File - EQX"
}
