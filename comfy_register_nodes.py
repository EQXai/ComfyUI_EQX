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
        # Si no existe el fichero, devolvemos vacíos
        if not os.path.isfile(file_path):
            return "", "", "", seed

        prompt_list = []
        # Leemos línea a línea y extraemos ID, prompt positivo y negativo
        with open(file_path, 'r', encoding='utf-8') as file:
            for raw_line in file:
                line = raw_line.strip()
                # Eliminamos coma final si existe
                if line.endswith(','):
                    line = line[:-1]
                # Pattern: {{{ID}}}{{positive}}{{negative}}
                m = re.match(r"\{\{\{([^}]*)\}\}\}\{\{([^}]*)\}\}\{([^}]*)\}", line)
                if m:
                    identificador, positive, negative = m.groups()
                    prompt_list.append((identificador.strip(), positive.strip(), negative.strip()))

        if not prompt_list:
            return "", "", "", seed

        # Selección basada en seed
        index = seed % len(prompt_list)
        identificador, positive, negative = prompt_list[index]
        seed += 1
        return (
            identificador,
            positive,
            negative,
            seed
        )

NODE_CLASS_MAPPINGS = {
    "Load Prompt From File - EQX": LoadPromptFromFileEQXNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load Prompt From File - EQX": "Load Prompt From File - EQX"
}
