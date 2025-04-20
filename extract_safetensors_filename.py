import re

class ExtractSafetensorsFilename:
    CATEGORY = "text"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    FUNCTION = "extract_all"

    def extract_all(self, text):
        matches = re.findall(r'([^\\/:]+)(?=\.safetensors)', text)
        if not matches:
            return ("", "", "")

        dash_sep  = " - ".join(matches)
        underscore_sep = "_".join(matches)
        newline_list   = "\n".join(matches)

        return (dash_sep, underscore_sep, newline_list)

NODE_CLASS_MAPPINGS = {
    "Extract LORA name - EQX": ExtractSafetensorsFilename
}
