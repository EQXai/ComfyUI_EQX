class ExtractFilename:
    CATEGORY = "text"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "extract"

    def extract(self, text):
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        parts = text.replace("/", "\\").split("\\")
        filename = parts[-1] if parts else ""
        return (filename,)

NODE_CLASS_MAPPINGS = {
    "Extract Filename - EQX": ExtractFilename
}
