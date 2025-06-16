class WorkFlowCheck:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Nudity": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "Faces": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "Close_Up_Face": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "BOOLEAN", "BOOLEAN",)
    RETURN_NAMES = ("Nudity", "Faces", "Close Up Face",)
    FUNCTION = "check"
    CATEGORY = "logic"

    def check(self, Nudity, Faces, Close_Up_Face):
        return (Nudity, Faces, Close_Up_Face,)

NODE_CLASS_MAPPINGS = {
    "WorkFlow Check": WorkFlowCheck
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WorkFlow Check": "WorkFlow Check"
} 