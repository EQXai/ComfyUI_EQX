import torch
import torch.nn.functional as F

class VideoFragmentsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "fragment1": ("IMAGE",),
                "fragment2": ("IMAGE",),
                "fragment3": ("IMAGE",),
                "fragment4": ("IMAGE",),
                "fragment5": ("IMAGE",),
                "fragment6": ("IMAGE",),
                "fragment7": ("IMAGE",),
                "fragment8": ("IMAGE",),
                "fragment9": ("IMAGE",),
                "fragment10": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "combine_fragments"
    CATEGORY = "EQX/Video"

    def combine_fragments(self, **kwargs):
        combined_images = []

        for i in range(1, 11):
            fragment_key = f"fragment{i}"
            if fragment_key in kwargs and kwargs[fragment_key] is not None:
                fragment = kwargs[fragment_key]
                if isinstance(fragment, torch.Tensor):
                    if fragment.dim() == 3:
                        fragment = fragment.unsqueeze(0)
                    combined_images.append(fragment)

        if not combined_images:
            empty_image = torch.zeros((1, 64, 64, 3))
            return (empty_image,)

        target_height = combined_images[0].shape[1]
        target_width = combined_images[0].shape[2]

        resized_images = []
        for img in combined_images:
            batch_size, height, width, channels = img.shape

            if height != target_height or width != target_width:
                img_permuted = img.permute(0, 3, 1, 2)
                img_resized = F.interpolate(img_permuted, size=(target_height, target_width), mode='bilinear', align_corners=False)
                img_resized = img_resized.permute(0, 2, 3, 1)
                resized_images.append(img_resized)
            else:
                resized_images.append(img)

        result = torch.cat(resized_images, dim=0)
        return (result,)

NODE_CLASS_MAPPINGS = {
    "VideoFragmentsNode": VideoFragmentsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFragmentsNode": "Video Fragments EQX"
}