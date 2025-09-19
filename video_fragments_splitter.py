import torch
import torch.nn.functional as F

class VideoFragmentsSplitter:
    """
    A ComfyUI node that takes up to 10 image fragment inputs and outputs them separately.
    Each input can contain multiple images (batch) and will resize to match the first fragment.
    Outputs each fragment individually for parallel processing or routing.
    """

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

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("output1", "output2", "output3", "output4", "output5", "output6", "output7", "output8", "output9", "output10")
    FUNCTION = "split_fragments"
    CATEGORY = "EQX/Video"

    def split_fragments(self, **kwargs):
        outputs = []
        target_height = None
        target_width = None
        first_valid_fragment = None

        # Find the first valid fragment to get target dimensions
        for i in range(1, 11):
            fragment_key = f"fragment{i}"
            if fragment_key in kwargs and kwargs[fragment_key] is not None:
                fragment = kwargs[fragment_key]
                if isinstance(fragment, torch.Tensor) and fragment.numel() > 0:
                    if fragment.dim() == 3:
                        fragment = fragment.unsqueeze(0)
                    first_valid_fragment = fragment
                    target_height = fragment.shape[1]
                    target_width = fragment.shape[2]
                    break

        # Process each fragment
        for i in range(1, 11):
            fragment_key = f"fragment{i}"

            if fragment_key in kwargs and kwargs[fragment_key] is not None:
                fragment = kwargs[fragment_key]

                if isinstance(fragment, torch.Tensor) and fragment.numel() > 0:
                    if fragment.dim() == 3:
                        fragment = fragment.unsqueeze(0)

                    # Resize if dimensions don't match the first fragment
                    if target_height is not None and target_width is not None:
                        batch_size, height, width, channels = fragment.shape

                        if height != target_height or width != target_width:
                            # Permute to (batch, channels, height, width) for interpolation
                            fragment_permuted = fragment.permute(0, 3, 1, 2)
                            fragment_resized = F.interpolate(
                                fragment_permuted,
                                size=(target_height, target_width),
                                mode='bilinear',
                                align_corners=False
                            )
                            # Permute back to (batch, height, width, channels)
                            fragment = fragment_resized.permute(0, 2, 3, 1)

                    outputs.append(fragment)
                else:
                    # Create empty placeholder with same dimensions as first valid fragment
                    if first_valid_fragment is not None:
                        empty = torch.zeros_like(first_valid_fragment[0:1])
                    else:
                        empty = torch.zeros((1, 64, 64, 3))
                    outputs.append(empty)
            else:
                # No input provided for this fragment
                if first_valid_fragment is not None:
                    empty = torch.zeros_like(first_valid_fragment[0:1])
                else:
                    empty = torch.zeros((1, 64, 64, 3))
                outputs.append(empty)

        # Ensure we always return exactly 10 outputs
        while len(outputs) < 10:
            if first_valid_fragment is not None:
                empty = torch.zeros_like(first_valid_fragment[0:1])
            else:
                empty = torch.zeros((1, 64, 64, 3))
            outputs.append(empty)

        return tuple(outputs)

NODE_CLASS_MAPPINGS = {
    "VideoFragmentsSplitter": VideoFragmentsSplitter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFragmentsSplitter": "Video Fragments Splitter EQX"
}