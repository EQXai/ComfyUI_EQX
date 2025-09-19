import torch

class BatchImageTrimmerMulti:
    """
    A ComfyUI node that trims multiple image batches simultaneously.
    Takes up to 10 image batch inputs and applies the same trim settings to each.
    """

    NODE_NAME = "Batch Image Trimmer Multi EQX"
    CATEGORY = "EQX/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trim_start": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "number"
                }),
                "trim_end": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "number"
                }),
            },
            "optional": {
                "batch1": ("IMAGE",),
                "batch2": ("IMAGE",),
                "batch3": ("IMAGE",),
                "batch4": ("IMAGE",),
                "batch5": ("IMAGE",),
                "batch6": ("IMAGE",),
                "batch7": ("IMAGE",),
                "batch8": ("IMAGE",),
                "batch9": ("IMAGE",),
                "batch10": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("output1", "output2", "output3", "output4", "output5", "output6", "output7", "output8", "output9", "output10", "info")
    FUNCTION = "trim_multiple_batches"

    def trim_single_batch(self, images, trim_start, trim_end):
        """
        Trim a single batch of images.

        Args:
            images: Input batch of images
            trim_start: Number of images to remove from the beginning
            trim_end: Number of images to remove from the end

        Returns:
            Trimmed batch of images
        """
        if images is None:
            return None

        batch_size = images.shape[0]

        if batch_size == 0:
            return images

        if trim_start + trim_end >= batch_size:
            # If trimming everything, return at least one image
            return images[0:1]
        else:
            end_index = batch_size - trim_end if trim_end > 0 else batch_size
            return images[trim_start:end_index]

    def trim_multiple_batches(self, trim_start=0, trim_end=0, **kwargs):
        """
        Apply the same trim settings to multiple image batches.

        Args:
            trim_start: Number of images to remove from the beginning of each batch
            trim_end: Number of images to remove from the end of each batch
            batch1-batch10: Optional image batches to trim

        Returns:
            Tuple of 10 trimmed batches plus info string
        """
        outputs = []
        info_lines = []
        processed_count = 0

        info_lines.append(f"Trim settings: remove {trim_start} from start, {trim_end} from end")
        info_lines.append("=" * 50)

        # Process each batch
        for i in range(1, 11):
            batch_key = f"batch{i}"

            if batch_key in kwargs and kwargs[batch_key] is not None:
                batch = kwargs[batch_key]
                original_size = batch.shape[0]

                # Apply trim
                trimmed = self.trim_single_batch(batch, trim_start, trim_end)

                if trimmed is not None:
                    new_size = trimmed.shape[0]
                    outputs.append(trimmed)
                    info_lines.append(f"Batch {i}: {original_size} → {new_size} images")
                    processed_count += 1
                else:
                    # Should not happen, but handle gracefully
                    empty = torch.zeros((1, 64, 64, 3))
                    outputs.append(empty)
                    info_lines.append(f"Batch {i}: Error during trimming")
            else:
                # No input for this batch - create empty placeholder
                empty = torch.zeros((1, 64, 64, 3))
                outputs.append(empty)

        # Summary
        info_lines.append("=" * 50)
        info_lines.append(f"Processed {processed_count} batches")

        if processed_count == 0:
            info_lines.append("Warning: No input batches provided")

        info = "\n".join(info_lines)

        # Ensure we always return exactly 10 outputs plus info
        while len(outputs) < 10:
            empty = torch.zeros((1, 64, 64, 3))
            outputs.append(empty)

        return tuple(outputs) + (info,)

NODE_CLASS_MAPPINGS = {
    "BatchImageTrimmerMulti": BatchImageTrimmerMulti
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchImageTrimmerMulti": "Batch Image Trimmer Multi EQX"
}