import torch

class BatchImageTrimmer:
    """
    A ComfyUI node for trimming images from the beginning and end of a batch.
    Allows precise control over which images to keep from a sequence.
    """

    NODE_NAME = "Batch Image Trimmer EQX"
    CATEGORY = "EQX/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
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
                "mode": (["trim", "keep_range", "keep_indices"],),
                "range_start": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1
                }),
                "range_end": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000,
                    "step": 1
                }),
                "indices": ("STRING", {
                    "default": "0,1,2",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("images", "original_count", "output_count", "info")
    FUNCTION = "trim_batch"

    def trim_batch(self, images, trim_start=0, trim_end=0, mode="trim",
                   range_start=0, range_end=-1, indices="0,1,2"):
        """
        Trim or select specific images from a batch.

        Args:
            images: Input batch of images
            trim_start: Number of images to remove from the beginning
            trim_end: Number of images to remove from the end
            mode: Operation mode - "trim", "keep_range", or "keep_indices"
            range_start: Start index for keep_range mode (inclusive)
            range_end: End index for keep_range mode (inclusive, -1 means last)
            indices: Comma-separated indices for keep_indices mode
        """

        batch_size = images.shape[0]

        if batch_size == 0:
            empty = torch.zeros((1, 64, 64, 3))
            return (empty, 0, 0, "No input images")

        if mode == "trim":
            # Standard trim mode - remove from start and end
            if trim_start + trim_end >= batch_size:
                # If trimming everything, return at least one image
                result = images[0:1]
                info = f"Warning: Trim values ({trim_start}+{trim_end}) >= batch size ({batch_size}). Keeping first image only."
            else:
                end_index = batch_size - trim_end if trim_end > 0 else batch_size
                result = images[trim_start:end_index]
                info = f"Trimmed: removed {trim_start} from start, {trim_end} from end. Range: [{trim_start}:{end_index}]"

        elif mode == "keep_range":
            # Keep a specific range of images
            if range_end == -1:
                range_end = batch_size - 1

            # Clamp values to valid range
            range_start = max(0, min(range_start, batch_size - 1))
            range_end = max(0, min(range_end, batch_size - 1))

            if range_start > range_end:
                range_start, range_end = range_end, range_start

            result = images[range_start:range_end + 1]
            info = f"Keep range: [{range_start}:{range_end + 1}] from {batch_size} images"

        elif mode == "keep_indices":
            # Keep specific indices
            try:
                # Parse comma-separated indices
                idx_list = []
                for idx_str in indices.split(','):
                    idx_str = idx_str.strip()
                    if ':' in idx_str:
                        # Handle range notation like "5:10"
                        start, end = idx_str.split(':')
                        start = int(start) if start else 0
                        end = int(end) if end else batch_size
                        idx_list.extend(range(start, min(end, batch_size)))
                    else:
                        idx = int(idx_str)
                        if 0 <= idx < batch_size:
                            idx_list.append(idx)

                if not idx_list:
                    # If no valid indices, keep first image
                    idx_list = [0]
                    info = "Warning: No valid indices provided. Keeping first image."
                else:
                    # Remove duplicates and sort
                    idx_list = sorted(set(idx_list))
                    info = f"Keep indices: {idx_list} from {batch_size} images"

                result = images[idx_list]

            except (ValueError, IndexError) as e:
                # If parsing fails, default to keeping all
                result = images
                info = f"Error parsing indices: {e}. Keeping all images."
        else:
            result = images
            info = f"Unknown mode: {mode}. Keeping all images."

        output_count = result.shape[0]

        # Add statistics to info
        info += f"\nOriginal: {batch_size} images → Output: {output_count} images"

        return (result, batch_size, output_count, info)

NODE_CLASS_MAPPINGS = {
    "BatchImageTrimmer": BatchImageTrimmer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchImageTrimmer": "Batch Image Trimmer EQX"
}