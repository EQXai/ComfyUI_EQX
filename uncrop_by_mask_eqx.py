import torch
import numpy as np
from PIL import Image, ImageFilter
from typing import Tuple, Optional


class UncropByMaskEQX:
    """
    A ComfyUI node that composites an image into another using a mask.
    The mask defines where the insert image will be placed over the base image.
    White areas in the mask = insert image, Black areas = base image.
    """

    NODE_NAME = "Uncrop by Mask EQX"
    CATEGORY = "EQX/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "insert_image": ("IMAGE",),
                "mask": ("MASK",),
                "resize_mode": (["source", "fit", "stretch", "crop_center"], {
                    "default": "source"
                }),
                "mask_blur": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "display": "slider"
                }),
                "blend_opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "feather_edges": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
                "invert_mask": ("BOOLEAN", {
                    "default": False,
                    "display": "checkbox"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("composite", "preview_mask")
    FUNCTION = "uncrop_by_mask"
    OUTPUT_NODE = False

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image."""
        # Assuming tensor is in [B, H, W, C] format with values 0-1
        if tensor.dim() == 4:
            tensor = tensor[0]  # Take first image from batch

        img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)

        if img_np.shape[-1] == 3:
            return Image.fromarray(img_np, mode='RGB')
        elif img_np.shape[-1] == 4:
            return Image.fromarray(img_np, mode='RGBA')
        elif len(img_np.shape) == 2 or img_np.shape[-1] == 1:
            img_np = img_np.squeeze()
            return Image.fromarray(img_np, mode='L')

    def pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor."""
        img_np = np.array(pil_image).astype(np.float32) / 255.0

        if len(img_np.shape) == 2:  # Grayscale
            img_np = np.expand_dims(img_np, axis=-1)

        return torch.from_numpy(img_np).unsqueeze(0)

    def mask_to_pil(self, mask: torch.Tensor) -> Image.Image:
        """Convert mask tensor to PIL Image."""
        # Mask can be [H, W] or [1, H, W] or [B, H, W]
        if mask.dim() == 3:
            mask = mask[0]  # Take first mask
        elif mask.dim() == 2:
            pass  # Already 2D

        mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(mask_np, mode='L')

    def resize_image_to_mask(self, image: Image.Image, mask_size: Tuple[int, int],
                           mode: str = "fit") -> Image.Image:
        """
        Resize image to match mask dimensions based on mode.

        Args:
            image: PIL Image to resize
            mask_size: (width, height) target size
            mode: Resize mode - "fit", "stretch", "crop_center", or "none"

        Returns:
            Resized PIL Image
        """
        target_width, target_height = mask_size

        print(f"[UncropByMaskEQX] Resizing from {image.size} to {mask_size} using mode: {mode}")

        if mode == "none":
            # If sizes don't match, we still need to handle it
            if image.size != mask_size:
                # Create a new image with target size and paste the original centered
                new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
                paste_x = (target_width - image.width) // 2
                paste_y = (target_height - image.height) // 2
                # Ensure we don't paste outside bounds
                paste_x = max(0, paste_x)
                paste_y = max(0, paste_y)
                new_image.paste(image, (paste_x, paste_y))
                return new_image
            return image

        elif mode == "stretch":
            # Simple resize without maintaining aspect ratio
            return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

        elif mode == "fit":
            # Calculate scale to fit
            scale_x = target_width / image.width
            scale_y = target_height / image.height
            scale = min(scale_x, scale_y)

            new_width = int(image.width * scale)
            new_height = int(image.height * scale)

            # Resize the image
            resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Create a new image with target dimensions and paste centered
            new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            new_image.paste(resized, (paste_x, paste_y))

            print(f"[UncropByMaskEQX] Fit mode: scaled to {new_width}x{new_height}, pasted at ({paste_x}, {paste_y})")
            return new_image

        elif mode == "crop_center":
            # Scale to fill and crop center
            scale_x = target_width / image.width
            scale_y = target_height / image.height
            scale = max(scale_x, scale_y)

            # Resize image
            new_width = int(image.width * scale)
            new_height = int(image.height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Center crop to target dimensions
            left = (new_width - target_width) // 2
            top = (new_height - target_height) // 2
            right = left + target_width
            bottom = top + target_height
            return image.crop((left, top, right, bottom))

        return image

    def apply_feather(self, mask: Image.Image, feather_amount: int) -> Image.Image:
        """Apply feathering (soft edges) to mask."""
        if feather_amount <= 0:
            return mask

        # Apply Gaussian blur for feathering
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_amount))

        # Adjust levels to maintain overall mask shape
        # This prevents the mask from becoming too transparent
        mask_np = np.array(mask).astype(np.float32) / 255.0
        mask_np = np.clip(mask_np * 1.2 - 0.1, 0, 1)  # Slight contrast adjustment

        return Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')

    def uncrop_by_mask(self, base_image: torch.Tensor, insert_image: torch.Tensor,
                      mask: torch.Tensor, resize_mode: str, mask_blur: int,
                      blend_opacity: float, feather_edges: int, invert_mask: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Composite insert_image onto base_image using mask.

        Args:
            base_image: Background image tensor
            insert_image: Image to insert tensor
            mask: Mask tensor (white = insert, black = keep base)
            resize_mode: How to resize insert image to match base
            mask_blur: Blur radius for mask edges
            blend_opacity: Opacity of the inserted image (1.0 = fully opaque)
            feather_edges: Amount of feathering for smooth edges

        Returns:
            Tuple of (composite_image, preview_mask)
        """

        print(f"[UncropByMaskEQX] Starting uncrop operation")
        print(f"[UncropByMaskEQX] Base image shape: {base_image.shape}")
        print(f"[UncropByMaskEQX] Insert image shape: {insert_image.shape}")
        print(f"[UncropByMaskEQX] Mask shape: {mask.shape}")
        print(f"[UncropByMaskEQX] Mask min: {mask.min()}, max: {mask.max()}")
        print(f"[UncropByMaskEQX] Resize mode: {resize_mode}, Opacity: {blend_opacity}")

        # Convert to PIL for processing
        base_pil = self.tensor_to_pil(base_image)
        insert_pil = self.tensor_to_pil(insert_image)
        mask_pil = self.mask_to_pil(mask)

        print(f"[UncropByMaskEQX] PIL sizes - Base: {base_pil.size}, Insert: {insert_pil.size}, Mask: {mask_pil.size}")

        # Debug: Check mask values
        mask_array = np.array(mask_pil)
        print(f"[UncropByMaskEQX] Mask array shape: {mask_array.shape}")
        print(f"[UncropByMaskEQX] Mask unique values: {np.unique(mask_array)[:10]}")  # Show first 10 unique values
        print(f"[UncropByMaskEQX] Mask white pixels: {np.sum(mask_array > 128)} / {mask_array.size}")

        # Ensure all images are in RGB mode for consistency
        if base_pil.mode != 'RGB':
            base_pil = base_pil.convert('RGB')
        if insert_pil.mode != 'RGB':
            insert_pil = insert_pil.convert('RGB')

        # Resize mask to match base image if needed
        if mask_pil.size != base_pil.size:
            print(f"[UncropByMaskEQX] WARNING: Mask size {mask_pil.size} doesn't match base {base_pil.size}")
            print(f"[UncropByMaskEQX] Resizing mask to match base image")
            mask_pil = mask_pil.resize(base_pil.size, Image.Resampling.NEAREST)  # Use NEAREST for masks

        # Handle resize mode
        if resize_mode == "source":
            # Special mode: Use insert image as-is from the source crop
            # This assumes insert_image is already the right crop from the original
            print(f"[UncropByMaskEQX] Source mode: Using insert image as direct replacement")

            # If sizes don't match exactly, we need to handle it
            if insert_pil.size != base_pil.size:
                # The insert image should be placed where the mask indicates
                # We'll resize the insert to match base for now
                print(f"[UncropByMaskEQX] Source mode: Resizing insert from {insert_pil.size} to {base_pil.size}")
                insert_pil = insert_pil.resize(base_pil.size, Image.Resampling.LANCZOS)
        else:
            # Other resize modes
            if insert_pil.size != base_pil.size:
                print(f"[UncropByMaskEQX] Resizing insert image using mode: {resize_mode}")
                insert_pil = self.resize_image_to_mask(insert_pil, base_pil.size, resize_mode)

        # Invert mask if requested
        if invert_mask:
            print(f"[UncropByMaskEQX] Inverting mask")
            mask_array = np.array(mask_pil)
            mask_array = 255 - mask_array  # Invert: white becomes black, black becomes white
            mask_pil = Image.fromarray(mask_array, mode='L')

        # Apply feathering to mask if requested
        if feather_edges > 0:
            print(f"[UncropByMaskEQX] Applying feather edges: {feather_edges}")
            mask_pil = self.apply_feather(mask_pil, feather_edges)

        # Apply blur to mask if requested
        if mask_blur > 0:
            print(f"[UncropByMaskEQX] Applying mask blur: {mask_blur}")
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=mask_blur))

        # Apply opacity to mask
        if blend_opacity < 1.0:
            mask_np = np.array(mask_pil).astype(np.float32) / 255.0
            mask_np = mask_np * blend_opacity
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')

        # Create composite using the mask
        # IMPORTANT: In PIL.composite, the mask works as follows:
        # composite(image1, image2, mask)
        # - Where mask is white (255) → show image1
        # - Where mask is black (0) → show image2
        #
        # We want:
        # - Where mask is white → insert the new image
        # - Where mask is black → keep the base image
        # So the base should be image2 and insert should be image1

        print(f"[UncropByMaskEQX] Creating composite...")
        print(f"[UncropByMaskEQX] Mask mode: {mask_pil.mode}, size: {mask_pil.size}")

        # Ensure mask is in L mode (grayscale)
        if mask_pil.mode != 'L':
            mask_pil = mask_pil.convert('L')

        # Debug: Check if images have content
        base_np = np.array(base_pil).astype(np.float32) / 255.0
        insert_np = np.array(insert_pil).astype(np.float32) / 255.0
        mask_np = np.array(mask_pil).astype(np.float32) / 255.0

        print(f"[UncropByMaskEQX] Base image - min: {base_np.min():.3f}, max: {base_np.max():.3f}, mean: {base_np.mean():.3f}")
        print(f"[UncropByMaskEQX] Insert image - min: {insert_np.min():.3f}, max: {insert_np.max():.3f}, mean: {insert_np.mean():.3f}")
        print(f"[UncropByMaskEQX] Mask - min: {mask_np.min():.3f}, max: {mask_np.max():.3f}, mean: {mask_np.mean():.3f}")

        # Check shapes
        print(f"[UncropByMaskEQX] Shapes - Base: {base_np.shape}, Insert: {insert_np.shape}, Mask: {mask_np.shape}")

        # Expand mask to 3 channels if needed
        if len(mask_np.shape) == 2:
            mask_np = np.expand_dims(mask_np, axis=-1)
            mask_np = np.repeat(mask_np, 3, axis=-1)

        # IMPORTANT: The composite formula
        # result = insert * mask + base * (1 - mask)
        # This means:
        # - Where mask is 1 (white), we get insert image
        # - Where mask is 0 (black), we get base image

        # Since your mask has white where you want to insert, this should work correctly
        composite_np = insert_np * mask_np + base_np * (1 - mask_np)

        # Debug composite result
        print(f"[UncropByMaskEQX] Composite - min: {composite_np.min():.3f}, max: {composite_np.max():.3f}, mean: {composite_np.mean():.3f}")

        # Convert back to PIL
        composite_np = np.clip(composite_np * 255, 0, 255).astype(np.uint8)
        composite = Image.fromarray(composite_np, mode='RGB')

        print(f"[UncropByMaskEQX] Composite created successfully using numpy blending")

        # Create preview mask (colored for visualization)
        preview_mask = mask_pil.convert('RGB')

        # Convert back to tensors
        composite_tensor = self.pil_to_tensor(composite)
        preview_mask_tensor = self.pil_to_tensor(preview_mask)

        print(f"[UncropByMaskEQX] Composite complete")
        print(f"[UncropByMaskEQX] Output shape: {composite_tensor.shape}")

        return (composite_tensor, preview_mask_tensor)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force re-execution when inputs change."""
        return float("nan")


# Node registration
NODE_CLASS_MAPPINGS = {
    "UncropByMaskEQX": UncropByMaskEQX
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UncropByMaskEQX": "Uncrop by Mask EQX"
}