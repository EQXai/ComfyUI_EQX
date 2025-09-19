import torch
import numpy as np
from PIL import Image
import hashlib
from typing import List, Tuple, Dict
from skimage.metrics import structural_similarity as ssim
import cv2

class ImageDuplicateRemover:
    """
    A ComfyUI node for detecting and removing duplicate or near-duplicate images from a batch.
    Uses multiple detection methods: dHash, pHash, histogram comparison, and optional SSIM.
    """

    NODE_NAME = "Image Duplicate Remover EQX"
    CATEGORY = "EQX/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "method": (["dhash", "phash", "histogram", "combined", "ssim"],),
                "threshold": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "keep_first": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "hash_size": ("INT", {
                    "default": 8,
                    "min": 4,
                    "max": 32,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("images", "removed_count", "report")
    FUNCTION = "remove_duplicates"

    def tensor_to_pil(self, tensor):
        """Convert tensor to PIL Image."""
        i = 255. * tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img

    def calculate_dhash(self, image, hash_size=8):
        """Calculate difference hash of an image."""
        img = image.convert('L').resize((hash_size + 1, hash_size), Image.LANCZOS)
        pixels = np.array(img)

        diff = pixels[:, 1:] > pixels[:, :-1]
        return diff.flatten()

    def calculate_phash(self, image, hash_size=8):
        """Calculate perceptual hash using DCT."""
        img = image.convert('L').resize((32, 32), Image.LANCZOS)
        pixels = np.array(img, dtype=np.float32)

        dct = cv2.dct(pixels)
        dct_low = dct[:hash_size, :hash_size]

        median = np.median(dct_low)
        diff = dct_low > median
        return diff.flatten()

    def calculate_histogram(self, image):
        """Calculate color histogram for comparison."""
        img_array = np.array(image)
        hist_r = np.histogram(img_array[:,:,0], bins=256, range=(0, 256))[0]
        hist_g = np.histogram(img_array[:,:,1], bins=256, range=(0, 256))[0]
        hist_b = np.histogram(img_array[:,:,2], bins=256, range=(0, 256))[0]

        hist_r = hist_r / hist_r.sum()
        hist_g = hist_g / hist_g.sum()
        hist_b = hist_b / hist_b.sum()

        return np.concatenate([hist_r, hist_g, hist_b])

    def hamming_distance(self, hash1, hash2):
        """Calculate normalized Hamming distance between two hashes."""
        if len(hash1) != len(hash2):
            return 1.0
        distance = np.sum(hash1 != hash2)
        return 1.0 - (distance / len(hash1))

    def histogram_similarity(self, hist1, hist2):
        """Calculate histogram correlation coefficient."""
        return np.corrcoef(hist1, hist2)[0, 1]

    def calculate_ssim(self, img1, img2):
        """Calculate SSIM between two images."""
        gray1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)

        min_dim = min(gray1.shape[0], gray1.shape[1], gray2.shape[0], gray2.shape[1])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)

        return ssim(gray1, gray2, win_size=win_size)

    def combined_similarity(self, img1, img2, hash_size=8):
        """Calculate combined similarity using multiple methods."""
        dhash1 = self.calculate_dhash(img1, hash_size)
        dhash2 = self.calculate_dhash(img2, hash_size)
        dhash_sim = self.hamming_distance(dhash1, dhash2)

        phash1 = self.calculate_phash(img1, hash_size)
        phash2 = self.calculate_phash(img2, hash_size)
        phash_sim = self.hamming_distance(phash1, phash2)

        hist1 = self.calculate_histogram(img1)
        hist2 = self.calculate_histogram(img2)
        hist_sim = self.histogram_similarity(hist1, hist2)

        return (dhash_sim * 0.4 + phash_sim * 0.4 + hist_sim * 0.2)

    def remove_duplicates(self, images, method="dhash", threshold=0.95, keep_first=True, hash_size=8):
        """Main function to remove duplicate images."""

        if images.shape[0] <= 1:
            return (images, 0, "No duplicates (only 1 image)")

        batch_size = images.shape[0]
        pil_images = []

        for i in range(batch_size):
            img_tensor = images[i]
            pil_img = self.tensor_to_pil(img_tensor)
            pil_images.append(pil_img)

        if method == "dhash":
            hashes = [self.calculate_dhash(img, hash_size) for img in pil_images]
            similarities = lambda i, j: self.hamming_distance(hashes[i], hashes[j])
        elif method == "phash":
            hashes = [self.calculate_phash(img, hash_size) for img in pil_images]
            similarities = lambda i, j: self.hamming_distance(hashes[i], hashes[j])
        elif method == "histogram":
            hists = [self.calculate_histogram(img) for img in pil_images]
            similarities = lambda i, j: self.histogram_similarity(hists[i], hists[j])
        elif method == "combined":
            similarities = lambda i, j: self.combined_similarity(pil_images[i], pil_images[j], hash_size)
        elif method == "ssim":
            similarities = lambda i, j: self.calculate_ssim(pil_images[i], pil_images[j])

        unique_indices = []
        duplicate_groups = []
        processed = set()

        for i in range(batch_size):
            if i in processed:
                continue

            group = [i]
            for j in range(i + 1, batch_size):
                if j in processed:
                    continue

                similarity = similarities(i, j)

                if similarity >= threshold:
                    group.append(j)
                    processed.add(j)

            if len(group) > 1:
                duplicate_groups.append(group)
                if keep_first:
                    unique_indices.append(group[0])
            else:
                unique_indices.append(i)

            processed.add(i)

        unique_images = images[unique_indices]
        removed_count = batch_size - len(unique_indices)

        report_lines = [
            f"Method: {method}",
            f"Threshold: {threshold:.2f}",
            f"Original images: {batch_size}",
            f"Unique images: {len(unique_indices)}",
            f"Duplicates removed: {removed_count}"
        ]

        if duplicate_groups:
            report_lines.append("\nDuplicate groups found:")
            for i, group in enumerate(duplicate_groups, 1):
                report_lines.append(f"  Group {i}: Images {', '.join(map(str, group))}")

        report = "\n".join(report_lines)

        return (unique_images, removed_count, report)

NODE_CLASS_MAPPINGS = {
    "ImageDuplicateRemover": ImageDuplicateRemover
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageDuplicateRemover": "Image Duplicate Remover EQX"
}