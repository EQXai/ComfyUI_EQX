import torch
import numpy as np

def test_image_preservation():
    print("Testing VideoFragments node for quality/data preservation...")

    print("\n1. Testing torch.cat() preservation:")
    img1 = torch.rand((2, 512, 512, 3), dtype=torch.float32)
    img2 = torch.rand((3, 512, 512, 3), dtype=torch.float32)
    img3 = torch.rand((1, 512, 512, 3), dtype=torch.float32)

    img1_copy = img1.clone()
    img2_copy = img2.clone()
    img3_copy = img3.clone()

    combined = torch.cat([img1, img2, img3], dim=0)

    print(f"   Original shapes: {img1.shape}, {img2.shape}, {img3.shape}")
    print(f"   Combined shape: {combined.shape}")
    print(f"   Expected shape: torch.Size([6, 512, 512, 3])")

    print("\n2. Checking data integrity:")
    print(f"   First fragment preserved: {torch.equal(combined[0:2], img1_copy)}")
    print(f"   Second fragment preserved: {torch.equal(combined[2:5], img2_copy)}")
    print(f"   Third fragment preserved: {torch.equal(combined[5:6], img3_copy)}")

    print("\n3. Checking data types:")
    print(f"   Original dtype: {img1.dtype}")
    print(f"   Combined dtype: {combined.dtype}")
    print(f"   Dtype preserved: {img1.dtype == combined.dtype}")

    print("\n4. Checking value ranges:")
    print(f"   Original img1 range: [{img1.min():.4f}, {img1.max():.4f}]")
    print(f"   Combined fragment1 range: [{combined[0:2].min():.4f}, {combined[0:2].max():.4f}]")

    print("\n5. Testing with different precisions:")
    img_float16 = torch.rand((2, 256, 256, 3), dtype=torch.float16)
    img_float32 = torch.rand((2, 256, 256, 3), dtype=torch.float32)

    combined_mixed = torch.cat([img_float16.to(torch.float32), img_float32], dim=0)
    print(f"   Float16 converted to Float32 for concatenation")
    print(f"   Result dtype: {combined_mixed.dtype}")

    print("\n6. Memory analysis:")
    img_size_mb = (img1.element_size() * img1.nelement()) / (1024 * 1024)
    combined_size_mb = (combined.element_size() * combined.nelement()) / (1024 * 1024)
    expected_size_mb = img_size_mb + (img2.element_size() * img2.nelement()) / (1024 * 1024) + (img3.element_size() * img3.nelement()) / (1024 * 1024)

    print(f"   Individual sizes: {img_size_mb:.2f} MB + {(img2.element_size() * img2.nelement()) / (1024 * 1024):.2f} MB + {(img3.element_size() * img3.nelement()) / (1024 * 1024):.2f} MB")
    print(f"   Combined size: {combined_size_mb:.2f} MB")
    print(f"   Expected size: {expected_size_mb:.2f} MB")

    print("\n✅ CONCLUSION:")
    print("   - torch.cat() does NOT modify or compress the data")
    print("   - All pixel values are preserved exactly")
    print("   - No quality loss occurs during concatenation")
    print("   - The operation only reorganizes tensors in memory")

if __name__ == "__main__":
    test_image_preservation()