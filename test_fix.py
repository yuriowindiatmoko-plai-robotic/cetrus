#!/usr/bin/env python3
"""Test script to verify the image preprocessing fix"""

import sys
import os
sys.path.insert(0, '/home/yurio/Public/PROJECT/cetrus')

from streamlit_app.core.image_processor import ImageProcessor
import cv2
import numpy as np

def test_image_preprocessing():
    """Test the image preprocessing with problematic images"""

    # Load the problematic images
    lr_path = "/home/yurio/Public/PROJECT/cetrus/lr-tes.png"
    ref_path = "/home/yurio/Public/PROJECT/cetrus/ri-tes.png"

    if not os.path.exists(lr_path) or not os.path.exists(ref_path):
        print(f"Images not found: {lr_path}, {ref_path}")
        # Create dummy images if they don't exist
        lr_img = np.random.randint(0, 255, (303, 285, 3), dtype=np.uint8)
        ref_img = np.random.randint(0, 255, (303, 285, 3), dtype=np.uint8)
        cv2.imwrite(lr_path, lr_img)
        cv2.imwrite(ref_path, ref_img)
        print("Created dummy test images")

    # Create a mock uploaded file class
    class MockUploadedFile:
        def __init__(self, path):
            self.path = path
            with open(path, 'rb') as f:
                self.bytes = f.read()
            self.name = os.path.basename(path)
            self.size = len(self.bytes)

        def read(self):
            return self.bytes

        def seek(self, position):
            pass

    try:
        # Create image processor
        processor = ImageProcessor(scale_factor=4)

        # Create mock uploaded files
        lr_file = MockUploadedFile(lr_path)
        ref_file = MockUploadedFile(ref_path)

        # Process images
        print("Processing images...")
        result = processor.preprocess_uploaded_images(lr_file, ref_file)

        print("\n✅ Processing successful!")
        print(f"Original size: {result['original_size']}")
        print(f"Processed size: {result['processed_size']}")
        print(f"LR size: {result['lr_size']}")
        print(f"Scale factor: {result['scale_factor']}")
        print(f"Padding applied: {result['padding']}")

        # Check tensor shapes
        print(f"\nLR tensor shape: {result['lr_tensor'].shape}")
        print(f"Ref tensor shape: {result['ref_tensor'].shape}")
        print(f"LR up tensor shape: {result['lr_up_tensor'].shape}")

        # Verify expected shapes (should be 640x640 HR -> 160x160 LR)
        expected_lr = 160
        expected_hr = 640

        assert result['lr_tensor'].shape == (1, 3, expected_lr, expected_lr), \
            f"LR tensor shape mismatch: expected (1, 3, {expected_lr}, {expected_lr}), got {result['lr_tensor'].shape}"
        assert result['ref_tensor'].shape == (1, 3, expected_lr, expected_lr), \
            f"Ref tensor shape mismatch: expected (1, 3, {expected_lr}, {expected_lr}), got {result['ref_tensor'].shape}"
        assert result['lr_up_tensor'].shape == (1, 3, expected_hr, expected_hr), \
            f"LR up tensor shape mismatch: expected (1, 3, {expected_hr}, {expected_hr}), got {result['lr_up_tensor'].shape}"

        print("\n✅ All tensor shapes are correct!")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_image_preprocessing()
    sys.exit(0 if success else 1)