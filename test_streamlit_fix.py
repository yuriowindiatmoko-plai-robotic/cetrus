#!/usr/bin/env python3
"""Test script to verify Streamlit app with fixed image processing"""

import sys
import os
sys.path.insert(0, '/home/yurio/Public/PROJECT/cetrus')

# Set environment for DATSR
os.environ['PYTHONPATH'] = f"/home/yurio/Public/PROJECT/cetrus:{os.environ.get('PYTHONPATH', '')}"

# Now import after setting PYTHONPATH
import streamlit as st
from streamlit_app.core.model_loader import ModelLoader
from streamlit_app.core.image_processor import ImageProcessor
from streamlit_app.core.inference_engine import InferenceEngine
import numpy as np
import cv2

def create_mock_images():
    """Create mock test images"""
    os.makedirs('/tmp/test_images', exist_ok=True)

    # Create test images
    lr_img = np.random.randint(0, 255, (303, 285, 3), dtype=np.uint8)
    ref_img = np.random.randint(0, 255, (303, 285, 3), dtype=np.uint8)

    lr_path = '/tmp/test_images/lr_test.png'
    ref_path = '/tmp/test_images/ref_test.png'

    cv2.imwrite(lr_path, lr_img)
    cv2.imwrite(ref_path, ref_img)

    return lr_path, ref_path

class MockUploadedFile:
    """Mock Streamlit uploaded file"""
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

def test_streamlit_pipeline():
    """Test the complete Streamlit pipeline"""

    print("Creating test images...")
    lr_path, ref_path = create_mock_images()

    try:
        print("\nInitializing Streamlit components...")

        # Initialize components
        model_loader = ModelLoader()
        image_processor = ImageProcessor(scale_factor=4)
        inference_engine = InferenceEngine(model_loader, image_processor)

        # Create mock uploaded files
        lr_file = MockUploadedFile(lr_path)
        ref_file = MockUploadedFile(ref_path)

        print("\nProcessing images with fixed preprocessing...")

        # Test just the preprocessing
        processed = image_processor.preprocess_uploaded_images(lr_file, ref_file)

        print(f"\n✅ Image processing successful!")
        print(f"  - Original size: {processed['original_size']}")
        print(f"  - Processed size: {processed['processed_size']}")
        print(f"  - LR size: {processed['lr_size']}")
        print(f"  - Tensor shapes: LR={processed['lr_tensor'].shape}, Ref={processed['ref_tensor'].shape}")

        # Verify the tensors are the expected size (640x640 HR, 160x160 LR)
        assert processed['lr_tensor'].shape == (1, 3, 160, 160), f"LR tensor wrong shape: {processed['lr_tensor'].shape}"
        assert processed['ref_tensor'].shape == (1, 3, 160, 160), f"Ref tensor wrong shape: {processed['ref_tensor'].shape}"
        assert processed['lr_up_tensor'].shape == (1, 3, 640, 640), f"LR up tensor wrong shape: {processed['lr_up_tensor'].shape}"

        print("\n✅ All tensor shapes are correct for DATSR!")

        # Clean up
        import shutil
        shutil.rmtree('/tmp/test_images')

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("Testing DATSR Streamlit App Fix")
    print("="*60)

    success = test_streamlit_pipeline()

    print("\n" + "="*60)
    if success:
        print("✅ Fix verified! Images are now resized to 640x640 HR (160x160 LR)")
        print("   This matches the training configuration and should prevent")
        print("   flow_warp assertion errors in DATSR.")
    else:
        print("❌ Test failed. Please check the error above.")
    print("="*60)

    sys.exit(0 if success else 1)