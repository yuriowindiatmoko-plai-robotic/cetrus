#!/usr/bin/env python3
"""Test script to verify the full inference pipeline"""

import sys
import os
sys.path.insert(0, '/home/yurio/Public/PROJECT/cetrus')

from streamlit_app.core.model_loader import ModelLoader
from streamlit_app.core.image_processor import ImageProcessor
from streamlit_app.core.inference_engine import InferenceEngine
import torch

def test_full_inference():
    """Test the full inference pipeline with the fixed image sizes"""

    # Load the problematic images
    lr_path = "/home/yurio/Public/PROJECT/cetrus/lr-tes.png"
    ref_path = "/home/yurio/Public/PROJECT/cetrus/ri-tes.png"

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
        print("Initializing components...")

        # Initialize components
        model_loader = ModelLoader()
        image_processor = ImageProcessor(scale_factor=4)
        inference_engine = InferenceEngine(model_loader, image_processor)

        # Create mock uploaded files
        lr_file = MockUploadedFile(lr_path)
        ref_file = MockUploadedFile(ref_path)

        print("Running inference pipeline...")

        # Run inference
        result = inference_engine.process_uploaded_images(
            lr_file=lr_file,
            ref_file=ref_file,
            model_type="mse",
            scale_factor=4
        )

        if result['success']:
            print("\n✅ Inference successful!")
            print(f"Output image shape: {result['sr_image'].shape}")
            print(f"Inference time: {result['inference_time']:.2f} seconds")
            print(f"Model type: {result['model_type']}")
            print(f"Device: {result['device']}")
            return True
        else:
            print(f"\n❌ Inference failed: {result['error']}")
            return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_inference()
    sys.exit(0 if success else 1)