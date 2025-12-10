#!/usr/bin/env python3
"""Test DATSR inference with custom images"""

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
import glob

# Add DATSR to path
sys.path.insert(0, os.path.dirname(__file__))

from datsr.models.ref_restoration_model import RefRestorationModel
from datsr.data.transforms import mod_crop
import mmcv

def create_test_data(lr_path, ref_path, output_dir="test_output"):
    """Create test data structure"""
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories
    lr_subdir = os.path.join(output_dir, "lq")
    ref_subdir = os.path.join(output_dir, "ref")
    os.makedirs(lr_subdir, exist_ok=True)
    os.makedirs(ref_subdir, exist_ok=True)

    # Copy and process images
    lr_img = cv2.imread(lr_path)
    ref_img = cv2.imread(ref_path)

    if lr_img is None or ref_img is None:
        raise ValueError("Could not load test images")

    print(f"Original LR size: {lr_img.shape[:2]}")
    print(f"Original Ref size: {ref_img.shape[:2]}")

    # CRITICAL FIX: Resize to training configuration
    # DATSR was trained with gt_size=160, which means HR=640x640
    scale_factor = 4
    TARGET_LR_SIZE = 160
    TARGET_HR_SIZE = TARGET_LR_SIZE * scale_factor  # 640x640

    # Resize to training dimensions
    lr_resized = cv2.resize(lr_img, (TARGET_HR_SIZE, TARGET_HR_SIZE))
    ref_resized = cv2.resize(ref_img, (TARGET_HR_SIZE, TARGET_HR_SIZE))

    print(f"Resized to training size: HR ({TARGET_HR_SIZE}, {TARGET_HR_SIZE})")

    # Apply mod_crop for scale compatibility
    lr_resized = mod_crop(lr_resized, scale_factor)
    ref_resized = mod_crop(ref_resized, scale_factor)

    # Create LR version by downsampling
    lr_pil = Image.fromarray(cv2.cvtColor(lr_resized, cv2.COLOR_BGR2RGB))
    lr_lr_pil = lr_pil.resize((TARGET_LR_SIZE, TARGET_LR_SIZE), Image.BICUBIC)
    lr_lr = cv2.cvtColor(np.array(lr_lr_pil), cv2.COLOR_RGB2BGR)

    ref_pil = Image.fromarray(cv2.cvtColor(ref_resized, cv2.COLOR_BGR2RGB))
    ref_lr_pil = ref_pil.resize((TARGET_LR_SIZE, TARGET_LR_SIZE), Image.BICUBIC)
    ref_lr = cv2.cvtColor(np.array(ref_lr_pil), cv2.COLOR_RGB2BGR)

    print(f"Final LR dimensions: {lr_lr.shape[:2]}")

    # Save processed images
    lr_out_path = os.path.join(lr_subdir, "test_lr.png")
    ref_out_path = os.path.join(ref_subdir, "test_ref.png")

    cv2.imwrite(lr_out_path, lr_lr)
    cv2.imwrite(ref_out_path, ref_lr)

    return output_dir, "test_lr", "test_ref"

def test_inference():
    """Test inference with MSE model"""

    # Check if test images exist
    lr_path = "lr-tes.png"
    ref_path = "ri-tes.png"

    if not os.path.exists(lr_path) or not os.path.exists(ref_path):
        print("Test images not found. Creating dummy images...")
        # Create dummy test images
        dummy_img = np.random.randint(0, 255, (303, 285, 3), dtype=np.uint8)
        cv2.imwrite(lr_path, dummy_img)
        cv2.imwrite(ref_path, dummy_img)

    try:
        # Create test data
        test_dir, lr_name, ref_name = create_test_data(lr_path, ref_path)
        print(f"\nCreated test data in: {test_dir}")

        # Create minimal options
        test_opt = {
            'name': 'test_restoration_mse',
            'model': 'ref_restoration',
            'scale': 4,
            'gpu_ids': [0] if torch.cuda.is_available() else [],
            'path': {
                'pretrain_model_g': 'experiments/pretrained_model/restoration_mse.pth',
                'pretrain_model_feature_extractor': 'experiments/pretrained_model/feature_extraction.pth',
                'strict_load': True
            },
            'datasets': {
                'test': {
                    'name': 'test_dataset',
                    'dataroot_LQ': os.path.join(test_dir, 'lq'),
                    'dataroot_ref': os.path.join(test_dir, 'ref'),
                    'io_backend': 'disk',
                    'scale': 4
                }
            },
            'network_g': {
                'type': 'SwinUnetv3RestorationNet',
                'upscale': 4,
                'in_chans': 3,
                'img_size': 64,
                'window_size': 8,
                'img_range': 1.,
                'depths': [6, 6, 6, 6, 6, 6],
                'embed_dim': 180,
                'num_heads': [6, 6, 6, 6, 6, 6],
                'mlp_ratio': 2,
                'upsampler': 'nearest+conv',
                'resi_connection': '3conv'
            },
            'network_d': None,
            'is_train': False,
            'dist': False
        }

        # Create model
        print("\nLoading model...")
        model = RefRestorationModel(test_opt)

        # Load test data manually
        lr_path = os.path.join(test_dir, 'lq', f'{lr_name}.png')
        ref_path = os.path.join(test_dir, 'ref', f'{ref_name}.png')

        # Load images
        img_lq = mmcv.imread(lr_path, channel_order='BGR').astype(np.float32) / 255.
        img_ref = mmcv.imread(ref_path, channel_order='BGR').astype(np.float32) / 255.

        # Create test data dictionary
        test_data = {
            'img_in_lq': torch.from_numpy(img_lq).unsqueeze(0).permute(0, 3, 1, 2).contiguous(),
            'img_ref': torch.from_numpy(img_ref).unsqueeze(0).permute(0, 3, 1, 2).contiguous(),
            'img_in_up': torch.from_numpy(img_lq).unsqueeze(0).permute(0, 3, 1, 2).contiguous(),  # Use LR as placeholder
            'img_in': torch.from_numpy(img_lq).unsqueeze(0).permute(0, 3, 1, 2).contiguous()  # Use LR as placeholder
        }

        print("\nRunning inference...")
        model.feed_data(test_data)
        model.test()

        # Get results
        visuals = model.get_current_visuals()
        output_img = visuals['rlt']

        # Convert to numpy
        if output_img.dim() == 4:
            output_img = output_img.squeeze(0)

        output_np = output_img.detach().cpu().numpy()
        output_np = output_np.transpose(1, 2, 0)
        output_np = (output_np * 255.0).round().astype(np.uint8)
        output_np = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

        # Save output
        output_path = "test_output_result.png"
        cv2.imwrite(output_path, output_np)

        print(f"\n✅ Inference successful!")
        print(f"Output saved to: {output_path}")
        print(f"Output shape: {output_np.shape}")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_inference()
    sys.exit(0 if success else 1)