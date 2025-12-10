"""Inference engine for DATSR Streamlit app"""

import torch
import numpy as np
import sys
import os
from typing import Dict, Any, Tuple, Optional
import time

# Add DATSR path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'DATSR'))

try:
    from datsr.utils.util import tensor2img
except ImportError:
    # Fallback implementation
    def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
        """Convert tensor to image"""
        tensor = tensor.squeeze(0).float().cpu().clamp_(*min_max)
        tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0, 1]
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)
        img_np = tensor.numpy()
        img_np = (img_np * 255.0).round()
        return img_np.astype(out_type)


class InferenceEngine:
    """Handle DATSR model inference"""

    def __init__(self, model_loader, image_processor):
        self.model_loader = model_loader
        self.image_processor = image_processor
        self.current_model = None
        self.current_model_type = None

    def run_inference(self, lr_tensor: torch.Tensor, ref_tensor: torch.Tensor,
                     lr_up_tensor: torch.Tensor, model_type: str = "mse",
                     progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Run DATSR inference on preprocessed tensors

        Args:
            lr_tensor: Low-resolution input tensor
            ref_tensor: Reference image tensor
            lr_up_tensor: Bicubic upsampled LR tensor
            model_type: Type of model to use ("mse" or "gan")
            progress_callback: Optional callback for progress updates

        Returns:
            Dict containing results and metadata
        """
        try:
            # Load model if not already loaded
            if self.current_model_type != model_type:
                if progress_callback:
                    progress_callback("Loading model...", 0.1)

                self.current_model = self.model_loader.load_model(model_type)
                self.current_model_type = model_type

            model = self.current_model

            # Move tensors to device
            device = self.model_loader.device
            lr_tensor = lr_tensor.to(device)
            ref_tensor = ref_tensor.to(device)
            lr_up_tensor = lr_up_tensor.to(device)

            if progress_callback:
                progress_callback("Running inference...", 0.3)

            # Run inference using proper DATSR pattern
            start_time = time.time()

            # Prepare data dictionary for DATSR model
            data = {
                'img_in_lq': lr_tensor,
                'img_ref': ref_tensor,
                'img_in_up': lr_up_tensor,
                'img_in': lr_up_tensor  # Add missing key - use bicubic upsampled as ground truth substitute
            }

            if progress_callback:
                progress_callback("Preparing model data...", 0.4)

            # Feed data to model and run inference
            model.feed_data(data)

            if progress_callback:
                progress_callback("Running inference...", 0.6)

            # Use DATSR's test() method for inference
            model.test()

            if progress_callback:
                progress_callback("Extracting results...", 0.8)

            # Get results from model
            visuals = model.get_current_visuals()
            output = visuals['rlt']

            inference_time = time.time() - start_time

            if progress_callback:
                progress_callback("Post-processing results...", 0.95)

            # Convert output to image
            sr_img = self._tensor_to_display_image(output)

            if progress_callback:
                progress_callback("Complete!", 1.0)

            return {
                'success': True,
                'sr_image': sr_img,
                'inference_time': inference_time,
                'model_type': model_type,
                'device': str(device),
                'output_shape': output.shape if hasattr(output, 'shape') else None
            }

        except Exception as e:
            import traceback
            error_msg = f"Inference failed: {str(e)}"
            print(error_msg)
            print("Full traceback:")
            traceback.print_exc()
            return {
                'success': False,
                'error': error_msg,
                'inference_time': 0,
                'model_type': model_type
            }

    def _tensor_to_display_image(self, tensor) -> np.ndarray:
        """Convert model output tensor to displayable image"""
        try:
            # Use DATSR's tensor2img if available
            img = tensor2img(tensor, out_type=np.uint8, min_max=(0, 1))

            # Convert RGB to BGR for OpenCV display
            if len(img.shape) == 3 and img.shape[2] == 3:
                import cv2
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            return img

        except Exception as e:
            print(f"Error converting tensor to image: {e}")
            # Fallback conversion
            return self._fallback_tensor_conversion(tensor)

    def _fallback_tensor_conversion(self, tensor) -> np.ndarray:
        """Fallback method to convert tensor to image"""
        try:
            # Move to CPU and convert to numpy
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)

            img = tensor.detach().cpu().numpy()

            # CHW to HWC
            if len(img.shape) == 3:
                img = img.transpose(1, 2, 0)

            # Normalize to [0, 255]
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)

            return img

        except Exception as e:
            print(f"Fallback conversion also failed: {e}")
            # Return a black image as last resort
            return np.zeros((256, 256, 3), dtype=np.uint8)

    def process_uploaded_images(self, lr_file, ref_file, model_type: str = "mse",
                               scale_factor: int = 4, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Complete pipeline: process uploaded images and run inference

        Args:
            lr_file: Low-resolution image file
            ref_file: Reference image file
            model_type: Type of model to use
            scale_factor: Super-resolution scale factor
            progress_callback: Progress callback function

        Returns:
            Dict containing all results and intermediate images
        """
        try:
            # Update image processor scale factor
            self.image_processor.scale_factor = scale_factor

            if progress_callback:
                progress_callback("Validating images...", 0.05)

            # Validate uploaded files
            lr_valid, lr_msg = self.image_processor.validate_image_file(lr_file)
            ref_valid, ref_msg = self.image_processor.validate_image_file(ref_file)

            if not lr_valid:
                return {'success': False, 'error': f"LR image: {lr_msg}"}
            if not ref_valid:
                return {'success': False, 'error': f"Reference image: {ref_msg}"}

            if progress_callback:
                progress_callback("Preprocessing images...", 0.15)

            # Preprocess images
            processed = self.image_processor.preprocess_uploaded_images(lr_file, ref_file)

            # Extract original LR image for display
            lr_display = self.image_processor.tensor_to_image(processed['lr_tensor'])

            # Extract reference image for display
            ref_display = self.image_processor.tensor_to_image(processed['ref_tensor'])

            # Extract bicubic upsampled version for comparison
            bicubic_display = self.image_processor.tensor_to_image(processed['lr_up_tensor'])

            if progress_callback:
                progress_callback("Running super-resolution...", 0.2)

            # Run inference
            inference_result = self.run_inference(
                processed['lr_tensor'],
                processed['ref_tensor'],
                processed['lr_up_tensor'],
                model_type,
                progress_callback
            )

            if not inference_result['success']:
                return inference_result

            # Combine all results
            result = {
                'success': True,
                'sr_image': inference_result['sr_image'],
                'lr_image': lr_display,
                'ref_image': ref_display,
                'bicubic_image': bicubic_display,
                'inference_time': inference_result['inference_time'],
                'model_type': model_type,
                'scale_factor': scale_factor,
                'original_size': processed['original_size'],
                'processed_size': processed['processed_size'],
                'device': inference_result['device']
            }

            return result

        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            print(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'inference_time': 0
            }

    def get_inference_info(self) -> Dict[str, Any]:
        """Get information about the inference engine"""
        info = {
            'model_loaded': self.current_model is not None,
            'current_model_type': self.current_model_type,
            'device': self.model_loader.device
        }

        if self.current_model:
            model_info = self.model_loader.get_model_info(self.current_model_type)
            info.update(model_info)

        return info