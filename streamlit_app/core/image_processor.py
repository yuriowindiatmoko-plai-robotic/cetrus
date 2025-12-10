"""Image processing pipeline for DATSR Streamlit app"""

import cv2
import numpy as np
import torch
from PIL import Image
import mmcv
import sys
import os

# Add DATSR path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'DATSR'))

try:
    from datsr.data.transforms import mod_crop
except ImportError:
    # Fallback implementation if import fails
    def mod_crop(img, scale):
        """Mod crop to ensure dimensions are divisible by scale"""
        h, w = img.shape[:2]
        h = h - h % scale
        w = w - w % scale
        return img[:h, :w, :]


class ImageProcessorRobust:
    """Handle image preprocessing for DATSR inference with robust size handling"""

    # Minimum dimensions based on model architecture requirements
    MIN_LR_SIZE = 48  # Minimum LR image dimension
    MIN_HR_SIZE = 192  # Minimum HR image dimension (MIN_LR_SIZE * scale_factor)

    # VGG feature map minimum sizes
    MIN_FEATURE2_SIZE = 32  # For relu2_1 feature maps
    MIN_FEATURE3_SIZE = 16  # For relu3_1 feature maps

    def __init__(self, scale_factor=4):
        self.scale_factor = scale_factor

    def preprocess_uploaded_images(self, lr_image_file, ref_image_file):
        """
        Preprocess uploaded images for DATSR inference with robust size handling

        Args:
            lr_image_file: Streamlit uploaded file for low-resolution input
            ref_image_file: Streamlit uploaded file for reference

        Returns:
            Dict containing processed tensors and metadata
        """
        try:
            # Read uploaded files
            img_in = self._read_uploaded_file(lr_image_file)
            img_ref = self._read_uploaded_file(ref_image_file)

            # Validate images
            if img_in is None or img_ref is None:
                raise ValueError("Failed to read uploaded images")

            return self._process_image_pair_robust(img_in, img_ref)

        except Exception as e:
            raise ValueError(f"Image preprocessing failed: {str(e)}")

    def _read_uploaded_file(self, uploaded_file):
        """Read Streamlit uploaded file to numpy array"""
        try:
            # Read file bytes
            file_bytes = uploaded_file.read()

            # Convert to numpy array using mmcv (same as DATSR)
            img = mmcv.imfrombytes(file_bytes).astype(np.float32) / 255.

            return img

        except Exception as e:
            print(f"Error reading uploaded file: {e}")
            return None

    def _process_image_pair_robust(self, img_in, img_ref):
        """
        Process a pair of images according to DATSR pipeline with robust size handling

        Args:
            img_in: Input image (BGR, float32, [0,1])
            img_ref: Reference image (BGR, float32, [0,1])

        Returns:
            Dict with processed tensors
        """
        # Store original dimensions
        original_h_in, original_w_in = img_in.shape[:2]
        original_h_ref, original_w_ref = img_ref.shape[:2]

        # Apply mod_crop for scale compatibility
        img_in = mod_crop(img_in, self.scale_factor)
        img_ref = mod_crop(img_ref, self.scale_factor)

        # Get processed dimensions
        gt_h, gt_w, _ = img_in.shape
        img_ref_h, img_ref_w, _ = img_ref.shape

        # Calculate minimum required dimensions
        min_gt_h = max(gt_h, self.MIN_HR_SIZE)
        min_gt_w = max(gt_w, self.MIN_HR_SIZE)

        # Ensure dimensions are divisible by scale_factor and window_size (8)
        window_size = 8
        min_gt_h = ((min_gt_h + max(self.scale_factor, window_size) - 1) // max(self.scale_factor, window_size)) * max(self.scale_factor, window_size)
        min_gt_w = ((min_gt_w + max(self.scale_factor, window_size) - 1) // max(self.scale_factor, window_size)) * max(self.scale_factor, window_size)

        # Pad images to same size and minimum dimensions
        padding = False
        target_h = max(min_gt_h, gt_h, img_ref_h)
        target_w = max(min_gt_w, gt_w, img_ref_w)

        # Ensure target dimensions are compatible with model architecture
        # Must be divisible by scale_factor for proper upsampling
        target_h = (target_h // self.scale_factor) * self.scale_factor
        target_w = (target_w // self.scale_factor) * self.scale_factor

        # Must be divisible by window_size for Swin Transformer
        target_h = ((target_h + window_size - 1) // window_size) * window_size
        target_w = ((target_w + window_size - 1) // window_size) * window_size

        if img_in.shape[0] != target_h or img_in.shape[1] != target_w:
            padding = True
            img_in = mmcv.impad(img_in, shape=(target_h, target_w), pad_val=0)

        if img_ref.shape[0] != target_h or img_ref.shape[1] != target_w:
            padding = True
            img_ref = mmcv.impad(img_ref, shape=(target_h, target_w), pad_val=0)

        # Update gt dimensions after padding
        gt_h, gt_w = target_h, target_w

        # Verify LR dimensions will be sufficient for feature extraction
        lq_h, lq_w = gt_h // self.scale_factor, gt_w // self.scale_factor

        # Ensure LR dimensions are large enough for VGG feature extraction
        if lq_h < self.MIN_FEATURE3_SIZE * 4 or lq_w < self.MIN_FEATURE3_SIZE * 4:
            # Increase dimensions to ensure minimum feature map sizes
            min_lq_h = max(lq_h, self.MIN_FEATURE3_SIZE * 4)
            min_lq_w = max(lq_w, self.MIN_FEATURE3_SIZE * 4)

            # Make divisible by window_size
            min_lq_h = ((min_lq_h + window_size - 1) // window_size) * window_size
            min_lq_w = ((min_lq_w + window_size - 1) // window_size) * window_size

            # Recalculate HR dimensions
            gt_h = min_lq_h * self.scale_factor
            gt_w = min_lq_w * self.scale_factor

            # Re-pad images if needed
            img_in = mmcv.impad(img_in, shape=(gt_h, gt_w), pad_val=0)
            img_ref = mmcv.impad(img_ref, shape=(gt_h, gt_w), pad_val=0)

            padding = True
            lq_h, lq_w = min_lq_h, min_lq_w

        print(f"Final image dimensions - HR: ({gt_h}, {gt_w}), LR: ({lq_h}, {lq_w})")

        # Create LR versions by downsampling
        # Convert to PIL for bicubic interpolation
        img_in_pil = self._bgr_to_pil(img_in)
        img_ref_pil = self._bgr_to_pil(img_ref)

        # Generate LR and upsampled versions
        img_in_lq = img_in_pil.resize((lq_w, lq_h), Image.BICUBIC)
        img_ref_lq = img_ref_pil.resize((lq_w, lq_h), Image.BICUBIC)

        img_in_up = img_in_lq.resize((gt_w, gt_h), Image.BICUBIC)
        img_ref_up = img_ref_lq.resize((gt_w, gt_h), Image.BICUBIC)

        # Convert back to BGR numpy arrays
        img_in_lq = self._pil_to_bgr(img_in_lq)
        img_in_up = self._pil_to_bgr(img_in_up)
        img_ref_lq = self._pil_to_bgr(img_ref_lq)
        img_ref_up = self._pil_to_bgr(img_ref_up)

        # Convert to tensors (BGR to RGB, HWC to CHW)
        lr_tensor = self._to_tensor(img_in_lq)
        ref_tensor = self._to_tensor(img_ref_lq)
        lr_up_tensor = self._to_tensor(img_in_up)

        # Add batch dimension
        lr_tensor = lr_tensor.unsqueeze(0)
        ref_tensor = ref_tensor.unsqueeze(0)
        lr_up_tensor = lr_up_tensor.unsqueeze(0)

        return {
            'lr_tensor': lr_tensor,
            'ref_tensor': ref_tensor,
            'lr_up_tensor': lr_up_tensor,
            'original_size': (original_h_in, original_w_in),
            'processed_size': (gt_h, gt_w),
            'padding': padding,
            'scale_factor': self.scale_factor,
            'lr_size': (lq_h, lq_w)
        }

    def _bgr_to_pil(self, img):
        """Convert BGR numpy array to PIL RGB Image"""
        img_rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

    def _pil_to_bgr(self, pil_img):
        """Convert PIL RGB Image to BGR numpy array"""
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img_bgr.astype(np.float32) / 255.

    def _to_tensor(self, img):
        """Convert BGR HWC image to RGB CHW tensor"""
        # BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # HWC to CHW
        img_chw = img_rgb.transpose(2, 0, 1)
        # Convert to tensor
        tensor = torch.from_numpy(img_chw).float()
        return tensor

    def tensor_to_image(self, tensor):
        """Convert tensor back to displayable image"""
        # Remove batch dimension if present
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # Convert to numpy
        img = tensor.detach().cpu().numpy()

        # CHW to HWC
        img = img.transpose(1, 2, 0)

        # Clamp values to [0, 1]
        img = np.clip(img, 0, 1)

        # Convert to uint8
        img = (img * 255).astype(np.uint8)

        # RGB to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    def validate_image_file(self, uploaded_file):
        """Validate uploaded image file"""
        if uploaded_file is None:
            return False, "No file uploaded"

        # Check file size
        if hasattr(uploaded_file, 'size') and uploaded_file.size > 50 * 1024 * 1024:
            return False, "File too large (max 50MB)"

        # Check file extension
        allowed_extensions = ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp']
        if not any(uploaded_file.name.lower().endswith(ext) for ext in allowed_extensions):
            return False, f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"

        return True, "Valid image file"

    def get_min_image_requirements(self):
        """Get minimum image size requirements for the model"""
        return {
            'min_lr_size': (self.MIN_LR_SIZE, self.MIN_LR_SIZE),
            'min_hr_size': (self.MIN_HR_SIZE, self.MIN_HR_SIZE),
            'scale_factor': self.scale_factor,
            'window_size': 8,
            'note': 'Images will be automatically padded to meet minimum requirements'
        }


class ImageProcessor:
    """Handle image preprocessing for DATSR inference"""

    def __init__(self, scale_factor=4):
        self.scale_factor = scale_factor

    def preprocess_uploaded_images(self, lr_image_file, ref_image_file):
        """
        Preprocess uploaded images for DATSR inference

        Args:
            lr_image_file: Streamlit uploaded file for low-resolution input
            ref_image_file: Streamlit uploaded file for reference

        Returns:
            Dict containing processed tensors and metadata
        """
        try:
            # Read uploaded files
            img_in = self._read_uploaded_file(lr_image_file)
            img_ref = self._read_uploaded_file(ref_image_file)

            # Validate images
            if img_in is None or img_ref is None:
                raise ValueError("Failed to read uploaded images")

            return self._process_image_pair(img_in, img_ref)

        except Exception as e:
            raise ValueError(f"Image preprocessing failed: {str(e)}")

    def _read_uploaded_file(self, uploaded_file):
        """Read Streamlit uploaded file to numpy array"""
        try:
            # Read file bytes
            file_bytes = uploaded_file.read()

            # Convert to numpy array using mmcv (same as DATSR)
            img = mmcv.imfrombytes(file_bytes).astype(np.float32) / 255.

            return img

        except Exception as e:
            print(f"Error reading uploaded file: {e}")
            return None

    def _process_image_pair(self, img_in, img_ref):
        """
        Process a pair of images according to DATSR pipeline with robust size handling

        Args:
            img_in: Input image (BGR, float32, [0,1])
            img_ref: Reference image (BGR, float32, [0,1])

        Returns:
            Dict with processed tensors
        """
        # Store original dimensions BEFORE any processing
        original_h_in, original_w_in = img_in.shape[:2]
        original_h_ref, original_w_ref = img_ref.shape[:2]

        print(f"\n=== Image Processing Debug ===")
        print(f"Original input size: ({original_h_in}, {original_w_in})")
        print(f"Original ref size: ({original_h_ref}, {original_w_ref})")

        # CRITICAL FIX: Check if images will produce LR smaller than minimum
        # Calculate what LR size would be after processing
        mod_cropped_h_in = original_h_in - (original_h_in % self.scale_factor)
        mod_cropped_w_in = original_w_in - (original_w_in % self.scale_factor)
        mod_cropped_h_ref = original_h_ref - (original_h_ref % self.scale_factor)
        mod_cropped_w_ref = original_w_ref - (original_w_ref % self.scale_factor)

        # Use max of both images
        max_gt_h = max(mod_cropped_h_in, mod_cropped_h_ref)
        max_gt_w = max(mod_cropped_w_in, mod_cropped_w_ref)

        # Calculate expected LR dimensions
        expected_lr_h = max_gt_h // self.scale_factor
        expected_lr_w = max_gt_w // self.scale_factor

        print(f"Expected LR after processing: ({expected_lr_h}, {expected_lr_w})")

        # Calculate optimal HR dimensions maintaining aspect ratio
        gt_h, gt_w = self.calculate_optimal_dimensions(original_h_in, original_w_in)

        print(f"Images: ({original_h_in}, {original_w_in}) -> Using HR ({gt_h}, {gt_w})")
        print(f"Corresponding LR dimensions: ({gt_h//self.scale_factor}, {gt_w//self.scale_factor})")

        # Resize images to optimal dimensions
        img_in = cv2.resize(img_in, (gt_w, gt_h))
        img_ref = cv2.resize(img_ref, (gt_w, gt_h))
        print(f"Resized to optimal size: HR ({gt_h}, {gt_w})")

        # Apply mod_crop for scale compatibility
        img_in = mod_crop(img_in, self.scale_factor)
        img_ref = mod_crop(img_ref, self.scale_factor)

        print(f"After mod_crop: input ({img_in.shape[0]}, {img_in.shape[1]}), ref ({img_ref.shape[0]}, {img_ref.shape[1]})")

        # Get processed dimensions
        gt_h, gt_w, _ = img_in.shape
        img_ref_h, img_ref_w, _ = img_ref.shape

        # Make both images the same size
        target_h = max(gt_h, img_ref_h)
        target_w = max(gt_w, img_ref_w)
        padding = False

        if img_in.shape[0] != target_h or img_in.shape[1] != target_w:
            img_in = mmcv.impad(img_in, shape=(target_h, target_w), pad_val=0)
            padding = True
        if img_ref.shape[0] != target_h or img_ref.shape[1] != target_w:
            img_ref = mmcv.impad(img_ref, shape=(target_h, target_w), pad_val=0)
            padding = True

        gt_h, gt_w = target_h, target_w

        print(f"Final HR dimensions: ({gt_h}, {gt_w})")

        # Calculate LR dimensions after downscaling
        lq_h, lq_w = gt_h // self.scale_factor, gt_w // self.scale_factor
        print(f"Final LR dimensions: ({lq_h}, {lq_w})")

        # Final verification - ensure LR is large enough for VGG feature extraction
        MIN_LR_FOR_VGG = 16  # Minimum size for VGG relu3_1 feature maps
        assert lq_h >= MIN_LR_FOR_VGG and lq_w >= MIN_LR_FOR_VGG, \
            f"LR dimensions ({lq_h}, {lq_w}) are still too small! Minimum required: ({MIN_LR_FOR_VGG}, {MIN_LR_FOR_VGG})"

        print(f"=== End Debug ===\n")

        # Convert to PIL for bicubic interpolation
        img_in_pil = self._bgr_to_pil(img_in)
        img_ref_pil = self._bgr_to_pil(img_ref)

        # Generate LR and upsampled versions
        img_in_lq = img_in_pil.resize((lq_w, lq_h), Image.BICUBIC)
        img_ref_lq = img_ref_pil.resize((lq_w, lq_h), Image.BICUBIC)

        img_in_up = img_in_lq.resize((gt_w, gt_h), Image.BICUBIC)
        img_ref_up = img_ref_lq.resize((gt_w, gt_h), Image.BICUBIC)

        # Convert back to BGR numpy arrays
        img_in_lq = self._pil_to_bgr(img_in_lq)
        img_in_up = self._pil_to_bgr(img_in_up)
        img_ref_lq = self._pil_to_bgr(img_ref_lq)
        img_ref_up = self._pil_to_bgr(img_ref_up)

        # Convert to tensors (BGR to RGB, HWC to CHW)
        lr_tensor = self._to_tensor(img_in_lq)
        ref_tensor = self._to_tensor(img_ref_lq)
        lr_up_tensor = self._to_tensor(img_in_up)

        # Add batch dimension
        lr_tensor = lr_tensor.unsqueeze(0)
        ref_tensor = ref_tensor.unsqueeze(0)
        lr_up_tensor = lr_up_tensor.unsqueeze(0)

        return {
            'lr_tensor': lr_tensor,
            'ref_tensor': ref_tensor,
            'lr_up_tensor': lr_up_tensor,
            'original_size': (original_h_in, original_w_in),
            'processed_size': (gt_h, gt_w),
            'padding': padding,
            'scale_factor': self.scale_factor,
            'lr_size': (lq_h, lq_w)
        }

    def _bgr_to_pil(self, img):
        """Convert BGR numpy array to PIL RGB Image"""
        img_rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

    def _pil_to_bgr(self, pil_img):
        """Convert PIL RGB Image to BGR numpy array"""
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img_bgr.astype(np.float32) / 255.

    def _to_tensor(self, img):
        """Convert BGR HWC image to RGB CHW tensor"""
        # BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # HWC to CHW
        img_chw = img_rgb.transpose(2, 0, 1)
        # Convert to tensor
        tensor = torch.from_numpy(img_chw).float()
        return tensor

    def tensor_to_image(self, tensor):
        """Convert tensor back to displayable image"""
        # Remove batch dimension if present
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # Convert to numpy
        img = tensor.detach().cpu().numpy()

        # CHW to HWC
        img = img.transpose(1, 2, 0)

        # Clamp values to [0, 1]
        img = np.clip(img, 0, 1)

        # Convert to uint8
        img = (img * 255).astype(np.uint8)

        # RGB to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    def validate_image_file(self, uploaded_file):
        """Validate uploaded image file"""
        if uploaded_file is None:
            return False, "No file uploaded"

        # Check file size
        if hasattr(uploaded_file, 'size') and uploaded_file.size > 50 * 1024 * 1024:
            return False, "File too large (max 50MB)"

        # Check file extension
        allowed_extensions = ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp']
        if not any(uploaded_file.name.lower().endswith(ext) for ext in allowed_extensions):
            return False, f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"

        return True, "Valid image file"
    def calculate_optimal_dimensions(self, img_h, img_w):
        """Calculate optimal HR dimensions maintaining aspect ratio"""
        # Minimum dimensions for VGG feature extraction
        min_lr = 64  # Minimum LR dimension for VGG
        min_hr = min_lr * self.scale_factor

        # Choose larger of minimum or original size
        target_h = max(min_hr, img_h)
        target_w = max(min_hr, img_w)

        # Ensure divisibility by scale_factor for proper upsampling
        target_h = (target_h // self.scale_factor) * self.scale_factor
        target_w = (target_w // self.scale_factor) * self.scale_factor

        # Ensure divisibility by 8 for Swin Transformer
        target_h = ((target_h + 7) // 8) * 8
        target_w = ((target_w + 7) // 8) * 8

        return target_h, target_w
