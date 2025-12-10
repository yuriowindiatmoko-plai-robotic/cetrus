"""Utility functions for DATSR Streamlit app"""

import os
import sys
import time
import streamlit as st
import torch
import numpy as np


def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="DATSR Super-Resolution",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def add_custom_css():
    """Add custom CSS styling"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }

    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }

    .upload-container {
        border: 2px dashed #ddd;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }

    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }

    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }

    .info-message {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }

    /* Hide Streamlit footer */
    footer {
        visibility: hidden;
    }

    /* Custom button styling */
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #155a8a;
    }
    </style>
    """, unsafe_allow_html=True)


def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []

    try:
        import torch
        import torchvision
    except ImportError:
        missing_deps.append("PyTorch")

    try:
        import cv2
    except ImportError:
        missing_deps.append("OpenCV")

    try:
        import mmcv
    except ImportError:
        missing_deps.append("mmcv-full")

    try:
        import PIL
    except ImportError:
        missing_deps.append("Pillow")

    try:
        import numpy
    except ImportError:
        missing_deps.append("NumPy")

    return missing_deps


def check_cuda_availability():
    """Check CUDA availability and provide recommendations"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        return {
            'available': True,
            'device_name': device_name,
            'memory_gb': memory_gb,
            'recommendation': f"GPU acceleration available - {device_name} ({memory_gb:.1f}GB)"
        }
    else:
        return {
            'available': False,
            'recommendation': "Using CPU processing - Install CUDA version for faster speed"
        }


def validate_model_files():
    """Check if pretrained model files exist"""
    base_path = os.path.join(os.path.dirname(__file__), '..', '..', 'DATSR', 'experiments', 'pretrained_model')

    required_files = [
        'feature_extraction.pth',
        'restoration_mse.pth',
        'restoration_gan.pth'
    ]

    missing_files = []
    existing_files = []

    for file in required_files:
        file_path = os.path.join(base_path, file)
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            existing_files.append(f"✅ {file} ({size_mb:.1f}MB)")
        else:
            missing_files.append(f"❌ {file}")

    return {
        'all_exist': len(missing_files) == 0,
        'existing_files': existing_files,
        'missing_files': missing_files,
        'model_path': base_path
    }


def render_system_status():
    """Render system status dashboard"""
    with st.expander("🔧 System Status Check", expanded=False):
        # Check dependencies
        missing_deps = check_dependencies()

        st.markdown("### 📦 Dependency Check")
        if not missing_deps:
            st.success("✅ All dependencies are properly installed")
        else:
            st.error("❌ Missing dependencies:")
            for dep in missing_deps:
                st.error(f"• {dep}")

        # Check CUDA
        st.markdown("### 🚀 GPU Status")
        cuda_info = check_cuda_availability()
        if cuda_info['available']:
            st.success(cuda_info['recommendation'])
            st.info(f"GPU Memory: {cuda_info['memory_gb']:.1f}GB")
        else:
            st.warning(cuda_info['recommendation'])

        # Check model files
        st.markdown("### 🤖 Model Files Check")
        model_status = validate_model_files()

        if model_status['all_exist']:
            st.success("✅ All pretrained model files exist")
        else:
            st.error("❌ Missing model files:")
            for missing in model_status['missing_files']:
                st.error(f"{missing}")

            st.info(f"📁 Model Path: {model_status['model_path']}")
            st.markdown("""
            **Download Instructions:**
            Please visit [DATSR GitHub Releases](https://github.com/caojiezhang/DATSR/releases) to download pretrained models,
            and place them in the `DATSR/experiments/pretrained_model/` directory.
            """)


def create_progress_callback():
    """Create a progress callback for Streamlit"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    def callback(message, progress):
        progress_bar.progress(progress)
        status_text.text(message)

    return callback, progress_bar, status_text


def clear_progress(progress_bar, status_text):
    """Clear progress indicators"""
    progress_bar.empty()
    status_text.empty()


def format_time(seconds):
    """Format time in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"


def estimate_processing_time(image_shape, scale_factor, device='cpu'):
    """Estimate processing time based on image size and device"""
    height, width = image_shape[:2]
    pixels = height * width

    # Base processing times (in seconds)
    base_time_cpu = pixels / (1000 * 1000) * 0.5  # 0.5s per megapixel on CPU
    base_time_gpu = pixels / (1000 * 1000) * 0.05  # 0.05s per megapixel on GPU

    # Scale factor adjustment
    scale_adjustment = scale_factor / 4.0  # Adjust based on scale factor

    if device == 'cuda':
        estimated_time = base_time_gpu * scale_adjustment
    else:
        estimated_time = base_time_cpu * scale_adjustment

    return estimated_time


def save_to_session_state(key, value):
    """Save value to Streamlit session state"""
    st.session_state[key] = value


def load_from_session_state(key, default=None):
    """Load value from Streamlit session state"""
    return st.session_state.get(key, default)


def clear_session_state(keys=None):
    """Clear specific keys or all session state"""
    if keys is None:
        st.session_state.clear()
    else:
        for key in keys:
            if key in st.session_state:
                del st.session_state[key]


class ErrorHandler:
    """Handle and display errors gracefully"""

    @staticmethod
    def handle_processing_error(error, show_traceback=False):
        """Handle processing errors with user-friendly messages"""
        error_str = str(error)

        # Common error patterns
        if "CUDA out of memory" in error_str:
            st.error("🚫 GPU out of memory! Please try:")
            st.error("• Use smaller images")
            st.error("• Switch to CPU mode")
            st.error("• Restart the app to free memory")

        elif "model" in error_str.lower() and "not found" in error_str.lower():
            st.error("🤖 Model files not found!")
            st.error("Please check if pretrained model files are correctly downloaded and placed in the specified directory")

        elif "Failed to load" in error_str:
            st.error("📁 File loading failed!")
            st.error("Please check if uploaded image formats are correct")

        else:
            st.error(f"❌ Processing failed: {error_str}")

        if show_traceback:
            with st.expander("View detailed error information"):
                st.code(error_str)

    @staticmethod
    def handle_validation_error(validation_result):
        """Handle file validation errors"""
        is_valid, errors = validation_result

        if not is_valid:
            st.error("❌ File validation failed:")
            for error in errors:
                st.error(f"• {error}")
            return False

        return True


def create_download_filename(prefix="datsr_result", extension="png"):
    """Create a unique filename for downloads"""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"