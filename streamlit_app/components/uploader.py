"""Image upload components for DATSR Streamlit app"""

import streamlit as st
import os
from config.ui_config import SUPPORTED_FORMATS, MAX_FILE_SIZE, EXAMPLE_IMAGES_DIR


class ImageUploader:
    """Handle dual image upload interface"""

    def __init__(self):
        self.example_images = self._load_example_images()

    def render_upload_section(self):
        """Render the main upload section with two columns"""
        st.markdown("---")
        st.subheader("📤 Upload Images")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Low-Resolution Input")
            lr_file = self._render_single_uploader(
                "lr_uploader",
                "Choose LR image or drop it here",
                "low-resolution input"
            )

        with col2:
            st.markdown("##### Reference Image")
            ref_file = self._render_single_uploader(
                "ref_uploader",
                "Choose reference image or drop it here",
                "reference"
            )

        # Example images section
        self._render_example_images()

        return lr_file, ref_file

    def _render_single_uploader(self, key, help_text, image_type):
        """Render a single image uploader with validation"""
        uploaded_file = st.file_uploader(
            help_text,
            type=SUPPORTED_FORMATS,
            key=key,
            help=f"Supported formats: {', '.join(SUPPORTED_FORMATS).upper()} | Max size: {MAX_FILE_SIZE//1024//1024}MB"
        )

        if uploaded_file is not None:
            # Display image preview
            st.image(uploaded_file, caption=f"Uploaded {image_type}", use_container_width=True)

            # Display file info
            self._display_file_info(uploaded_file)

        return uploaded_file

    def _display_file_info(self, uploaded_file):
        """Display information about uploaded file"""
        col1, col2 = st.columns(2)

        with col1:
            st.metric("File Name", uploaded_file.name[:20] + "..." if len(uploaded_file.name) > 20 else uploaded_file.name)

        with col2:
            size_mb = uploaded_file.size / (1024 * 1024)
            st.metric("File Size", f"{size_mb:.2f} MB")

    def _load_example_images(self):
        """Load available example images"""
        examples = {}

        if os.path.exists(EXAMPLE_IMAGES_DIR):
            for filename in os.listdir(EXAMPLE_IMAGES_DIR):
                if filename.lower().endswith(tuple(SUPPORTED_FORMATS)):
                    file_path = os.path.join(EXAMPLE_IMAGES_DIR, filename)

                    # Categorize images by naming convention
                    if 'lr' in filename.lower() or 'input' in filename.lower():
                        if 'lr_examples' not in examples:
                            examples['lr_examples'] = []
                        examples['lr_examples'].append({
                            'name': filename,
                            'path': file_path,
                            'type': 'LR'
                        })
                    elif 'ref' in filename.lower() or 'reference' in filename.lower():
                        if 'ref_examples' not in examples:
                            examples['ref_examples'] = []
                        examples['ref_examples'].append({
                            'name': filename,
                            'path': file_path,
                            'type': 'Reference'
                        })

        return examples

    def _render_example_images(self):
        """Render example image selection"""
        if not self.example_images:
            return

        with st.expander("📷 Try Example Images"):
            st.write("Click on any example image to load it automatically")

            # LR examples
            if 'lr_examples' in self.example_images and self.example_images['lr_examples']:
                st.markdown("**Low-Resolution Examples:**")
                cols = st.columns(min(len(self.example_images['lr_examples']), 4))

                for i, example in enumerate(self.example_images['lr_examples']):
                    with cols[i % 4]:
                        if st.button(example['name'], key=f"lr_{i}"):
                            self._load_example_to_session(example, 'lr')
                        st.image(example['path'], caption=example['name'], width=120)

            # Reference examples
            if 'ref_examples' in self.example_images and self.example_images['ref_examples']:
                st.markdown("**Reference Examples:**")
                cols = st.columns(min(len(self.example_images['ref_examples']), 4))

                for i, example in enumerate(self.example_images['ref_examples']):
                    with cols[i % 4]:
                        if st.button(example['name'], key=f"ref_{i}"):
                            self._load_example_to_session(example, 'ref')
                        st.image(example['path'], caption=example['name'], width=120)

    def _load_example_to_session(self, example, image_type):
        """Load example image into session state"""
        try:
            with open(example['path'], 'rb') as f:
                file_bytes = f.read()

            # Store in session state
            session_key = f"{image_type}_file"
            st.session_state[session_key] = {
                'name': example['name'],
                'type': example['type'],
                'data': file_bytes
            }

            st.success(f"Loaded {example['type']} example: {example['name']}")
            st.rerun()

        except Exception as e:
            st.error(f"Failed to load example image: {str(e)}")

    def get_uploaded_files(self):
        """Get currently uploaded or selected files"""
        lr_file = None
        ref_file = None

        # Check for uploaded files
        if 'lr_uploader' in st.session_state and st.session_state.lr_uploader is not None:
            lr_file = st.session_state.lr_uploader

        if 'ref_uploader' in st.session_state and st.session_state.ref_uploader is not None:
            ref_file = st.session_state.ref_uploader

        # Check for session state files (from examples)
        if lr_file is None and 'lr_file' in st.session_state:
            lr_file = st.session_state.lr_file['data']
            # Convert to file-like object
            from io import BytesIO
            lr_file = BytesIO(lr_file)
            lr_file.name = st.session_state.lr_file['name']

        if ref_file is None and 'ref_file' in st.session_state:
            ref_file = st.session_state.ref_file['data']
            # Convert to file-like object
            from io import BytesIO
            ref_file = BytesIO(ref_file)
            ref_file.name = st.session_state.ref_file['name']

        return lr_file, ref_file

    def validate_files(self, lr_file, ref_file):
        """Validate uploaded files"""
        errors = []

        if lr_file is None:
            errors.append("Please upload a low-resolution input image")

        if ref_file is None:
            errors.append("Please upload a reference image")

        if lr_file is not None and ref_file is not None:
            # Check file sizes
            if hasattr(lr_file, 'size') and lr_file.size > MAX_FILE_SIZE:
                errors.append(f"LR image is too large (max {MAX_FILE_SIZE//1024//1024}MB)")

            if hasattr(ref_file, 'size') and ref_file.size > MAX_FILE_SIZE:
                errors.append(f"Reference image is too large (max {MAX_FILE_SIZE//1024//1024}MB)")

        return len(errors) == 0, errors

    def render_file_validation(self, lr_file, ref_file):
        """Render file validation status"""
        is_valid, errors = self.validate_files(lr_file, ref_file)

        if not is_valid:
            st.error("Please fix the following issues:")
            for error in errors:
                st.error(f"• {error}")
            return False
        else:
            st.success("✅ Both images uploaded successfully!")
            return True