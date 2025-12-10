"""Image viewer and comparison components for DATSR Streamlit app"""

import streamlit as st
import cv2
import numpy as np
from io import BytesIO
import base64
from config.ui_config import COMPARISON_VIEWER_HEIGHT


class ImageViewer:
    """Handle image display and comparison"""

    def render_results_section(self, results):
        """Render comprehensive results section"""
        if not results['success']:
            st.error(f"❌ Processing failed: {results['error']}")
            return

        st.markdown("---")
        st.subheader("🎉 Super-Resolution Results")

        # Results summary
        self._render_results_summary(results)

        # Main comparison viewer
        self._render_comparison_viewer(results)

        # Detailed view tabs
        self._render_detailed_views(results)

        # Download section
        self._render_download_section(results)

    def _render_results_summary(self, results):
        """Render results summary metrics"""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Processing Time",
                f"{results['inference_time']:.2f}s"
            )

        with col2:
            st.metric(
                "Model Used",
                results['model_type'].upper()
            )

        with col3:
            st.metric(
                "Scale Factor",
                f"{results['scale_factor']}x"
            )

        with col4:
            st.metric(
                "Device",
                results['device'].upper()
            )

        # Image size information
        with st.expander("📏 Image Information"):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Input Size:**")
                st.code(f"{results['original_size'][1]} × {results['original_size'][0]} pixels")

            with col2:
                st.write("**Output Size:**")
                st.code(f"{results['processed_size'][1]} × {results['processed_size'][0]} pixels")

            scale_achieved = (results['processed_size'][1] / results['original_size'][1],
                            results['processed_size'][0] / results['original_size'][0])

            st.write("**Actual Scale:**")
            st.code(f"{scale_achieved[0]:.1f}x × {scale_achieved[1]:.1f}x")

    def _render_comparison_viewer(self, results):
        """Render main comparison viewer"""
        st.markdown("### 🔍 Interactive Comparison")

        # Create comparison tabs
        tab1, tab2, tab3 = st.tabs(["📊 Side-by-Side", "🔄 Slider Comparison", "📈 Quality Analysis"])

        with tab1:
            self._render_side_by_side(results)

        with tab2:
            self._render_slider_comparison(results)

        with tab3:
            self._render_quality_analysis(results)

    def _render_side_by_side(self, results):
        """Render side-by-side comparison"""
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Input / Bicubic**")
            if results.get('bicubic_image') is not None:
                st.image(results['bicubic_image'],
                        caption="Bicubic Upscaled",
                        use_container_width=True)
            else:
                st.image(results['lr_image'],
                        caption="Low Resolution Input",
                        use_container_width=True)

        with col2:
            st.markdown("**DATSR Result**")
            st.image(results['sr_image'],
                    caption="DATSR Super-Resolution",
                    use_container_width=True)

    def _render_slider_comparison(self, results):
        """Render interactive slider comparison"""
        st.markdown("**Drag the slider to compare images**")

        # For now, show before/after images
        # TODO: Implement actual slider component using custom HTML/CSS
        col1, col2 = st.columns(2)

        with col1:
            st.image(results['bicubic_image'] if results.get('bicubic_image') is not None else results['lr_image'],
                    caption="Before", use_container_width=True)

        with col2:
            st.image(results['sr_image'], caption="After", use_container_width=True)

        st.info("🚀 Advanced slider comparison coming soon!")

    def _render_quality_analysis(self, results):
        """Render quality analysis and metrics"""
        st.markdown("**Image Quality Analysis**")

        # Basic metrics calculation
        if results.get('lr_image') is not None and results.get('sr_image') is not None:
            # Calculate some basic metrics
            lr_resized = self._resize_to_match(results['lr_image'], results['sr_image'])

            # Calculate PSNR
            psnr = self._calculate_psnr(lr_resized, results['sr_image'])
            st.metric("Peak Signal-to-Noise Ratio (PSNR)", f"{psnr:.2f} dB")

            # Calculate SSIM
            try:
                ssim = self._calculate_ssim(lr_resized, results['sr_image'])
                st.metric("Structural Similarity (SSIM)", f"{ssim:.4f}")
            except:
                st.info("SSIM calculation not available")

        # Image statistics
        with st.expander("📊 Detailed Statistics"):
            self._render_image_statistics(results)

    def _render_detailed_views(self, results):
        """Render detailed view tabs"""
        st.markdown("### 🔬 Detailed Views")

        tab1, tab2, tab3, tab4 = st.tabs(["📱 Original LR", "🖼️ Reference", "⬆️ Bicubic", "✨ Super-Resolution"])

        with tab1:
            st.image(results['lr_image'], caption="Original Low Resolution", use_container_width=True)
            self._add_image_controls("lr", results['lr_image'])

        with tab2:
            st.image(results['ref_image'], caption="Reference Image", use_container_width=True)
            self._add_image_controls("ref", results['ref_image'])

        with tab3:
            if results.get('bicubic_image') is not None:
                st.image(results['bicubic_image'], caption="Bicubic Upscaled", use_container_width=True)
                self._add_image_controls("bicubic", results['bicubic_image'])
            else:
                st.info("Bicubic image not available")

        with tab4:
            st.image(results['sr_image'], caption="DATSR Super-Resolution", use_container_width=True)
            self._add_image_controls("sr", results['sr_image'])

    def _add_image_controls(self, prefix, image):
        """Add zoom and pan controls for image"""
        st.markdown("**Image Controls:**")

        zoom_level = st.slider(f"Zoom Level", min_value=0.25, max_value=2.0, value=1.0, step=0.25, key=f"zoom_{prefix}")

        if zoom_level != 1.0:
            # Resize image based on zoom level
            height, width = image.shape[:2]
            new_height = int(height * zoom_level)
            new_width = int(width * zoom_level)

            resized_image = cv2.resize(image, (new_width, new_height))
            st.image(resized_image, caption=f"Zoomed {zoom_level}x", width=None)

    def _render_download_section(self, results):
        """Render download options"""
        st.markdown("---")
        st.markdown("### 💾 Download Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Download super-resolution result
            sr_bytes = self._image_to_bytes(results['sr_image'], 'PNG')
            st.download_button(
                label="📥 Download Super-Resolution",
                data=sr_bytes,
                file_name="datsr_result.png",
                mime="image/png"
            )

        with col2:
            # Download bicubic comparison
            if results.get('bicubic_image') is not None:
                bicubic_bytes = self._image_to_bytes(results['bicubic_image'], 'PNG')
                st.download_button(
                    label="📥 Download Bicubic",
                    data=bicubic_bytes,
                    file_name="bicubic_result.png",
                    mime="image/png"
                )

        with col3:
            # Download all results as zip
            self._render_download_all_button(results)

    def _render_download_all_button(self, results):
        """Render download all results button"""
        if st.button("📦 Download All (ZIP)"):
            # Create zip file with all results
            import zipfile
            from datetime import datetime

            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add super-resolution result
                sr_bytes = self._image_to_bytes(results['sr_image'], 'PNG')
                zip_file.writestr("datsr_result.png", sr_bytes.getvalue())

                # Add bicubic if available
                if results.get('bicubic_image') is not None:
                    bicubic_bytes = self._image_to_bytes(results['bicubic_image'], 'PNG')
                    zip_file.writestr("bicubic_result.png", bicubic_bytes.getvalue())

                # Add metadata
                metadata = f"""DATSR Super-Resolution Results
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {results['model_type']}
Scale Factor: {results['scale_factor']}x
Processing Time: {results['inference_time']:.2f}s
Device: {results['device']}
"""
                zip_file.writestr("results_info.txt", metadata)

            st.download_button(
                label="📥 Download ZIP",
                data=zip_buffer.getvalue(),
                file_name=f"datsr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )

    def _image_to_bytes(self, image, format='PNG'):
        """Convert numpy image to bytes"""
        _, buffer = cv2.imencode(f'.{format.lower()}', image)
        return BytesIO(buffer.tobytes())

    def _resize_to_match(self, img1, img2):
        """Resize img1 to match img2 dimensions"""
        h2, w2 = img2.shape[:2]
        return cv2.resize(img1, (w2, h2))

    def _calculate_psnr(self, img1, img2):
        """Calculate PSNR between two images"""
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))

    def _calculate_ssim(self, img1, img2):
        """Calculate SSIM between two images"""
        try:
            from skimage.metrics import structural_similarity as ssim

            # Convert to grayscale for SSIM calculation
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            return ssim(gray1, gray2, data_range=255)
        except ImportError:
            return None

    def _render_image_statistics(self, results):
        """Render detailed image statistics"""
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Super-Resolution Image:**")
            self._display_image_stats(results['sr_image'])

        with col2:
            st.write("**Bicubic Image:**")
            if results.get('bicubic_image') is not None:
                self._display_image_stats(results['bicubic_image'])
            else:
                self._display_image_stats(results['lr_image'])

    def _display_image_stats(self, image):
        """Display statistics for an image"""
        if len(image.shape) == 3:
            # Color image
            mean_rgb = np.mean(image, axis=(0, 1))
            std_rgb = np.std(image, axis=(0, 1))

            st.write(f"Mean (BGR): {mean_rgb.round(1)}")
            st.write(f"Std (BGR): {std_rgb.round(1)}")
        else:
            # Grayscale
            st.write(f"Mean: {np.mean(image):.1f}")
            st.write(f"Std: {np.std(image):.1f}")

        st.write(f"Min: {np.min(image)}")
        st.write(f"Max: {np.max(image)}")
        st.write(f"Shape: {image.shape}")