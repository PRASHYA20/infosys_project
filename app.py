import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import io
import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Set page config
st.set_page_config(
    page_title="Oil Spill Detection - Correct",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Oil Spill Detection - Image Segmentation")
st.write("Proper oil spill detection using image segmentation techniques")

# Check for model files
def find_segmentation_models():
    """Find proper segmentation model files"""
    model_files = []
    for f in os.listdir('.'):
        if f.endswith(('.pth', '.pt', '.pkl', '.h5', '.onnx')):
            model_files.append(f)
    return model_files

model_files = find_segmentation_models()

# Display model status
st.sidebar.header("🔧 Model Status")
if model_files:
    for model_file in model_files:
        st.sidebar.success(f"✅ {model_file}")
else:
    st.sidebar.warning("🤖 No model files found - Using computer vision")

# Settings
st.sidebar.header("⚙️ Detection Settings")
confidence_threshold = st.sidebar.slider(
    "Detection Confidence", 
    0.1, 0.9, 0.5, 0.1
)

min_spill_size = st.sidebar.slider("Minimum Spill Size", 100, 2000, 500, 100)
max_spill_size = st.sidebar.slider("Maximum Spill Size", 1000, 10000, 3000, 500)

def proper_image_segmentation(image_array):
    """
    Proper oil spill detection using computer vision and segmentation
    """
    h, w = image_array.shape[:2]
    
    # Convert to appropriate color spaces
    rgb = image_array
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    
    # Extract channels
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    l, a, lab_b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    
    # 1. Water detection using multiple methods
    water_mask = detect_water_areas(rgb, hsv, lab)
    
    # 2. Oil spill detection within water areas
    oil_mask = detect_oil_spills(rgb, hsv, lab, water_mask)
    
    # 3. Post-processing
    oil_mask = post_process_mask(oil_mask, min_spill_size, max_spill_size)
    
    return oil_mask, water_mask

def detect_water_areas(rgb, hsv, lab):
    """
    Detect water areas using multiple computer vision techniques
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    # Method 1: Blue dominance (water is typically blue)
    blue_dominance = (b > r * 1.2) & (b > g * 1.2)
    
    # Method 2: HSV-based water detection
    water_hsv = (h > 90) & (h < 130) & (s > 30) & (s < 200) & (v > 50)
    
    # Method 3: NDWI-like calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = (g.astype(float) - r.astype(float)) / (g.astype(float) + r.astype(float) + 1e-8)
        ndwi = np.nan_to_num(ndwi)
    water_ndwi = ndwi > 0.1
    
    # Method 4: Texture-based (water is smoother)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    texture = cv2.Laplacian(gray, cv2.CV_64F)
    texture = np.abs(texture)
    smooth_areas = texture < np.percentile(texture, 40)
    
    # Combine methods
    water_confidence = (
        blue_dominance.astype(float) * 0.4 +
        water_hsv.astype(float) * 0.3 +
        water_ndwi.astype(float) * 0.2 +
        smooth_areas.astype(float) * 0.1
    )
    
    water_mask = water_confidence > 0.4
    
    # Clean up water mask
    kernel = np.ones((5, 5), np.uint8)
    water_mask = water_mask.astype(np.uint8)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
    
    return water_mask.astype(bool)

def detect_oil_spills(rgb, hsv, lab, water_mask):
    """
    Detect oil spills within water areas using visual characteristics
    """
    h, w = rgb.shape[:2]
    oil_mask = np.zeros((h, w), dtype=np.float32)
    
    if not np.any(water_mask):
        return oil_mask
    
    # Work only within water areas
    water_pixels = np.where(water_mask)
    
    if len(water_pixels[0]) == 0:
        return oil_mask
    
    # Oil spill characteristics in water:
    # 1. Darker than surrounding water
    # 2. Rainbow sheen patterns
    # 3. Smooth texture differences
    # 4. Specific color patterns
    
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    # 1. Dark regions in water (oil is often darker)
    water_gray = gray[water_mask]
    if len(water_gray) > 0:
        dark_threshold = np.percentile(water_gray, 30)
        dark_regions = (gray < dark_threshold) & water_mask
        
        # Add dark regions to oil mask
        oil_mask[dark_regions] += 0.4
    
    # 2. Rainbow sheen detection (oil creates rainbow patterns)
    rainbow_mask = detect_rainbow_sheen(rgb, hsv, water_mask)
    oil_mask[rainbow_mask] += 0.6
    
    # 3. Edge detection for spill boundaries
    edges = cv2.Canny(gray, 50, 150)
    oil_edges = edges & water_mask
    oil_mask[oil_edges] += 0.3
    
    # 4. Color anomaly detection
    color_anomalies = detect_color_anomalies(rgb, water_mask)
    oil_mask[color_anomalies] += 0.5
    
    # Normalize and threshold
    oil_mask = np.clip(oil_mask, 0, 1)
    oil_mask = oil_mask * water_mask  # Only in water areas
    
    return oil_mask

def detect_rainbow_sheen(rgb, hsv, water_mask):
    """
    Detect rainbow sheen patterns characteristic of oil spills
    """
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    # Rainbow sheen typically has high saturation variation
    saturation_variation = cv2.blur(s, (5, 5))
    saturation_std = cv2.blur(np.abs(s - saturation_variation), (5, 5))
    
    # High saturation variation in small areas
    high_sat_variation = saturation_std > np.percentile(saturation_std[water_mask], 70) if np.any(water_mask) else np.zeros_like(water_mask)
    
    # Combined with medium-high saturation
    medium_saturation = (s > 50) & (s < 200)
    
    rainbow_sheen = high_sat_variation & medium_saturation & water_mask
    
    return rainbow_sheen

def detect_color_anomalies(rgb, water_mask):
    """
    Detect color anomalies that indicate oil contamination
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    
    # Oil often creates specific color shifts
    # High a-channel values (green-red axis) can indicate oil
    a_channel_anomaly = a > np.percentile(a[water_mask], 70) if np.any(water_mask) else np.zeros_like(water_mask)
    
    # Specific b-channel patterns (blue-yellow axis)
    b_channel_anomaly = (b > np.percentile(b[water_mask], 60)) & (b < np.percentile(b[water_mask], 90)) if np.any(water_mask) else np.zeros_like(water_mask)
    
    color_anomalies = (a_channel_anomaly | b_channel_anomaly) & water_mask
    
    return color_anomalies

def post_process_mask(oil_mask, min_size, max_size):
    """
    Post-process the oil spill mask
    """
    oil_mask_uint8 = (oil_mask * 255).astype(np.uint8)
    
    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)
    oil_mask_clean = cv2.morphologyEx(oil_mask_uint8, cv2.MORPH_OPEN, kernel)
    
    # Fill small holes
    oil_mask_clean = cv2.morphologyEx(oil_mask_clean, cv2.MORPH_CLOSE, kernel)
    
    # Remove small components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(oil_mask_clean, connectivity=8)
    
    final_mask = np.zeros_like(oil_mask_clean)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_size <= area <= max_size:
            final_mask[labels == i] = 255
    
    # Smooth boundaries
    final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)
    final_mask = (final_mask > 127).astype(np.float32)
    
    return final_mask

def create_correct_overlay(original_image, oil_mask, water_mask=None):
    """
    Create correct overlay showing actual detections
    """
    if isinstance(original_image, np.ndarray):
        overlay_array = original_image.copy()
    else:
        overlay_array = np.array(original_image)
    
    # Convert to RGB if needed
    if len(overlay_array.shape) == 2:
        overlay_array = cv2.cvtColor(overlay_array, cv2.COLOR_GRAY2RGB)
    elif overlay_array.shape[2] == 1:
        overlay_array = cv2.cvtColor(overlay_array, cv2.COLOR_GRAY2RGB)
    
    # Highlight water areas lightly
    if water_mask is not None:
        water_color = [100, 150, 255]  # Light blue
        overlay_array[water_mask] = overlay_array[water_mask] * 0.8 + np.array(water_color) * 0.2
    
    # Highlight oil spills in red
    oil_areas = oil_mask > confidence_threshold
    
    if np.any(oil_areas):
        # Different colors based on confidence
        oil_intensities = oil_mask[oil_areas]
        
        for intensity, (y, x) in zip(oil_intensities, zip(*np.where(oil_areas))):
            if intensity < 0.5:
                # Light spill - orange
                overlay_array[y, x] = [255, 165, 0]
            elif intensity < 0.8:
                # Medium spill - red-orange
                overlay_array[y, x] = [255, 100, 0]
            else:
                # Heavy spill - bright red
                overlay_array[y, x] = [255, 0, 0]
    
    return Image.fromarray(overlay_array.astype(np.uint8))

def resize_mask_properly(mask, target_shape):
    """Resize mask properly maintaining boundaries"""
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_resized = cv2.resize(mask_uint8, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask_resized.astype(np.float32) / 255.0

def main():
    uploaded_file = st.file_uploader(
        "📤 Upload Satellite Image", 
        type=["jpg", "jpeg", "png", "tiff"],
        help="Upload clear satellite imagery with water bodies"
    )
    
    if uploaded_file is not None:
        try:
            # Load image
            image = Image.open(uploaded_file).convert("RGB")
            image_array = np.array(image)
            
            # Display layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("🛰️ Original Image")
                st.image(image, use_container_width=True)
                st.caption(f"Size: {image.size} | Channels: {image_array.shape[2]}")
            
            # Process image
            with st.spinner("🔍 Performing proper oil spill detection..."):
                # Perform actual image segmentation
                oil_mask, water_mask = proper_image_segmentation(image_array)
                
                # Create binary mask
                binary_mask = (oil_mask > confidence_threshold).astype(np.uint8) * 255
                
                # Create correct overlay
                overlay_img = create_correct_overlay(image_array, oil_mask, water_mask)
                
                # Convert to PIL for display
                mask_display = Image.fromarray(binary_mask)
                water_display = Image.fromarray((water_mask * 255).astype(np.uint8))
                
                # Display results
                with col2:
                    st.subheader("💧 Water Detection")
                    st.image(water_display, use_container_width=True, clamp=True)
                    water_coverage = np.sum(water_mask) / water_mask.size * 100
                    st.caption(f"Water Coverage: {water_coverage:.1f}%")
                
                with col3:
                    st.subheader("🛢️ Oil Spill Detection")
                    st.image(overlay_img, use_container_width=True)
                    st.caption("Red = Oil spills | Blue = Water areas")
            
            # Analysis
            spill_pixels = np.sum(binary_mask > 0)
            water_pixels = np.sum(water_mask)
            total_pixels = binary_mask.size
            
            spill_coverage = (spill_pixels / total_pixels) * 100
            water_coverage = (water_pixels / total_pixels) * 100
            
            st.subheader("📊 Correct Analysis Results")
            
            col_anal1, col_anal2, col_anal3, col_anal4 = st.columns(4)
            
            with col_anal1:
                st.metric("Water Coverage", f"{water_coverage:.1f}%")
            
            with col_anal2:
                st.metric("Spill Coverage", f"{spill_coverage:.4f}%")
            
            with col_anal3:
                if water_pixels > 0:
                    contamination = (spill_pixels / water_pixels) * 100
                    st.metric("Water Contamination", f"{contamination:.3f}%")
                else:
                    st.metric("Water Contamination", "0%")
            
            with col_anal4:
                st.metric("Detection Method", "Computer Vision")
            
            # Technical details
            with st.expander("🔍 Technical Detection Details"):
                st.write("**Detection Methods Used:**")
                st.write("- Water detection: Color analysis + NDWI + texture")
                st.write("- Oil detection: Dark regions + rainbow sheen + color anomalies")
                st.write("- Post-processing: Size filtering + boundary smoothing")
                
                st.write("**Detection Parameters:**")
                st.write(f"- Confidence threshold: {confidence_threshold}")
                st.write(f"- Min spill size: {min_spill_size} pixels")
                st.write(f"- Max spill size: {max_spill_size} pixels")
            
            # Download results
            st.subheader("💾 Download Correct Results")
            dl_col1, dl_col2, dl_col3 = st.columns(3)
            
            with dl_col1:
                buf_water = io.BytesIO()
                water_display.save(buf_water, format="PNG")
                st.download_button(
                    "📥 Water Mask",
                    data=buf_water.getvalue(),
                    file_name="water_detection.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with dl_col2:
                buf_mask = io.BytesIO()
                mask_display.save(buf_mask, format="PNG")
                st.download_button(
                    "📥 Oil Spill Mask",
                    data=buf_mask.getvalue(),
                    file_name="oil_spill_mask.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with dl_col3:
                buf_overlay = io.BytesIO()
                overlay_img.save(buf_overlay, format="PNG")
                st.download_button(
                    "📥 Detection Overlay",
                    data=buf_overlay.getvalue(),
                    file_name="detection_overlay.png",
                    mime="image/png",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"❌ Detection error: {str(e)}")
    
    else:
        st.info("👆 **Upload a satellite image for proper oil spill detection**")

if __name__ == "__main__":
    main()
