import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import io
import os
import math

# Set page config
st.set_page_config(
    page_title="Oil Spill Detection - No Dependencies",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Oil Spill Detection - Computer Vision")
st.write("Proper oil spill detection using only basic Python libraries")

# Settings
st.sidebar.header("⚙️ Detection Settings")
confidence_threshold = st.sidebar.slider(
    "Detection Confidence", 
    0.1, 0.9, 0.5, 0.1
)

st.sidebar.header("🔧 Advanced Settings")
min_spill_size = st.sidebar.slider("Minimum Spill Size", 100, 2000, 500, 100)
detection_sensitivity = st.sidebar.slider("Detection Sensitivity", 0.1, 1.0, 0.7, 0.1)

def proper_water_detection(image_array):
    """
    Proper water detection using only basic libraries
    """
    h, w = image_array.shape[:2]
    r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
    
    # Convert to float for calculations
    r_f, g_f, b_f = r.astype(float), g.astype(float), b.astype(float)
    
    # Method 1: Blue dominance (water is typically blue)
    blue_dominance = (b_f > r_f * 1.15) & (b_f > g_f * 1.15)
    
    # Method 2: NDWI-like calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = (g_f - r_f) / (g_f + r_f + 1e-8)
        ndwi = np.nan_to_num(ndwi)
    water_ndwi = ndwi > 0.1
    
    # Method 3: Simple texture analysis (water is smoother)
    gray = np.mean(image_array, axis=2)
    
    # Calculate gradient manually (no OpenCV)
    grad_x = np.abs(gray[:, 1:] - gray[:, :-1])
    grad_y = np.abs(gray[1:, :] - gray[:-1, :])
    
    # Pad gradients to original size
    grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode='edge')
    grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode='edge')
    
    texture = (grad_x + grad_y) / 2
    smooth_areas = texture < np.percentile(texture, 40)
    
    # Method 4: Brightness range
    brightness = (r_f + g_f + b_f) / 3
    good_brightness = (brightness > 40) & (brightness < 200)
    
    # Combine all methods
    water_confidence = (
        blue_dominance.astype(float) * 0.4 +
        water_ndwi.astype(float) * 0.3 +
        smooth_areas.astype(float) * 0.2 +
        good_brightness.astype(float) * 0.1
    )
    
    water_mask = water_confidence > 0.4
    
    # Clean mask using PIL filters instead of OpenCV morphology
    water_pil = Image.fromarray(water_mask.astype(np.uint8) * 255)
    
    # Closing operation (dilate then erode)
    water_closed = water_pil.filter(ImageFilter.MaxFilter(5))
    water_closed = water_closed.filter(ImageFilter.MinFilter(5))
    
    # Opening operation (erode then dilate)  
    water_clean = water_closed.filter(ImageFilter.MinFilter(3))
    water_clean = water_clean.filter(ImageFilter.MaxFilter(3))
    
    return np.array(water_clean) > 128

def detect_oil_spills_basic(image_array, water_mask):
    """
    Detect oil spills using only basic image processing
    """
    h, w = image_array.shape[:2]
    oil_mask = np.zeros((h, w), dtype=np.float32)
    
    if not np.any(water_mask):
        return oil_mask
    
    r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
    gray = np.mean(image_array, axis=2)
    
    # 1. Dark regions in water (oil is often darker)
    water_gray = gray[water_mask]
    if len(water_gray) > 0:
        dark_threshold = np.percentile(water_gray, 30)
        dark_regions = (gray < dark_threshold) & water_mask
        oil_mask[dark_regions] += 0.4 * detection_sensitivity
    
    # 2. Color anomalies (oil creates color shifts)
    color_anomalies = detect_color_anomalies_basic(image_array, water_mask)
    oil_mask[color_anomalies] += 0.6 * detection_sensitivity
    
    # 3. Edge detection for spill boundaries
    edges = detect_edges_basic(gray)
    oil_edges = edges & water_mask
    oil_mask[oil_edges] += 0.3 * detection_sensitivity
    
    # 4. Saturation variations (rainbow sheen)
    saturation_variations = detect_saturation_variations(image_array, water_mask)
    oil_mask[saturation_variations] += 0.5 * detection_sensitivity
    
    # Normalize and apply only to water areas
    oil_mask = np.clip(oil_mask, 0, 1)
    oil_mask = oil_mask * water_mask.astype(float)
    
    return oil_mask

def detect_color_anomalies_basic(image_array, water_mask):
    """
    Detect color anomalies without OpenCV
    """
    r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
    
    # Convert to simple LAB-like space manually
    # Simple luminance
    luminance = (r + g + b) / 3
    
    # Simple color channels (approximate a and b channels)
    a_like = r - g  # Red-green difference
    b_like = b - (r + g) / 2  # Blue relative to red/green average
    
    # Oil often creates specific color shifts
    if np.any(water_mask):
        a_threshold = np.percentile(a_like[water_mask], 70)
        b_threshold_low = np.percentile(b_like[water_mask], 60)
        b_threshold_high = np.percentile(b_like[water_mask], 90)
        
        a_anomaly = a_like > a_threshold
        b_anomaly = (b_like > b_threshold_low) & (b_like < b_threshold_high)
    else:
        a_anomaly = np.zeros_like(water_mask)
        b_anomaly = np.zeros_like(water_mask)
    
    color_anomalies = (a_anomaly | b_anomaly) & water_mask
    return color_anomalies

def detect_edges_basic(gray):
    """
    Basic edge detection without OpenCV
    """
    h, w = gray.shape
    
    # Simple gradient calculation
    grad_x = np.abs(gray[:, 1:] - gray[:, :-1])
    grad_y = np.abs(gray[1:, :] - gray[:-1, :])
    
    # Pad to original size
    grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode='edge')
    grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode='edge')
    
    # Combine gradients
    edges = (grad_x + grad_y) > 50  # Threshold for edges
    
    return edges

def detect_saturation_variations(image_array, water_mask):
    """
    Detect saturation variations (rainbow sheen) without OpenCV
    """
    r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
    
    # Simple saturation calculation
    max_channel = np.maximum(np.maximum(r, g), b)
    min_channel = np.minimum(np.minimum(r, g), b)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        saturation = (max_channel - min_channel) / (max_channel + 1e-8)
        saturation = np.nan_to_num(saturation)
    
    # Calculate local saturation variation
    # Simple blur using averaging
    saturation_blurred = simple_blur(saturation, kernel_size=5)
    saturation_diff = np.abs(saturation - saturation_blurred)
    
    # High variation areas
    if np.any(water_mask):
        variation_threshold = np.percentile(saturation_diff[water_mask], 70)
        high_variation = saturation_diff > variation_threshold
    else:
        high_variation = np.zeros_like(water_mask)
    
    # Combined with medium saturation
    medium_saturation = (saturation > 0.2) & (saturation < 0.8)  # 0-255 scaled to 0-1
    
    return high_variation & medium_saturation & water_mask

def simple_blur(image, kernel_size=3):
    """
    Simple blur implementation without OpenCV
    """
    h, w = image.shape
    blurred = np.zeros_like(image)
    
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='edge')
    
    for i in range(h):
        for j in range(w):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            blurred[i, j] = np.mean(window)
    
    return blurred

def post_process_mask_basic(oil_mask, min_size, max_size):
    """
    Post-process mask without OpenCV
    """
    h, w = oil_mask.shape
    binary_mask = (oil_mask > confidence_threshold).astype(np.uint8)
    
    # Find connected components manually
    visited = np.zeros_like(binary_mask, dtype=bool)
    components = []
    
    def flood_fill(x, y, component):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if (0 <= cx < w and 0 <= cy < h and 
                binary_mask[cy, cx] and not visited[cy, cx]):
                visited[cy, cx] = True
                component.append((cx, cy))
                # 4-directional
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    stack.append((cx + dx, cy + dy))
    
    # Find all components
    for y in range(h):
        for x in range(w):
            if binary_mask[y, x] and not visited[y, x]:
                component = []
                flood_fill(x, y, component)
                if len(component) >= min_size and len(component) <= max_size:
                    components.append(component)
    
    # Create clean mask with only valid components
    clean_mask = np.zeros_like(binary_mask)
    for component in components:
        for x, y in component:
            clean_mask[y, x] = 1
    
    # Smooth the mask
    clean_mask_pil = Image.fromarray((clean_mask * 255).astype(np.uint8))
    clean_mask_smooth = clean_mask_pil.filter(ImageFilter.GaussianBlur(radius=2))
    clean_mask = np.array(clean_mask_smooth).astype(float) / 255.0
    
    return clean_mask

def create_accurate_overlay(original_image, oil_mask, water_mask=None):
    """
    Create accurate overlay showing real detections
    """
    if isinstance(original_image, np.ndarray):
        overlay_array = original_image.copy()
    else:
        overlay_array = np.array(original_image)
    
    # Ensure RGB
    if len(overlay_array.shape) == 2:
        overlay_array = np.stack([overlay_array] * 3, axis=-1)
    elif overlay_array.shape[2] == 1:
        overlay_array = np.concatenate([overlay_array] * 3, axis=-1)
    
    # Highlight water areas lightly
    if water_mask is not None:
        water_color = [100, 150, 255]  # Light blue
        water_alpha = 0.15
        
        water_coords = np.where(water_mask)
        for y, x in zip(water_coords[0], water_coords[1]):
            overlay_array[y, x] = (
                (1 - water_alpha) * overlay_array[y, x] + 
                water_alpha * np.array(water_color)
            )
    
    # Highlight oil spills with intensity-based coloring
    oil_areas = oil_mask > confidence_threshold
    
    if np.any(oil_areas):
        oil_coords = np.where(oil_areas)
        oil_intensities = oil_mask[oil_areas]
        
        for intensity, (y, x) in zip(oil_intensities, zip(oil_coords[0], oil_coords[1])):
            if intensity < 0.4:
                # Light spill - yellow
                overlay_array[y, x] = [255, 255, 0]
            elif intensity < 0.7:
                # Medium spill - orange
                overlay_array[y, x] = [255, 165, 0]
            else:
                # Heavy spill - red
                overlay_array[y, x] = [255, 0, 0]
    
    return Image.fromarray(overlay_array.astype(np.uint8))

def main():
    uploaded_file = st.file_uploader(
        "📤 Upload Satellite Image", 
        type=["jpg", "jpeg", "png"],
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
                st.caption(f"Size: {image.size}")
            
            # Process image
            with st.spinner("🔍 Performing accurate oil spill detection..."):
                # Detect water areas
                water_mask = proper_water_detection(image_array)
                
                # Detect oil spills
                oil_mask = detect_oil_spills_basic(image_array, water_mask)
                
                # Post-process
                final_mask = post_process_mask_basic(oil_mask, min_spill_size, 10000)
                
                # Create outputs
                binary_mask = (final_mask > confidence_threshold).astype(np.uint8) * 255
                overlay_img = create_accurate_overlay(image_array, final_mask, water_mask)
                
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
                    st.caption("Yellow/Orange/Red = Oil spill intensity")
            
            # Analysis
            spill_pixels = np.sum(binary_mask > 0)
            water_pixels = np.sum(water_mask)
            total_pixels = binary_mask.size
            
            spill_coverage = (spill_pixels / total_pixels) * 100
            water_coverage = (water_pixels / total_pixels) * 100
            
            st.subheader("📊 Accurate Analysis Results")
            
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
            with st.expander("🔍 Detection Methodology"):
                st.write("**Computer Vision Techniques Used:**")
                st.write("- Water detection: Color analysis + texture + NDWI")
                st.write("- Oil detection: Dark regions + color anomalies + edges")
                st.write("- All processing: Pure Python (no OpenCV dependency)")
                
                st.write("**Real Oil Spill Indicators Detected:**")
                st.write("- Darker regions in water bodies")
                st.write("- Color anomalies and shifts")
                st.write("- Edge patterns characteristic of spills")
                st.write("- Saturation variations (rainbow sheen)")
            
            # Download results
            st.subheader("💾 Download Accurate Results")
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
        st.info("👆 **Upload a satellite image for accurate oil spill detection**")

if __name__ == "__main__":
    main()
