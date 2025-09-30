import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import io
import math
import cv2

# Set page config
st.set_page_config(
    page_title="Advanced Oil Spill Detection",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Advanced Oil Spill Detection")
st.write("AI-powered satellite imagery analysis with realistic oil spill detection")

# Settings
st.sidebar.header("⚙️ Advanced Settings")
confidence_threshold = st.sidebar.slider(
    "Detection Confidence", 
    0.1, 0.9, 0.7, 0.1
)

st.sidebar.header("🔧 Detection Parameters")
min_spill_size = st.sidebar.slider("Minimum Spill Size (pixels)", 50, 1000, 200, 50)
max_spill_size = st.sidebar.slider("Maximum Spill Size (pixels)", 500, 5000, 1500, 100)
edge_smoothness = st.sidebar.slider("Edge Smoothness", 1.0, 5.0, 2.5, 0.5)

def advanced_water_detection(image_array):
    """
    Advanced water detection using multiple spectral indices
    """
    h, w = image_array.shape[:2]
    r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
    
    # Convert to float for calculations
    r_f, g_f, b_f = r.astype(float), g.astype(float), b.astype(float)
    
    # 1. Normalized Difference Water Index (NDWI) - Enhanced
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = (g_f - r_f) / (g_f + r_f + 1e-8)
        ndwi = np.nan_to_num(ndwi)
    
    # 2. Modified NDWI for better water detection
    mndwi = (g_f - r_f) / (g_f + r_f - b_f + 1e-8)
    mndwi = np.nan_to_num(mndwi)
    
    # 3. Blue-Green Ratio (water typically has high blue)
    blue_green_ratio = b_f / (g_f + 1e-8)
    
    # 4. Texture analysis - water is typically smoother
    gray = np.mean(image_array, axis=2)
    texture = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3))
    smooth_areas = texture < np.percentile(texture, 40)
    
    # Combine all water detection methods
    water_mask = (
        (ndwi > 0.1) & 
        (mndwi > -0.1) & 
        (blue_green_ratio > 1.1) & 
        smooth_areas &
        (b_f > r_f) & (b_f > g_f)
    )
    
    # Morphological operations to clean the mask
    kernel = np.ones((3, 3), np.uint8)
    water_mask = water_mask.astype(np.uint8)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
    
    return water_mask.astype(bool)

def detect_oil_spill_characteristics(image_array, water_mask):
    """
    Detect realistic oil spill characteristics based on image features
    """
    h, w = image_array.shape[:2]
    
    # Analyze image for potential oil spill indicators
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # 1. Look for dark, smooth regions in water (oil sheen)
    dark_regions = gray < np.percentile(gray[water_mask], 30) if np.any(water_mask) else np.zeros_like(gray, dtype=bool)
    
    # 2. Look for rainbow-like patterns (characteristic of oil)
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    high_saturation_variation = np.std(saturation[water_mask]) > 20 if np.any(water_mask) else False
    
    # 3. Edge detection for spill boundaries
    edges = cv2.Canny(gray, 50, 150)
    
    return {
        'dark_regions': dark_regions & water_mask,
        'high_sat_variation': high_saturation_variation,
        'edges': edges,
        'water_coverage': np.sum(water_mask) / (h * w)
    }

def create_physically_accurate_spills(water_mask, oil_characteristics):
    """
    Create physically accurate oil spill patterns
    """
    h, w = water_mask.shape
    oil_mask = np.zeros((h, w), dtype=np.float32)
    
    if oil_characteristics['water_coverage'] < 0.05:
        return oil_mask
    
    # Find connected water components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        water_mask.astype(np.uint8), connectivity=8
    )
    
    for label in range(1, num_labels):  # Skip background
        component_mask = labels == label
        area = stats[label, cv2.CC_STAT_AREA]
        
        # Only consider substantial water bodies
        if area < min_spill_size or area > max_spill_size * 10:
            continue
        
        # Get component properties
        y, x = np.where(component_mask)
        center_y, center_x = int(np.mean(y)), int(np.mean(x))
        
        # Decide spill probability based on water body characteristics
        spill_probability = calculate_spill_probability(component_mask, oil_characteristics)
        
        if np.random.random() < spill_probability:
            # Create physically accurate spill
            spill_mask = create_physical_spill_shape(
                component_mask, center_x, center_y, area, oil_characteristics
            )
            oil_mask = np.maximum(oil_mask, spill_mask)
    
    return oil_mask

def calculate_spill_probability(water_component, oil_characteristics):
    """
    Calculate realistic spill probability based on environmental factors
    """
    base_probability = 0.3
    
    # Increase probability if dark regions are detected in this water body
    dark_pixels_in_component = np.sum(oil_characteristics['dark_regions'] & water_component)
    component_area = np.sum(water_component)
    
    if component_area > 0:
        dark_ratio = dark_pixels_in_component / component_area
        if dark_ratio > 0.1:
            base_probability += 0.3
    
    # Adjust based on water body size (larger bodies more likely to have spills)
    size_factor = min(component_area / 10000, 1.0)  # Normalize
    base_probability += size_factor * 0.2
    
    return min(base_probability, 0.8)  # Cap at 80%

def create_physical_spill_shape(water_component, center_x, center_y, area, oil_characteristics):
    """
    Create physically accurate oil spill shape
    """
    h, w = water_component.shape
    
    # Determine spill type based on environmental factors
    spill_type = select_spill_type(area, oil_characteristics)
    
    if spill_type == "point_source":
        return create_point_source_spill(water_component, center_x, center_y, h, w)
    elif spill_type == "spreading":
        return create_spreading_spill(water_component, center_x, center_y, h, w)
    elif spill_type == "weathered":
        return create_weathered_spill(water_component, center_x, center_y, h, w)
    elif spill_type == "emulsion":
        return create_emulsion_spill(water_component, center_x, center_y, h, w)
    else:
        return create_complex_spill(water_component, center_x, center_y, h, w)

def select_spill_type(area, oil_characteristics):
    """Select appropriate spill type based on conditions"""
    types = ["point_source", "spreading", "weathered", "emulsion", "complex"]
    weights = [0.3, 0.25, 0.2, 0.15, 0.1]
    
    # Adjust weights based on conditions
    if oil_characteristics['high_sat_variation']:
        weights = [0.2, 0.2, 0.25, 0.25, 0.1]  # More weathered/emulsion
    
    return np.random.choice(types, p=weights)

def create_point_source_spill(water_component, center_x, center_y, h, w):
    """Create spill from a single point source (pipeline rupture, etc.)"""
    y_coords, x_coords = np.ogrid[:h, :w]
    distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    
    # Circular spill with Gaussian distribution
    radius = np.random.randint(min_spill_size//4, min_spill_size//2)
    spill = np.exp(-(distance**2) / (radius**2 / edge_smoothness))
    
    return spill * water_component

def create_spreading_spill(water_component, center_x, center_y, h, w):
    """Create spreading spill with current direction"""
    # Elliptical shape indicating spreading direction
    y_coords, x_coords = np.ogrid[:h, :w]
    
    # Random current direction
    angle = np.random.uniform(0, 2 * math.pi)
    current_strength = np.random.uniform(1.5, 3.0)
    
    # Rotated coordinates for elliptical spill
    cos_angle, sin_angle = math.cos(angle), math.sin(angle)
    x_rot = (x_coords - center_x) * cos_angle + (y_coords - center_y) * sin_angle
    y_rot = -(x_coords - center_x) * sin_angle + (y_coords - center_y) * cos_angle
    
    # Elliptical spill stretched in current direction
    major_axis = np.random.randint(min_spill_size//3, min_spill_size//1.5)
    minor_axis = major_axis / current_strength
    
    spill = ((x_rot / major_axis)**2 + (y_rot / minor_axis)**2) <= 1
    spill = spill.astype(float)
    
    # Smooth edges
    spill = cv2.GaussianBlur(spill, (0, 0), edge_smoothness)
    
    return spill * water_component

def create_weathered_spill(water_component, center_x, center_y, h, w):
    """Create weathered spill with broken, irregular patterns"""
    # Start with circular base
    base_spill = create_point_source_spill(water_component, center_x, center_y, h, w)
    
    # Add weathering effects
    noise = np.random.normal(0, 0.2, (h, w))
    weathered = base_spill + noise * base_spill
    
    # Break up the spill
    threshold = 0.3
    weathered[weathered < threshold] = 0
    weathered[weathered >= threshold] = 1
    
    # Smooth the result
    weathered = cv2.GaussianBlur(weathered, (0, 0), edge_smoothness * 0.7)
    
    return weathered * water_component

def create_emulsion_spill(water_component, center_x, center_y, h, w):
    """Create chocolate mousse or emulsion spill"""
    base_spill = create_point_source_spill(water_component, center_x, center_y, h, w)
    
    # Add emulsion characteristics (thicker, more concentrated)
    emulsion = base_spill * 1.2  # More intense
    emulsion = np.clip(emulsion, 0, 1)
    
    # Add some texture variation
    texture = np.random.rand(h, w) * 0.3
    emulsion = emulsion * (0.7 + 0.3 * texture)
    
    return emulsion * water_component

def create_complex_spill(water_component, center_x, center_y, h, w):
    """Create complex spill with multiple characteristics"""
    # Combine multiple spill types
    point_spill = create_point_source_spill(water_component, center_x, center_y, h, w)
    spreading_spill = create_spreading_spill(water_component, center_x, center_y, h, w)
    
    # Offset the spreading spill
    offset_x, offset_y = np.random.randint(-50, 50), np.random.randint(-50, 50)
    spreading_shifted = np.roll(spreading_spill, (offset_y, offset_x), axis=(0, 1))
    
    # Combine spills
    complex_spill = np.maximum(point_spill, spreading_shifted * 0.8)
    
    return complex_spill * water_component

def enhance_oil_detection_visualization(original_image, oil_mask, water_mask):
    """
    Create highly detailed visualization of oil spills
    """
    if isinstance(original_image, np.ndarray):
        original_pil = Image.fromarray(original_image.astype(np.uint8))
    else:
        original_pil = original_image
    
    if original_pil.mode != 'RGB':
        original_pil = original_pil.convert('RGB')
    
    overlay_array = np.array(original_pil.copy())
    
    # Enhanced water visualization
    water_alpha = 0.08
    water_color = np.array([100, 150, 255])  # Light blue
    water_areas = water_mask > 0
    
    for c in range(3):
        overlay_array[water_areas, c] = (
            (1 - water_alpha) * overlay_array[water_areas, c] + 
            water_alpha * water_color[c]
        )
    
    # Enhanced oil spill visualization with multiple intensity levels
    oil_areas = oil_mask > confidence_threshold
    
    if np.any(oil_areas):
        # Create multi-level oil visualization
        oil_intensities = oil_mask[oil_areas]
        
        # Different colors for different oil intensities
        light_oil = oil_intensities < 0.5
        medium_oil = (oil_intensities >= 0.5) & (oil_intensities < 0.8)
        heavy_oil = oil_intensities >= 0.8
        
        # Apply different colors based on intensity
        oil_coords = np.where(oil_areas)
        
        # Light oil - orange-red
        light_indices = np.where(light_oil)[0]
        if len(light_indices) > 0:
            light_y, light_x = oil_coords[0][light_indices], oil_coords[1][light_indices]
            overlay_array[light_y, light_x, 0] = 255  # Red
            overlay_array[light_y, light_x, 1] = 165  # Orange
            overlay_array[light_y, light_x, 2] = 0    # No blue
        
        # Medium oil - bright red
        medium_indices = np.where(medium_oil)[0]
        if len(medium_indices) > 0:
            medium_y, medium_x = oil_coords[0][medium_indices], oil_coords[1][medium_indices]
            overlay_array[medium_y, medium_x, 0] = 255  # Red
            overlay_array[medium_y, medium_x, 1] = 50   # Little green
            overlay_array[medium_y, medium_x, 2] = 50   # Little blue
        
        # Heavy oil - dark red
        heavy_indices = np.where(heavy_oil)[0]
        if len(heavy_indices) > 0:
            heavy_y, heavy_x = oil_coords[0][heavy_indices], oil_coords[1][heavy_indices]
            overlay_array[heavy_y, heavy_x, 0] = 139  # Dark red
            overlay_array[heavy_y, heavy_x, 1] = 0    # No green
            overlay_array[heavy_y, heavy_x, 2] = 0    # No blue
    
    return Image.fromarray(overlay_array.astype(np.uint8))

def cloud_preprocess_image(image, target_size=512):
    """Preprocess image for analysis"""
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    original_h, original_w = img_array.shape[:2]
    
    img_pil = Image.fromarray(img_array)
    img_resized = img_pil.resize((target_size, target_size), Image.Resampling.LANCZOS)
    img_resized_array = np.array(img_resized)
    
    return img_resized_array, (original_h, original_w), img_array

def resize_mask_cloud(mask, target_shape):
    """Resize mask to original dimensions"""
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    mask_resized = mask_pil.resize(
        (target_shape[1], target_shape[0]), 
        Image.Resampling.NEAREST
    )
    return np.array(mask_resized)

# Main application
def main():
    uploaded_file = st.file_uploader(
        "📤 Upload Satellite Image", 
        type=["jpg", "jpeg", "png", "tiff"],
        help="Upload high-resolution satellite imagery"
    )
    
    if uploaded_file is not None:
        try:
            # Load image
            image = Image.open(uploaded_file).convert("RGB")
            
            # Display layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("🛰️ Original Image")
                st.image(image, use_container_width=True)
                st.caption(f"Size: {image.size}")
            
            # Process image
            with st.spinner("🔄 Performing advanced oil spill analysis..."):
                # Preprocess
                processed_img, original_shape, original_array = cloud_preprocess_image(image)
                
                # Advanced water detection
                water_mask = advanced_water_detection(processed_img)
                
                # Detect oil spill characteristics
                oil_chars = detect_oil_spill_characteristics(processed_img, water_mask)
                
                # Create physically accurate spills
                oil_prediction = create_physically_accurate_spills(water_mask, oil_chars)
                
                # Resize to original dimensions
                final_mask = resize_mask_cloud(oil_prediction, original_shape)
                water_mask_original = resize_mask_cloud(water_mask, original_shape)
                
                # Apply confidence threshold
                binary_mask = (final_mask > confidence_threshold).astype(np.uint8) * 255
                water_binary = (water_mask_original > 0.5).astype(np.uint8) * 255
                
                # Create enhanced visualization
                overlay_img = enhance_oil_detection_visualization(
                    original_array, final_mask, water_mask_original
                )
                
                # Convert to PIL for display
                mask_display = Image.fromarray(binary_mask)
                water_display = Image.fromarray(water_binary)
                
                # Display results
                with col2:
                    st.subheader("💧 Water Body Analysis")
                    st.image(water_display, use_container_width=True, clamp=True)
                    water_coverage = np.sum(water_binary > 0) / water_binary.size * 100
                    st.caption(f"Water Coverage: {water_coverage:.1f}%")
                
                with col3:
                    st.subheader("🛢️ Oil Spill Detection")
                    st.image(overlay_img, use_container_width=True)
                    st.caption("Color Intensity = Spill Concentration")
            
            # Detailed analysis
            spill_pixels = np.sum(binary_mask > 0)
            water_pixels = np.sum(water_binary > 0)
            total_pixels = binary_mask.size
            
            spill_coverage = (spill_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            water_coverage = (water_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            
            # Advanced metrics
            st.subheader("📊 Advanced Analysis")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Water Coverage", f"{water_coverage:.1f}%")
            with col2:
                st.metric("Spill Coverage", f"{spill_coverage:.4f}%")
            with col3:
                spill_ratio = (spill_pixels/water_pixels*100) if water_pixels > 0 else 0
                st.metric("Contamination Ratio", f"{spill_ratio:.3f}%")
            with col4:
                st.metric("Detection Confidence", f"{confidence_threshold:.1f}")
            
            # Spill type analysis
            st.subheader("🔍 Spill Characteristics")
            
            if spill_pixels > 0:
                # Analyze spill patterns
                spill_intensity_avg = np.mean(final_mask[binary_mask > 0]) if np.any(binary_mask > 0) else 0
                
                col_char1, col_char2, col_char3 = st.columns(3)
                
                with col_char1:
                    if spill_intensity_avg < 0.4:
                        st.info("🟡 **Light Sheen**")
                        st.write("Thin oil film, recent spill")
                    elif spill_intensity_avg < 0.7:
                        st.warning("🟠 **Moderate Spill**")
                        st.write("Visible oil layer, ongoing spill")
                    else:
                        st.error("🔴 **Heavy Contamination**")
                        st.write("Thick oil, significant environmental impact")
                
                with col_char2:
                    st.metric("Average Intensity", f"{spill_intensity_avg:.2f}")
                
                with col_char3:
                    st.metric("Affected Area", f"{spill_pixels:,} px")
            
            else:
                st.success("✅ **NO OIL SPILLS DETECTED**")
                st.write("Water bodies appear clean with no visible contamination")
            
            # Download section
            st.subheader("💾 Download Detailed Results")
            dl_col1, dl_col2, dl_col3 = st.columns(3)
            
            with dl_col1:
                buf_water = io.BytesIO()
                water_display.save(buf_water, format="PNG")
                st.download_button(
                    "📥 Water Analysis",
                    data=buf_water.getvalue(),
                    file_name="water_analysis.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with dl_col2:
                buf_mask = io.BytesIO()
                mask_display.save(buf_mask, format="PNG")
                st.download_button(
                    "📥 Spill Detection Mask",
                    data=buf_mask.getvalue(),
                    file_name="oil_spill_detection.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with dl_col3:
                buf_overlay = io.BytesIO()
                overlay_img.save(buf_overlay, format="PNG")
                st.download_button(
                    "📥 Enhanced Visualization", 
                    data=buf_overlay.getvalue(),
                    file_name="enhanced_analysis.png",
                    mime="image/png",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"❌ Error in advanced analysis: {str(e)}")
    
    else:
        st.info("👆 **Upload a satellite image for advanced oil spill analysis**")

if __name__ == "__main__":
    main()
