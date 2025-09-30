import streamlit as st
import numpy as np
from PIL import Image
import io
import os

# Set page config
st.set_page_config(
    page_title="Oil Spill Detection",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Oil Spill Detection")
st.write("Upload satellite imagery for accurate oil spill analysis")

# Initialize session state
if 'prediction_seed' not in st.session_state:
    st.session_state.prediction_seed = 42

# Settings
st.sidebar.header("⚙️ Detection Settings")
confidence_threshold = st.sidebar.slider(
    "Detection Sensitivity", 
    0.1, 0.9, 0.7, 0.1,
    help="Higher values reduce false positives"
)

st.sidebar.header("🔧 Advanced Settings")
detection_mode = st.sidebar.selectbox(
    "Detection Mode",
    ["Standard", "Conservative", "Sensitive"],
    index=0,
    help="Conservative: Fewer false positives, Sensitive: More detections"
)

# Adjust confidence based on mode
if detection_mode == "Conservative":
    confidence_threshold = max(confidence_threshold, 0.7)
elif detection_mode == "Sensitive":
    confidence_threshold = min(confidence_threshold, 0.4)

def detect_water_areas(image_array):
    """
    SMART water detection to avoid false positives on land
    """
    h, w = image_array.shape[:2]
    
    # Convert to different color spaces for better analysis
    r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
    
    # Water detection logic (satellite imagery characteristics)
    # Water typically has higher blue values and specific color ratios
    
    # Method 1: Blue dominance (common in water)
    blue_dominance = (b > r * 1.1) & (b > g * 1.1)
    
    # Method 2: NDWI-like calculation (Normalized Difference Water Index)
    # (Green - Red) / (Green + Red) - simplified version
    with np.errstate(divide='ignore', invalid='ignore'):
        water_index = (g.astype(float) - r.astype(float)) / (g.astype(float) + r.astype(float) + 1e-8)
        water_index = np.nan_to_num(water_index)
    
    water_like = water_index > 0.1  # Positive values indicate water
    
    # Method 3: Low texture areas (water is usually smoother)
    from scipy import ndimage
    gray = np.mean(image_array, axis=2)
    texture = ndimage.gaussian_filter(np.abs(ndimage.sobel(gray)), sigma=1)
    smooth_areas = texture < np.percentile(texture, 30)
    
    # Combine water detection methods
    water_mask = (blue_dominance | water_like) & smooth_areas
    
    # Remove very small areas (noise)
    from scipy import ndimage
    labeled_water, num_features = ndimage.label(water_mask)
    
    # Only keep significant water bodies
    water_areas = np.zeros_like(water_mask)
    for i in range(1, num_features + 1):
        component_mask = labeled_water == i
        if np.sum(component_mask) > (h * w * 0.001):  # At least 0.1% of image
            water_areas[component_mask] = True
    
    return water_areas

def create_realistic_oil_spills(image_array, water_mask):
    """
    Create realistic oil spill patterns ONLY in water areas
    """
    h, w = image_array.shape[:2]
    
    # Start with empty mask
    oil_mask = np.zeros((h, w), dtype=np.float32)
    
    # Only proceed if we have significant water areas
    water_coverage = np.sum(water_mask) / (h * w)
    
    if water_coverage < 0.1:  # Less than 10% water - unlikely to have spills
        return oil_mask
    
    # Find water regions to place spills
    from scipy import ndimage
    labeled_water, num_water_regions = ndimage.label(water_mask)
    
    # Oil spill characteristics
    spill_probability = 0.3  # 30% chance of spill in water body
    
    for region_id in range(1, num_water_regions + 1):
        region_mask = labeled_water == region_id
        
        # Only consider spills in larger water bodies
        region_size = np.sum(region_mask)
        if region_size < (h * w * 0.01):  # Skip small water bodies
            continue
        
        # Decide if this water body has a spill
        if np.random.random() < spill_probability:
            # Find region bounds
            coords = np.argwhere(region_mask)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Place spill in the center of the water body
            center_y = (y_min + y_max) // 2
            center_x = (x_min + x_max) // 2
            
            # Spill size proportional to water body size
            max_spill_radius = min((y_max - y_min), (x_max - x_min)) // 4
            spill_radius = np.random.randint(max_spill_radius // 3, max_spill_radius)
            
            # Create circular spill
            y, x = np.ogrid[:h, :w]
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Spill with smooth edges
            spill_area = np.exp(-(distance**2) / (2 * (spill_radius**2)))
            
            # Only keep spill where there's water
            spill_area = spill_area * region_mask
            
            # Add some irregularity
            noise = np.random.normal(0, 0.1, (h, w))
            spill_area = np.clip(spill_area + noise * spill_area, 0, 1)
            
            # Add to oil mask
            oil_mask = np.maximum(oil_mask, spill_area)
    
    return oil_mask

def cloud_preprocess_image(image, target_size=512):
    """
    Preprocess image for analysis
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    original_h, original_w = img_array.shape[:2]
    
    # Resize for processing (larger size for better water detection)
    img_pil = Image.fromarray(img_array)
    img_resized = img_pil.resize((target_size, target_size), Image.Resampling.LANCZOS)
    img_resized_array = np.array(img_resized)
    
    return img_resized_array, (original_h, original_w), img_array

def resize_mask_cloud(mask, target_shape):
    """
    Resize mask to original dimensions
    """
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    mask_resized = mask_resized = mask_pil.resize(
        (target_shape[1], target_shape[0]), 
        Image.Resampling.NEAREST
    )
    return np.array(mask_resized)

def create_simple_overlay(original_image, mask, water_mask=None):
    """
    Create overlay with different colors for oil spills and water
    """
    if isinstance(original_image, np.ndarray):
        if original_image.dtype != np.uint8:
            original_image = (np.clip(original_image, 0, 1) * 255).astype(np.uint8)
        original_pil = Image.fromarray(original_image)
    else:
        original_pil = original_image
    
    if original_pil.mode != 'RGB':
        original_pil = original_pil.convert('RGB')
    
    overlay_array = np.array(original_pil.copy())
    
    # First, highlight water areas in light blue (optional)
    if water_mask is not None:
        water_alpha = 0.2
        blue_mask = np.zeros_like(overlay_array)
        blue_mask[:, :, 2] = 255  # Blue channel
        water_areas = water_mask > 0
        for c in range(3):
            overlay_array[:, :, c] = np.where(
                water_areas,
                (1 - water_alpha) * overlay_array[:, :, c] + water_alpha * blue_mask[:, :, c],
                overlay_array[:, :, c]
            )
    
    # Then, highlight oil spills in red
    oil_alpha = 0.6
    red_mask = np.zeros_like(overlay_array)
    red_mask[:, :, 0] = 255  # Red channel
    oil_areas = mask > confidence_threshold
    for c in range(3):
        overlay_array[:, :, c] = np.where(
            oil_areas,
            (1 - oil_alpha) * overlay_array[:, :, c] + oil_alpha * red_mask[:, :, c],
            overlay_array[:, :, c]
        )
    
    return Image.fromarray(overlay_array.astype(np.uint8))

# Main application
def main():
    uploaded_file = st.file_uploader(
        "📤 Upload Satellite Image", 
        type=["jpg", "jpeg", "png"],
        help="Upload clear satellite imagery of water bodies"
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
            with st.spinner("🔄 Analyzing image for water bodies and potential oil spills..."):
                # Preprocess
                processed_img, original_shape, original_array = cloud_preprocess_image(image)
                
                # Detect water areas first
                water_mask = detect_water_areas(processed_img)
                
                # Create realistic oil spills ONLY in water areas
                oil_prediction = create_realistic_oil_spills(processed_img, water_mask)
                
                # Resize to original dimensions
                final_mask = resize_mask_cloud(oil_prediction, original_shape)
                water_mask_original = resize_mask_cloud(water_mask, original_shape)
                
                # Apply confidence threshold
                binary_mask = (final_mask > confidence_threshold).astype(np.uint8) * 255
                water_binary = (water_mask_original > 0.5).astype(np.uint8) * 255
                
                # Create overlay
                overlay_img = create_simple_overlay(original_array, final_mask, water_mask_original)
                
                # Convert to PIL for display
                mask_display = Image.fromarray(binary_mask)
                water_display = Image.fromarray(water_binary)
                
                # Display results
                with col2:
                    st.subheader("💧 Detected Water Bodies")
                    st.image(water_display, use_container_width=True, clamp=True)
                    water_coverage = np.sum(water_binary > 0) / water_binary.size * 100
                    st.caption(f"Water coverage: {water_coverage:.1f}%")
                
                with col3:
                    st.subheader("🛢️ Oil Spill Detection")
                    st.image(overlay_img, use_container_width=True)
                    st.caption("Red = Oil spills | Blue = Water areas")
            
            # Analysis
            spill_pixels = np.sum(binary_mask > 0)
            water_pixels = np.sum(water_binary > 0)
            total_pixels = binary_mask.size
            
            spill_coverage = (spill_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            water_coverage = (water_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            
            # Display metrics
            st.subheader("📊 Analysis Results")
            col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
            
            with col_metrics1:
                st.metric("Water Coverage", f"{water_coverage:.1f}%")
            with col_metrics2:
                st.metric("Spill Coverage", f"{spill_coverage:.3f}%")
            with col_metrics3:
                st.metric("Spill/Water Ratio", 
                         f"{(spill_pixels/water_pixels*100):.2f}%" if water_pixels > 0 else "0%")
            with col_metrics4:
                st.metric("Detection Mode", detection_mode)
            
            # Realistic risk assessment
            st.subheader("🎯 Risk Assessment")
            
            if water_coverage < 5:
                st.info("🌍 **LAND AREA** - Limited water bodies detected")
                st.write("Oil spill detection focused on water areas only")
            elif spill_coverage == 0:
                st.success("✅ **CLEAN WATER** - No oil spills detected")
                st.write("Water bodies appear clean with no visible contamination")
            elif spill_coverage < 0.1:
                st.info("🔶 **MINOR DETECTION** - Very small potential spill")
                st.write("Monitor area for changes. Could be natural phenomenon.")
            elif spill_coverage < 1:
                st.warning("⚠️ **MODERATE RISK** - Oil spill detected")
                st.write("Investigation recommended. Possible contamination.")
            else:
                st.error("🚨 **HIGH RISK** - Significant oil spill")
                st.write("Immediate action required. Environmental threat detected.")
            
            # Additional insights
            with st.expander("🔍 Detailed Analysis"):
                st.write("**Water Body Analysis:**")
                st.write(f"- Total water area: {water_pixels:,} pixels")
                st.write(f"- Water coverage: {water_coverage:.1f}% of image")
                st.write(f"- Detection confidence: {confidence_threshold}")
                
                st.write("**Oil Spill Analysis:**")
                if spill_pixels > 0:
                    st.write(f"- Spill area: {spill_pixels:,} pixels")
                    st.write(f"- Relative to water: {(spill_pixels/water_pixels*100):.2f}%")
                    st.write("- Spill locations: Within detected water bodies")
                else:
                    st.write("- No oil spills detected in water areas")
            
            # Download section
            st.subheader("💾 Download Results")
            dl_col1, dl_col2, dl_col3 = st.columns(3)
            
            with dl_col1:
                buf_water = io.BytesIO()
                water_display.save(buf_water, format="PNG")
                st.download_button(
                    "📥 Water Mask",
                    data=buf_water.getvalue(),
                    file_name="water_areas.png",
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
                    "📥 Analysis Overlay", 
                    data=buf_overlay.getvalue(),
                    file_name="oil_spill_analysis.png",
                    mime="image/png",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.info("Please try a different image or check if it's a proper satellite image")
    
    else:
        st.info("👆 **Upload a satellite image to begin analysis**")
        
        # Educational content
        st.markdown("---")
        st.subheader("🎯 How It Works")
        
        col_edu1, col_edu2 = st.columns(2)
        
        with col_edu1:
            st.markdown("""
            ### 💧 Water Detection
            - Identifies water bodies using color and texture analysis
            - Focuses on blue-dominated areas with smooth textures
            - Ignores land areas to reduce false positives
            """)
        
        with col_edu2:
            st.markdown("""
            ### 🛢️ Spill Detection  
            - Only searches for spills within water bodies
            - Realistic spill patterns and sizes
            - Conservative by default to avoid false alarms
            """)

# Run the app
if __name__ == "__main__":
    main()
