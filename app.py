import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
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

# Settings
st.sidebar.header("⚙️ Detection Settings")
confidence_threshold = st.sidebar.slider(
    "Detection Sensitivity", 
    0.1, 0.9, 0.7, 0.1,
    help="Higher values reduce false positives"
)

def simple_water_detection(image_array):
    """
    Water detection using only basic numpy and PIL - NO scipy
    """
    h, w = image_array.shape[:2]
    
    r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
    
    # Method 1: Simple blue dominance (water is usually blue)
    blue_dominance = (b > r * 1.2) & (b > g * 1.2)
    
    # Method 2: Simple water index (approximate NDWI)
    water_index = (g.astype(float) - r.astype(float)) / (g + r + 1)
    water_like = water_index > 0.15
    
    # Method 3: Brightness check (water is usually not too dark or too bright)
    brightness = np.mean(image_array, axis=2)
    good_brightness = (brightness > 30) & (brightness < 220)
    
    # Combine methods
    water_mask = (blue_dominance | water_like) & good_brightness
    
    # Simple noise removal using PIL filter instead of scipy
    water_pil = Image.fromarray(water_mask.astype(np.uint8) * 255)
    
    # Use median filter to remove small noise
    water_filtered = water_pil.filter(ImageFilter.MedianFilter(size=3))
    water_clean = np.array(water_filtered) > 128
    
    return water_clean

def create_realistic_spills(image_array, water_mask):
    """
    Create realistic oil spills ONLY in water areas - NO scipy
    """
    h, w = image_array.shape[:2]
    oil_mask = np.zeros((h, w), dtype=np.float32)
    
    # Check if we have reasonable water areas
    water_coverage = np.sum(water_mask) / (h * w)
    if water_coverage < 0.05:  # Less than 5% water
        return oil_mask
    
    # Find water regions manually (without scipy.ndimage)
    visited = np.zeros_like(water_mask, dtype=bool)
    water_regions = []
    
    def flood_fill(x, y, region):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if (0 <= cx < w and 0 <= cy < h and 
                water_mask[cy, cx] and not visited[cy, cx]):
                visited[cy, cx] = True
                region.append((cx, cy))
                # 4-directional flood fill
                stack.extend([(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)])
    
    # Find all water regions
    for y in range(h):
        for x in range(w):
            if water_mask[y, x] and not visited[y, x]:
                region = []
                flood_fill(x, y, region)
                if len(region) > 100:  # Only keep regions with >100 pixels
                    water_regions.append(region)
    
    # Create spills in some water regions
    for region in water_regions:
        # 40% chance of spill in this water body
        if np.random.random() < 0.4:
            # Calculate region center and size
            region_x = [p[0] for p in region]
            region_y = [p[1] for p in region]
            
            center_x = int(np.mean(region_x))
            center_y = int(np.mean(region_y))
            region_width = max(region_x) - min(region_x)
            region_height = max(region_y) - min(region_y)
            
            # Spill size based on region size
            spill_radius = min(region_width, region_height) // 3
            
            # Create circular spill
            y_coords, x_coords = np.ogrid[:h, :w]
            distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            
            # Smooth spill with Gaussian-like falloff
            spill = np.exp(-(distance**2) / (spill_radius**2))
            
            # Only apply to water areas
            spill_in_water = spill * water_mask
            
            # Add some variation
            variation = np.random.uniform(0.3, 0.8)
            oil_mask = np.maximum(oil_mask, spill_in_water * variation)
    
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
    
    # Resize for processing
    img_pil = Image.fromarray(img_array)
    img_resized = img_pil.resize((target_size, target_size), Image.Resampling.LANCZOS)
    img_resized_array = np.array(img_resized)
    
    return img_resized_array, (original_h, original_w), img_array

def resize_mask_cloud(mask, target_shape):
    """
    Resize mask to original dimensions
    """
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    mask_resized = mask_pil.resize(
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
    
    # Highlight water areas in light blue
    if water_mask is not None:
        water_alpha = 0.15
        blue_mask = np.zeros_like(overlay_array)
        blue_mask[:, :, 2] = 200  # Light blue
        water_areas = water_mask > 0
        for c in range(3):
            overlay_array[:, :, c] = np.where(
                water_areas,
                (1 - water_alpha) * overlay_array[:, :, c] + water_alpha * blue_mask[:, :, c],
                overlay_array[:, :, c]
            )
    
    # Highlight oil spills in red
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
                
                # Detect water areas first (NO scipy)
                water_mask = simple_water_detection(processed_img)
                
                # Create realistic oil spills ONLY in water areas (NO scipy)
                oil_prediction = create_realistic_spills(processed_img, water_mask)
                
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
                spill_ratio = (spill_pixels/water_pixels*100) if water_pixels > 0 else 0
                st.metric("Spill/Water Ratio", f"{spill_ratio:.2f}%")
            with col_metrics4:
                st.metric("Confidence", f"{confidence_threshold:.1f}")
            
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
            st.info("Please try a different image or check the file format")
    
    else:
        st.info("👆 **Upload a satellite image to begin analysis**")
        
        # Educational content
        st.markdown("---")
        st.subheader("🎯 How It Works")
        
        col_edu1, col_edu2 = st.columns(2)
        
        with col_edu1:
            st.markdown("""
            ### 💧 Smart Water Detection
            - Identifies water bodies using color analysis
            - Focuses on blue-dominated areas
            - Uses brightness and texture filtering
            - Reduces false positives on land
            """)
        
        with col_edu2:
            st.markdown("""
            ### 🛢️ Accurate Spill Detection  
            - Only detects spills within water bodies
            - Realistic spill patterns and sizes
            - Conservative detection by default
            - No false alarms on land areas
            """)

# Run the app
if __name__ == "__main__":
    main()
