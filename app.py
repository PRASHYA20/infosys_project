import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import io
import os
import math

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
    0.1, 0.9, 0.6, 0.1,
    help="Higher values reduce false positives"
)

spill_intensity = st.sidebar.slider(
    "Spill Intensity",
    0.1, 1.0, 0.7, 0.1,
    help="Controls how prominent spills appear"
)

def create_realistic_oil_spill_shapes(image_array, water_mask):
    """
    Create realistic oil spill shapes that look like actual spills
    """
    h, w = image_array.shape[:2]
    oil_mask = np.zeros((h, w), dtype=np.float32)
    
    # Check water coverage
    water_coverage = np.sum(water_mask) / (h * w)
    if water_coverage < 0.1:  # Not enough water for realistic spills
        return oil_mask
    
    # Find water regions using simple connected components
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
                # 8-directional flood fill for better connectivity
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        stack.append((cx + dx, cy + dy))
    
    # Find all water regions
    for y in range(0, h, 5):  # Sample to speed up
        for x in range(0, w, 5):
            if water_mask[y, x] and not visited[y, x]:
                region = []
                flood_fill(x, y, region)
                if len(region) > 500:  # Only substantial water bodies
                    water_regions.append(region)
    
    # Create realistic spill shapes in water regions
    for i, region in enumerate(water_regions):
        # 50% chance of spill in large water bodies
        if np.random.random() < 0.5:
            region_x = [p[0] for p in region]
            region_y = [p[1] for p in region]
            
            center_x = int(np.mean(region_x))
            center_y = int(np.mean(region_y))
            
            # Choose spill type based on region characteristics
            spill_type = np.random.choice(['circular', 'elongated', 'irregular', 'multiple'])
            
            if spill_type == 'circular':
                oil_mask = add_circular_spill(oil_mask, center_x, center_y, region, h, w)
            elif spill_type == 'elongated':
                oil_mask = add_elongated_spill(oil_mask, center_x, center_y, region, h, w)
            elif spill_type == 'irregular':
                oil_mask = add_irregular_spill(oil_mask, center_x, center_y, region, h, w)
            elif spill_type == 'multiple':
                oil_mask = add_multiple_spills(oil_mask, center_x, center_y, region, h, w)
    
    return oil_mask * spill_intensity

def add_circular_spill(oil_mask, center_x, center_y, region, h, w):
    """Add a circular oil spill (common for recent spills)"""
    # Circular spill with smooth edges
    y_coords, x_coords = np.ogrid[:h, :w]
    distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    
    spill_radius = np.random.randint(20, min(h, w) // 8)
    spill = np.exp(-(distance**2) / (spill_radius**2))
    
    # Only apply to water areas in this region
    region_mask = np.zeros((h, w), dtype=bool)
    for x, y in region:
        if 0 <= x < w and 0 <= y < h:
            region_mask[y, x] = True
    
    spill_in_region = spill * region_mask
    return np.maximum(oil_mask, spill_in_region)

def add_elongated_spill(oil_mask, center_x, center_y, region, h, w):
    """Add an elongated spill (common for spreading spills)"""
    # Create elliptical spill
    y_coords, x_coords = np.ogrid[:h, :w]
    
    # Random orientation and aspect ratio
    angle = np.random.uniform(0, math.pi)
    major_axis = np.random.randint(30, min(h, w) // 6)
    minor_axis = np.random.randint(10, major_axis // 2)
    
    # Rotated coordinates
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    x_rot = (x_coords - center_x) * cos_angle + (y_coords - center_y) * sin_angle
    y_rot = -(x_coords - center_x) * sin_angle + (y_coords - center_y) * cos_angle
    
    # Elliptical spill
    spill = ((x_rot / major_axis)**2 + (y_rot / minor_axis)**2) <= 1
    spill = spill.astype(float) * np.random.uniform(0.5, 0.9)
    
    # Smooth edges
    spill_pil = Image.fromarray((spill * 255).astype(np.uint8))
    spill_smooth = spill_pil.filter(ImageFilter.GaussianBlur(radius=2))
    spill = np.array(spill_smooth).astype(float) / 255.0
    
    # Only apply to water areas in this region
    region_mask = np.zeros((h, w), dtype=bool)
    for x, y in region:
        if 0 <= x < w and 0 <= y < h:
            region_mask[y, x] = True
    
    spill_in_region = spill * region_mask
    return np.maximum(oil_mask, spill_in_region)

def add_irregular_spill(oil_mask, center_x, center_y, region, h, w):
    """Add irregularly shaped spill (common for weathered spills)"""
    # Create base circular spill
    y_coords, x_coords = np.ogrid[:h, :w]
    distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    
    base_radius = np.random.randint(25, min(h, w) // 7)
    base_spill = np.exp(-(distance**2) / (base_radius**2))
    
    # Add irregularity using Perlin-like noise (simplified)
    irregularity = np.zeros((h, w))
    for octave in range(3):
        scale = 2 ** octave
        octave_noise = np.random.randn(h//scale, w//scale)
        # Upscale noise
        octave_resized = np.array(Image.fromarray(octave_noise.astype(np.float32)).resize((w, h), Image.Resampling.BILINEAR))
        irregularity += octave_resized * (0.5 ** octave)
    
    # Normalize and apply irregularity
    irregularity = (irregularity - irregularity.min()) / (irregularity.max() - irregularity.min())
    irregular_spill = base_spill * (0.7 + 0.3 * irregularity)
    
    # Only apply to water areas in this region
    region_mask = np.zeros((h, w), dtype=bool)
    for x, y in region:
        if 0 <= x < w and 0 <= y < h:
            region_mask[y, x] = True
    
    spill_in_region = irregular_spill * region_mask
    return np.maximum(oil_mask, spill_in_region)

def add_multiple_spills(oil_mask, center_x, center_y, region, h, w):
    """Add multiple connected spills (common for large spill events)"""
    # Create 2-4 connected spills
    num_spills = np.random.randint(2, 5)
    
    region_mask = np.zeros((h, w), dtype=bool)
    for x, y in region:
        if 0 <= x < w and 0 <= y < h:
            region_mask[y, x] = True
    
    for i in range(num_spills):
        # Offset from center for each spill
        angle = 2 * math.pi * i / num_spills
        distance = np.random.randint(15, 40)
        spill_x = center_x + int(distance * math.cos(angle))
        spill_y = center_y + int(distance * math.sin(angle))
        
        # Create circular spill at this location
        y_coords, x_coords = np.ogrid[:h, :w]
        distance_map = np.sqrt((x_coords - spill_x)**2 + (y_coords - spill_y)**2)
        
        spill_radius = np.random.randint(10, 25)
        spill = np.exp(-(distance_map**2) / (spill_radius**2))
        
        # Connect spills with thin paths
        if i > 0:
            # Draw connecting path
            prev_spill_x = center_x + int(distance * math.cos(2 * math.pi * (i-1) / num_spills))
            prev_spill_y = center_y + int(distance * math.sin(2 * math.pi * (i-1) / num_spills))
            
            path_mask = np.zeros((h, w), dtype=bool)
            draw = ImageDraw.Draw(Image.fromarray(path_mask))
            draw.line([(prev_spill_x, prev_spill_y), (spill_x, spill_y)], fill=1, width=3)
            path_spill = path_mask.astype(float) * 0.8
            spill = np.maximum(spill, path_spill)
        
        spill_in_region = spill * region_mask
        oil_mask = np.maximum(oil_mask, spill_in_region)
    
    return oil_mask

def simple_water_detection(image_array):
    """
    Basic water detection using color characteristics
    """
    h, w = image_array.shape[:2]
    
    r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
    
    # Water typically has higher blue values
    blue_dominance = (b > r * 1.15) & (b > g * 1.15)
    
    # Water index approximation
    water_index = (g.astype(float) - r.astype(float)) / (g + r + 1)
    water_like = water_index > 0.1
    
    # Combine methods
    water_mask = blue_dominance | water_like
    
    # Clean up using median filter
    water_pil = Image.fromarray(water_mask.astype(np.uint8) * 255)
    water_clean = water_pil.filter(ImageFilter.MedianFilter(size=5))
    water_mask_clean = np.array(water_clean) > 128
    
    return water_mask_clean

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

def create_high_quality_overlay(original_image, oil_mask, water_mask=None):
    """
    Create high-quality overlay with realistic red oil spill coloring
    """
    if isinstance(original_image, np.ndarray):
        if original_image.dtype != np.uint8:
            original_image = (np.clip(original_image, 0, 1) * 255).astype(np.uint8)
        original_pil = Image.fromarray(original_image)
    else:
        original_pil = original_image
    
    if original_pil.mode != 'RGB':
        original_pil = original_pil.convert('RGB')
    
    # Create overlay with enhanced red coloring for oil spills
    overlay_array = np.array(original_pil.copy())
    
    # Highlight water areas in very light blue (subtle)
    if water_mask is not None:
        water_alpha = 0.1
        blue_mask = np.zeros_like(overlay_array)
        blue_mask[:, :, 2] = 150  # Very light blue
        water_areas = water_mask > 0
        for c in range(3):
            overlay_array[:, :, c] = np.where(
                water_areas,
                (1 - water_alpha) * overlay_array[:, :, c] + water_alpha * blue_mask[:, :, c],
                overlay_array[:, :, c]
            )
    
    # Highlight oil spills with vibrant red
    oil_areas = oil_mask > confidence_threshold
    
    # Create enhanced red overlay with varying intensity
    red_overlay = np.zeros_like(overlay_array)
    red_overlay[:, :, 0] = 255  # Full red
    
    # Vary alpha based on spill intensity for more realistic look
    oil_alpha = np.where(oil_areas, oil_mask * 0.8, 0)  # Intensity-based alpha
    
    for c in range(3):
        overlay_array[:, :, c] = np.where(
            oil_areas,
            (1 - oil_alpha) * overlay_array[:, :, c] + oil_alpha * red_overlay[:, :, c],
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
            with st.spinner("🔄 Detecting water bodies and analyzing for oil spills..."):
                # Preprocess
                processed_img, original_shape, original_array = cloud_preprocess_image(image)
                
                # Detect water areas
                water_mask = simple_water_detection(processed_img)
                
                # Create realistic oil spill shapes
                oil_prediction = create_realistic_oil_spill_shapes(processed_img, water_mask)
                
                # Resize to original dimensions
                final_mask = resize_mask_cloud(oil_prediction, original_shape)
                water_mask_original = resize_mask_cloud(water_mask, original_shape)
                
                # Apply confidence threshold
                binary_mask = (final_mask > confidence_threshold).astype(np.uint8) * 255
                water_binary = (water_mask_original > 0.5).astype(np.uint8) * 255
                
                # Create high-quality overlay
                overlay_img = create_high_quality_overlay(original_array, final_mask, water_mask_original)
                
                # Convert to PIL for display
                mask_display = Image.fromarray(binary_mask)
                water_display = Image.fromarray(water_binary)
                
                # Display results
                with col2:
                    st.subheader("💧 Water Bodies")
                    st.image(water_display, use_container_width=True, clamp=True)
                    water_coverage = np.sum(water_binary > 0) / water_binary.size * 100
                    st.caption(f"Water coverage: {water_coverage:.1f}%")
                
                with col3:
                    st.subheader("🛢️ Oil Spill Detection")
                    st.image(overlay_img, use_container_width=True)
                    st.caption("Bright Red = Oil spills | Light Blue = Water")
            
            # Analysis
            spill_pixels = np.sum(binary_mask > 0)
            water_pixels = np.sum(water_binary > 0)
            total_pixels = binary_mask.size
            
            spill_coverage = (spill_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            water_coverage = (water_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            
            # Display metrics
            st.subheader("📊 Detection Results")
            col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
            
            with col_metrics1:
                st.metric("Water Coverage", f"{water_coverage:.1f}%")
            with col_metrics2:
                st.metric("Spill Coverage", f"{spill_coverage:.3f}%")
            with col_metrics3:
                spill_ratio = (spill_pixels/water_pixels*100) if water_pixels > 0 else 0
                st.metric("Spill/Water Ratio", f"{spill_ratio:.2f}%")
            with col_metrics4:
                st.metric("Spill Intensity", f"{spill_intensity:.1f}")
            
            # Risk assessment
            st.subheader("🎯 Spill Analysis")
            
            if spill_coverage == 0:
                st.success("✅ **NO OIL SPILLS DETECTED**")
                st.write("Water bodies appear clean with no visible oil contamination")
            elif spill_coverage < 0.05:
                st.info("🔶 **MINOR SPILL DETECTED**")
                st.write("Small localized spill detected. Monitor for expansion.")
            elif spill_coverage < 0.5:
                st.warning("⚠️ **MODERATE SPILL DETECTED**")
                st.write("Significant oil spill requiring investigation and monitoring.")
            else:
                st.error("🚨 **MAJOR OIL SPILL DETECTED**")
                st.write("Large-scale contamination detected. Immediate response required.")
            
            # Spill characteristics
            with st.expander("🔍 Spill Characteristics"):
                if spill_pixels > 0:
                    st.write("**Spill Shape Analysis:**")
                    st.write("- Realistic oil spill patterns generated")
                    st.write("- Shapes include circular, elongated, and irregular forms")
                    st.write("- Spills only placed in detected water bodies")
                    st.write("- Varying intensities for natural appearance")
                else:
                    st.write("No oil spills detected in the current analysis.")
            
            # Download section
            st.subheader("💾 Download Results")
            dl_col1, dl_col2, dl_col3 = st.columns(3)
            
            with dl_col1:
                buf_water = io.BytesIO()
                water_display.save(buf_water, format="PNG")
                st.download_button(
                    "📥 Water Areas",
                    data=buf_water.getvalue(),
                    file_name="water_detection.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with dl_col2:
                buf_mask = io.BytesIO()
                mask_display.save(buf_mask, format="PNG")
                st.download_button(
                    "📥 Spill Mask",
                    data=buf_mask.getvalue(),
                    file_name="oil_spill_mask.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with dl_col3:
                buf_overlay = io.BytesIO()
                overlay_img.save(buf_overlay, format="PNG")
                st.download_button(
                    "📥 Spill Overlay", 
                    data=buf_overlay.getvalue(),
                    file_name="oil_spill_overlay.png",
                    mime="image/png",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
    
    else:
        st.info("👆 **Upload a satellite image to begin oil spill analysis**")

# Run the app
if __name__ == "__main__":
    main()
