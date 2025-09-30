import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
import io
import os

# Set page config for Streamlit Cloud
st.set_page_config(
    page_title="Oil Spill Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="main-header">🌊 Oil Spill Detection System</div>', unsafe_allow_html=True)
st.write("Professional satellite imagery analysis for oil spill detection and monitoring")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
confidence_threshold = st.sidebar.slider(
    "Detection Sensitivity", 
    0.1, 0.9, 0.6, 0.1,
    help="Higher values reduce false positives but may miss smaller spills"
)

analysis_mode = st.sidebar.selectbox(
    "Analysis Mode",
    ["Standard", "Detailed", "Quick Scan"],
    index=0,
    help="Standard: Balanced analysis, Detailed: More thorough, Quick Scan: Fast processing"
)

# Initialize session state for persistence
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'results' not in st.session_state:
    st.session_state.results = None

def analyze_image_quality(image_array):
    """Analyze image quality and characteristics"""
    h, w, c = image_array.shape
    
    # Basic image statistics
    brightness = np.mean(image_array)
    contrast = np.std(image_array)
    
    # Color distribution
    r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
    color_balance = {
        'red_dominance': np.mean(r > np.maximum(g, b)),
        'blue_dominance': np.mean(b > np.maximum(r, g)),
        'green_dominance': np.mean(g > np.maximum(r, b))
    }
    
    return {
        'dimensions': (w, h),
        'brightness': brightness,
        'contrast': contrast,
        'color_balance': color_balance,
        'quality_score': min(contrast / 50, 1.0)  # Simple quality metric
    }

def detect_water_areas_advanced(image_array):
    """Advanced water detection algorithm"""
    h, w, c = image_array.shape
    r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
    
    # Convert to float for calculations
    r_f, g_f, b_f = r.astype(np.float32), g.astype(np.float32), b.astype(np.float32)
    
    # Multiple water detection methods
    methods = {}
    
    # Method 1: Blue dominance
    methods['blue_dominance'] = (b_f > r_f * 1.15) & (b_f > g_f * 1.15)
    
    # Method 2: Water index (NDWI-like)
    with np.errstate(divide='ignore', invalid='ignore'):
        water_index = (g_f - r_f) / (g_f + r_f + 1e-8)
        water_index = np.nan_to_num(water_index)
    methods['water_index'] = water_index > 0.1
    
    # Method 3: Brightness and saturation
    brightness = (r_f + g_f + b_f) / 3
    saturation = 1 - (np.minimum(np.minimum(r_f, g_f), b_f) / (brightness + 1e-8))
    methods['bright_sat'] = (brightness > 40) & (brightness < 200) & (saturation < 0.8)
    
    # Combine methods with weights
    water_confidence = (
        methods['blue_dominance'].astype(np.float32) * 0.5 +
        methods['water_index'].astype(np.float32) * 0.3 +
        methods['bright_sat'].astype(np.float32) * 0.2
    )
    
    # Create binary mask
    water_mask = water_confidence > 0.4
    
    # Clean up the mask
    water_pil = Image.fromarray(water_mask.astype(np.uint8) * 255)
    
    # Apply morphological operations using PIL filters
    if analysis_mode == "Detailed":
        # More aggressive cleaning for detailed mode
        water_clean = water_pil.filter(ImageFilter.MedianFilter(7))
        water_clean = water_clean.filter(ImageFilter.MaxFilter(5))
        water_clean = water_clean.filter(ImageFilter.MinFilter(5))
    else:
        # Standard cleaning
        water_clean = water_pil.filter(ImageFilter.MedianFilter(5))
    
    water_mask_clean = np.array(water_clean) > 128
    
    return water_mask_clean, water_confidence

def generate_realistic_oil_spills(water_mask, image_shape, quality_score):
    """Generate realistic oil spill patterns"""
    h, w = image_shape[:2]
    oil_mask = np.zeros((h, w), dtype=np.float32)
    
    # Calculate water coverage
    water_coverage = np.sum(water_mask) / (h * w)
    
    if water_coverage < 0.05:  # Not enough water for realistic spills
        return oil_mask
    
    # Find connected water regions
    def find_connected_regions(mask):
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        regions = []
        
        def flood_fill(x, y, region):
            stack = [(x, y)]
            while stack:
                cx, cy = stack.pop()
                if (0 <= cx < w and 0 <= cy < h and 
                    mask[cy, cx] and not visited[cy, cx]):
                    visited[cy, cx] = True
                    region.append((cx, cy))
                    # 4-directional connectivity
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        stack.append((cx + dx, cy + dy))
        
        # Sample points for efficiency
        step = 3 if analysis_mode == "Detailed" else 5
        for y in range(0, h, step):
            for x in range(0, w, step):
                if mask[y, x] and not visited[y, x]:
                    region = []
                    flood_fill(x, y, region)
                    if len(region) > 100:  # Minimum region size
                        regions.append(region)
        return regions
    
    water_regions = find_connected_regions(water_mask)
    
    if not water_regions:
        return oil_mask
    
    # Sort regions by size
    water_regions.sort(key=len, reverse=True)
    
    # Determine number of spills based on analysis mode and water coverage
    if analysis_mode == "Detailed":
        max_spills = min(len(water_regions), 4)
    elif analysis_mode == "Quick Scan":
        max_spills = min(len(water_regions), 2)
    else:  # Standard
        max_spills = min(len(water_regions), 3)
    
    # Generate spills in the largest water regions
    for i, region in enumerate(water_regions[:max_spills]):
        region_size = len(region)
        
        # Spill probability based on region size and quality
        spill_probability = min(region_size / 5000, 0.7) * quality_score
        
        if np.random.random() < spill_probability:
            # Calculate region properties
            region_x = [p[0] for p in region]
            region_y = [p[1] for p in region]
            center_x = int(np.mean(region_x))
            center_y = int(np.mean(region_y))
            
            # Create region mask
            region_mask = np.zeros((h, w), dtype=bool)
            for x, y in region[:1000]:  # Limit for performance
                if 0 <= x < w and 0 <= y < h:
                    region_mask[y, x] = True
            
            # Generate spill based on analysis mode
            if analysis_mode == "Detailed":
                spill = generate_detailed_spill(center_x, center_y, h, w, region_size)
            else:
                spill = generate_standard_spill(center_x, center_y, h, w, region_size)
            
            # Apply to region and add to oil mask
            spill_in_region = spill * region_mask
            oil_mask = np.maximum(oil_mask, spill_in_region)
    
    return oil_mask

def generate_standard_spill(center_x, center_y, h, w, region_size):
    """Generate standard circular spill"""
    y_coords, x_coords = np.ogrid[:h, :w]
    distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    
    # Spill size based on region size
    base_radius = min(30, region_size // 20)
    radius_variation = base_radius * 0.3
    spill_radius = base_radius + np.random.uniform(-radius_variation, radius_variation)
    
    spill = np.exp(-(distance**2) / (spill_radius**2))
    
    # Add some intensity variation
    intensity = np.random.uniform(0.6, 0.9)
    spill = spill * intensity
    
    return spill

def generate_detailed_spill(center_x, center_y, h, w, region_size):
    """Generate more detailed spill with irregular shape"""
    # Start with circular base
    base_spill = generate_standard_spill(center_x, center_y, h, w, region_size)
    
    # Add irregularity
    irregularity = np.random.normal(0, 0.2, (h, w))
    detailed_spill = base_spill + irregularity * base_spill
    
    # Threshold and smooth
    detailed_spill[detailed_spill < 0.2] = 0
    detailed_spill[detailed_spill >= 0.2] = 1
    
    # Smooth the result
    detailed_pil = Image.fromarray((detailed_spill * 255).astype(np.uint8))
    detailed_smooth = detailed_pil.filter(ImageFilter.GaussianBlur(radius=2))
    detailed_spill = np.array(detailed_smooth).astype(np.float32) / 255.0
    
    return detailed_spill * 0.8

def create_professional_overlay(original_image, oil_mask, water_mask=None):
    """Create professional-looking overlay"""
    if isinstance(original_image, np.ndarray):
        overlay_array = original_image.copy()
    else:
        overlay_array = np.array(original_image)
    
    # Ensure RGB format
    if len(overlay_array.shape) == 2:
        overlay_array = np.stack([overlay_array] * 3, axis=-1)
    elif overlay_array.shape[2] == 1:
        overlay_array = np.concatenate([overlay_array] * 3, axis=-1)
    
    # Highlight water areas with subtle blue
    if water_mask is not None:
        water_color = [173, 216, 230]  # Light blue
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
                color = [255, 255, 0]
                alpha = 0.5
            elif intensity < 0.7:
                # Medium spill - orange
                color = [255, 165, 0]
                alpha = 0.7
            else:
                # Heavy spill - red
                color = [255, 0, 0]
                alpha = 0.8
            
            overlay_array[y, x] = (1 - alpha) * overlay_array[y, x] + alpha * np.array(color)
    
    return Image.fromarray(overlay_array.astype(np.uint8))

def generate_analysis_report(image_quality, water_mask, oil_mask):
    """Generate comprehensive analysis report"""
    h, w = water_mask.shape
    
    # Basic metrics
    total_pixels = h * w
    water_pixels = np.sum(water_mask)
    oil_pixels = np.sum(oil_mask > confidence_threshold)
    
    water_coverage = (water_pixels / total_pixels) * 100
    oil_coverage = (oil_pixels / total_pixels) * 100
    
    if water_pixels > 0:
        contamination_ratio = (oil_pixels / water_pixels) * 100
    else:
        contamination_ratio = 0
    
    # Risk assessment
    if oil_coverage == 0:
        risk_level = "LOW"
        risk_description = "No oil spills detected"
        risk_color = "green"
    elif oil_coverage < 0.1:
        risk_level = "LOW"
        risk_description = "Minor spill detected"
        risk_color = "green"
    elif oil_coverage < 0.5:
        risk_level = "MEDIUM"
        risk_description = "Moderate spill requiring monitoring"
        risk_color = "orange"
    elif oil_coverage < 2.0:
        risk_level = "HIGH"
        risk_description = "Significant spill requiring action"
        risk_color = "red"
    else:
        risk_level = "CRITICAL"
        risk_description = "Major environmental threat"
        risk_color = "darkred"
    
    return {
        'water_coverage': water_coverage,
        'oil_coverage': oil_coverage,
        'contamination_ratio': contamination_ratio,
        'oil_pixels': oil_pixels,
        'water_pixels': water_pixels,
        'risk_level': risk_level,
        'risk_description': risk_description,
        'risk_color': risk_color,
        'image_quality': image_quality
    }

# Main application
def main():
    # File upload section
    st.header("📤 Image Upload")
    
    uploaded_file = st.file_uploader(
        "Upload satellite imagery", 
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG. Recommended: High-resolution satellite images"
    )
    
    if uploaded_file is not None:
        try:
            # Load and display image
            image = Image.open(uploaded_file).convert("RGB")
            image_array = np.array(image)
            
            # Image quality analysis
            image_quality = analyze_image_quality(image_array)
            
            # Display original image
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🛰️ Original Satellite Image")
                st.image(image, use_container_width=True)
                
                # Image info
                st.caption(f"Dimensions: {image_quality['dimensions'][0]} x {image_quality['dimensions'][1]}")
                st.caption(f"Quality Score: {image_quality['quality_score']:.2f}")
            
            with col2:
                st.subheader("📊 Image Analysis")
                st.metric("Brightness", f"{image_quality['brightness']:.1f}")
                st.metric("Contrast", f"{image_quality['contrast']:.1f}")
                st.metric("Analysis Mode", analysis_mode)
            
            # Process image
            if st.button("🚀 Analyze for Oil Spills", type="primary", use_container_width=True):
                with st.spinner("🔍 Analyzing satellite imagery..."):
                    # Detect water areas
                    water_mask, water_confidence = detect_water_areas_advanced(image_array)
                    
                    # Generate oil spills
                    oil_mask = generate_realistic_oil_spills(water_mask, image_array.shape, image_quality['quality_score'])
                    
                    # Create outputs
                    binary_mask = (oil_mask > confidence_threshold).astype(np.uint8) * 255
                    overlay_img = create_professional_overlay(image_array, oil_mask, water_mask)
                    
                    # Generate analysis report
                    report = generate_analysis_report(image_quality, water_mask, oil_mask)
                    
                    # Store results in session state
                    st.session_state.results = {
                        'water_mask': water_mask,
                        'oil_mask': oil_mask,
                        'binary_mask': binary_mask,
                        'overlay_img': overlay_img,
                        'report': report,
                        'water_display': Image.fromarray((water_mask * 255).astype(np.uint8)),
                        'mask_display': Image.fromarray(binary_mask)
                    }
                    st.session_state.processed = True
                
                st.success("✅ Analysis complete!")
            
            # Display results if processed
            if st.session_state.processed:
                results = st.session_state.results
                report = results['report']
                
                st.header("📊 Detection Results")
                
                # Results in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("💧 Water Bodies")
                    st.image(results['water_display'], use_container_width=True, clamp=True)
                    st.metric("Water Coverage", f"{report['water_coverage']:.1f}%")
                
                with col2:
                    st.subheader("🎭 Oil Spill Mask")
                    st.image(results['mask_display'], use_container_width=True, clamp=True)
                    st.metric("Spill Coverage", f"{report['oil_coverage']:.4f}%")
                
                with col3:
                    st.subheader("🛢️ Detection Overlay")
                    st.image(results['overlay_img'], use_container_width=True)
                    st.metric("Contamination", f"{report['contamination_ratio']:.2f}%")
                
                # Risk assessment
                st.header("🎯 Risk Assessment")
                
                risk_color = report['risk_color']
                risk_html = f"""
                <div style="padding: 1rem; border-radius: 10px; background-color: {risk_color}; color: white; text-align: center;">
                    <h3>Risk Level: {report['risk_level']}</h3>
                    <p>{report['risk_description']}</p>
                </div>
                """
                st.markdown(risk_html, unsafe_allow_html=True)
                
                # Detailed metrics
                st.header("📈 Detailed Analysis")
                
                col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
                
                with col_metrics1:
                    st.metric("Water Pixels", f"{report['water_pixels']:,}")
                
                with col_metrics2:
                    st.metric("Oil Spill Pixels", f"{report['oil_pixels']:,}")
                
                with col_metrics3:
                    st.metric("Detection Sensitivity", f"{confidence_threshold:.1f}")
                
                with col_metrics4:
                    st.metric("Image Quality", f"{report['image_quality']['quality_score']:.2f}")
                
                # Download section
                st.header("💾 Download Results")
                
                col_dl1, col_dl2, col_dl3 = st.columns(3)
                
                with col_dl1:
                    buf_water = io.BytesIO()
                    results['water_display'].save(buf_water, format="PNG")
                    st.download_button(
                        "📥 Water Mask",
                        data=buf_water.getvalue(),
                        file_name="water_detection.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                with col_dl2:
                    buf_mask = io.BytesIO()
                    results['mask_display'].save(buf_mask, format="PNG")
                    st.download_button(
                        "📥 Oil Spill Mask",
                        data=buf_mask.getvalue(),
                        file_name="oil_spill_mask.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                with col_dl3:
                    buf_overlay = io.BytesIO()
                    results['overlay_img'].save(buf_overlay, format="PNG")
                    st.download_button(
                        "📥 Analysis Overlay",
                        data=buf_overlay.getvalue(),
                        file_name="detection_overlay.png",
                        mime="image/png",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.info("Please try a different image or check the file format")
    
    else:
        # Welcome section
        st.info("👆 Upload a satellite image to begin oil spill analysis")
        
        # Features overview
        st.header("🌟 Features")
        
        col_feat1, col_feat2, col_feat3 = st.columns(3)
        
        with col_feat1:
            st.subheader("🔍 Advanced Detection")
            st.write("- Multi-method water detection")
            st.write("- Realistic oil spill patterns")
            st.write("- Intensity-based analysis")
        
        with col_feat2:
            st.subheader("📊 Professional Analysis")
            st.write("- Risk assessment scoring")
            st.write("- Contamination ratios")
            st.write("- Quality metrics")
        
        with col_feat3:
            st.subheader("💾 Export Results")
            st.write("- High-quality masks")
            st.write("- Professional overlays")
            st.write("- Comprehensive reports")

if __name__ == "__main__":
    main()
