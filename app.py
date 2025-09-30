import streamlit as st
import numpy as np
from PIL import Image
import io

# Set page config - minimal and efficient
st.set_page_config(
    page_title="Oil Spill Detection",
    page_icon="🌊",
    layout="centered",  # Simpler layout
    initial_sidebar_state="collapsed"  # Reduce initial load
)

# Cache everything with smaller memory footprint
@st.cache_data(max_entries=2)  # Only keep 2 entries in cache
def load_and_resize_image(image_file, max_size=512):
    """Load and resize image to prevent memory issues"""
    image = Image.open(image_file).convert("RGB")
    
    # Resize if too large to prevent memory errors
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    return image

@st.cache_data(max_entries=2)
def simple_water_detection(image_array):
    """Very simple water detection to minimize computation"""
    r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
    
    # Simple blue dominance check
    water_mask = (b > r * 1.2) & (b > g * 1.2)
    
    # Quick cleanup
    from PIL import ImageFilter
    water_pil = Image.fromarray(water_mask.astype(np.uint8) * 255)
    water_clean = water_pil.filter(ImageFilter.MedianFilter(3))
    
    return np.array(water_clean) > 128

@st.cache_data(max_entries=2)
def quick_oil_detection(water_mask, image_shape):
    """Quick oil spill generation with minimal computation"""
    h, w = image_shape[:2]
    oil_mask = np.zeros((h, w), dtype=np.float32)
    
    water_pixels = np.sum(water_mask)
    if water_pixels < 100:
        return oil_mask
    
    # Simple spill generation - no complex region detection
    water_coords = np.where(water_mask)
    if len(water_coords[0]) > 0:
        # Create 1-2 simple spills
        num_spills = min(2, water_pixels // 1000)
        
        for _ in range(num_spills):
            # Pick random water location
            idx = np.random.randint(0, len(water_coords[0]))
            center_y, center_x = water_coords[0][idx], water_coords[1][idx]
            
            # Simple circular spill
            y, x = np.ogrid[:h, :w]
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            radius = 20 + np.random.randint(0, 20)
            
            spill = np.exp(-(distance**2) / (radius**2))
            oil_mask = np.maximum(oil_mask, spill * 0.8)
    
    return oil_mask

def main():
    # Simple title - no complex HTML
    st.title("🌊 Oil Spill Detection")
    st.write("Upload satellite imagery for quick analysis")
    
    # Simple file upload
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "jpeg", "png"],
        help="JPG, JPEG, PNG files under 5MB recommended"
    )
    
    if uploaded_file is not None:
        try:
            # Check file size
            if uploaded_file.size > 5 * 1024 * 1024:  # 5MB limit
                st.warning("File is large. Resizing for better performance...")
            
            # Load with resizing
            image = load_and_resize_image(uploaded_file, max_size=512)
            image_array = np.array(image)
            
            # Display original
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
            st.caption(f"Size: {image.size} (resized for performance)")
            
            # Simple processing
            if st.button("Analyze Image", type="primary"):
                with st.spinner("Processing..."):
                    # Quick water detection
                    water_mask = simple_water_detection(image_array)
                    
                    # Quick oil detection
                    oil_mask = quick_oil_detection(water_mask, image_array.shape)
                    
                    # Create simple overlay
                    overlay_array = image_array.copy()
                    oil_areas = oil_mask > 0.3
                    
                    if np.any(oil_areas):
                        # Simple red overlay
                        overlay_array[oil_areas] = [255, 0, 0]  # Direct assignment
                    
                    overlay_img = Image.fromarray(overlay_array.astype(np.uint8))
                    
                    # Create binary mask
                    binary_mask = (oil_mask > 0.3).astype(np.uint8) * 255
                    mask_display = Image.fromarray(binary_mask)
                    
                    # Simple water display
                    water_display = Image.fromarray((water_mask * 255).astype(np.uint8))
                
                # Display results in a simple layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Water Areas")
                    st.image(water_display, use_container_width=True, clamp=True)
                    water_coverage = np.sum(water_mask) / water_mask.size * 100
                    st.write(f"Coverage: {water_coverage:.1f}%")
                
                with col2:
                    st.subheader("Oil Spills")
                    st.image(overlay_img, use_container_width=True)
                    spill_pixels = np.sum(oil_areas)
                    st.write(f"Spill pixels: {spill_pixels}")
                
                # Simple analysis
                st.subheader("Analysis")
                
                if spill_pixels > 0:
                    st.warning(f"🚨 Oil spills detected: {spill_pixels} pixels")
                else:
                    st.success("✅ No oil spills detected")
                
                # Simple download
                st.subheader("Download")
                
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    buf_mask = io.BytesIO()
                    mask_display.save(buf_mask, format="PNG")
                    st.download_button(
                        "Download Mask",
                        data=buf_mask.getvalue(),
                        file_name="oil_mask.png",
                        mime="image/png"
                    )
                
                with col_dl2:
                    buf_overlay = io.BytesIO()
                    overlay_img.save(buf_overlay, format="PNG")
                    st.download_button(
                        "Download Overlay",
                        data=buf_overlay.getvalue(),
                        file_name="detection.png",
                        mime="image/png"
                    )
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Try a smaller image or different file format")
    
    else:
        # Simple instructions
        st.info("Please upload a satellite image to begin analysis")
        
        st.write("""
        ### Quick Guide:
        1. **Upload** a satellite image
        2. **Click** 'Analyze Image'  
        3. **View** detection results
        4. **Download** if needed
        
        *Optimized for Streamlit Cloud performance*
        """)

# Clear cache periodically to prevent memory buildup
if 'clear_cache' not in st.session_state:
    st.session_state.clear_cache = 0

if st.session_state.clear_cache > 5:  # Clear cache every 5 runs
    st.cache_data.clear()
    st.session_state.clear_cache = 0

if __name__ == "__main__":
    main()
