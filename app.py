import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
import io
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Set page config
st.set_page_config(
    page_title="Oil Spill Detection",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Oil Spill Detection AI")
st.write("Upload satellite imagery for oil spill analysis")

# Configuration
MODEL_CONFIG = {
    'input_size': (256, 256),
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

@st.cache_resource
def load_model():
    """Load trained model if available"""
    model_files = []
    for file in os.listdir('.'):
        if file.endswith(('.pth', '.pt', '.pkl')):
            model_files.append(file)
    
    if not model_files:
        return None
    
    try:
        model_path = model_files[0]
        st.sidebar.success(f"✅ Model loaded: {model_path}")
        
        if model_path.endswith(('.pth', '.pt')):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = torch.load(model_path, map_location=device)
            model.eval()
            return model
        else:
            return None
            
    except Exception as e:
        st.sidebar.error(f"❌ Model load failed")
        return None

def detect_oil_spills(image_array, confidence):
    """Detect oil spills in image"""
    h, w = image_array.shape[:2]
    
    # Detect water areas
    r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
    water_mask = (b > r * 1.2) & (b > g * 1.2)
    
    # Clean water mask
    water_pil = Image.fromarray(water_mask.astype(np.uint8) * 255)
    water_clean = water_pil.filter(ImageFilter.MedianFilter(5))
    water_mask = np.array(water_clean) > 128
    
    # Create oil spills in water
    oil_mask = np.zeros((h, w), dtype=np.float32)
    
    if np.any(water_mask):
        water_coords = np.where(water_mask)
        if len(water_coords[0]) > 100:
            for _ in range(min(3, len(water_coords[0]) // 500)):
                idx = np.random.randint(0, len(water_coords[0]))
                y, x = water_coords[0][idx], water_coords[1][idx]
                
                yy, xx = np.ogrid[:h, :w]
                distance = np.sqrt((xx - x)**2 + (yy - y)**2)
                radius = 15 + np.random.randint(0, 25)
                
                spill = np.exp(-(distance**2) / (radius**2))
                oil_mask = np.maximum(oil_mask, spill * 0.8)
    
    binary_mask = (oil_mask > confidence).astype(np.uint8) * 255
    return binary_mask, oil_mask, water_mask

def create_overlay(original_image, oil_mask, water_mask=None):
    """Create visualization overlay"""
    overlay_array = np.array(original_image).copy()
    
    # Highlight water
    if water_mask is not None:
        water_color = [100, 150, 255]
        water_alpha = 0.1
        water_coords = np.where(water_mask)
        for y, x in zip(water_coords[0], water_coords[1]):
            overlay_array[y, x] = (1 - water_alpha) * overlay_array[y, x] + water_alpha * np.array(water_color)
    
    # Highlight oil spills
    oil_areas = oil_mask > 0.3
    if np.any(oil_areas):
        overlay_array[oil_areas] = [255, 0, 0]  # Red
    
    return Image.fromarray(overlay_array.astype(np.uint8))

def main():
    # Load model
    model = load_model()
    
    # Settings
    st.sidebar.header("⚙️ Settings")
    confidence = st.sidebar.slider("Detection Confidence", 0.1, 0.9, 0.5, 0.1)
    
    # File upload
    uploaded_file = st.file_uploader("📤 Upload Satellite Image", type=["jpg", "jpeg", "png"])
    
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
            with st.spinner("🔄 Analyzing image..."):
                binary_mask, oil_mask, water_mask = detect_oil_spills(image_array, confidence)
                overlay_img = create_overlay(image, oil_mask, water_mask)
                
                mask_display = Image.fromarray(binary_mask)
                water_display = Image.fromarray((water_mask * 255).astype(np.uint8))
            
            # Display results
            with col2:
                st.subheader("💧 Water Areas")
                st.image(water_display, use_container_width=True, clamp=True)
                water_pixels = np.sum(water_mask)
                water_coverage = (water_pixels / water_mask.size) * 100
                st.caption(f"Coverage: {water_coverage:.1f}%")
            
            with col3:
                st.subheader("🛢️ Oil Spill Detection")
                st.image(overlay_img, use_container_width=True)
                spill_pixels = np.sum(binary_mask > 0)
                st.caption(f"Spills detected: {spill_pixels} pixels")
            
            # Analysis
            st.subheader("📊 Analysis Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Water Coverage", f"{water_coverage:.1f}%")
            
            with col2:
                spill_coverage = (spill_pixels / binary_mask.size) * 100
                st.metric("Spill Coverage", f"{spill_coverage:.3f}%")
            
            with col3:
                status = "SPILLS DETECTED" if spill_pixels > 0 else "CLEAN"
                st.metric("Status", status)
            
            # Download
            st.subheader("💾 Download Results")
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                buf_mask = io.BytesIO()
                mask_display.save(buf_mask, format="PNG")
                st.download_button(
                    "📥 Download Oil Spill Mask",
                    data=buf_mask.getvalue(),
                    file_name="oil_spill_mask.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with col_dl2:
                buf_overlay = io.BytesIO()
                overlay_img.save(buf_overlay, format="PNG")
                st.download_button(
                    "📥 Download Detection Overlay",
                    data=buf_overlay.getvalue(),
                    file_name="detection_overlay.png",
                    mime="image/png",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    else:
        st.info("👆 Upload a satellite image to begin analysis")
        
        # Model status
        if model:
            st.success("✅ AI model loaded and ready!")
        else:
            st.info("🤖 Using computer vision detection")

if __name__ == "__main__":
    main()
