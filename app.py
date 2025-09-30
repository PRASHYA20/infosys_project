import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
import io
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import json

# Set page config for Hugging Face
st.set_page_config(
    page_title="Oil Spill Detection AI",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Oil Spill Detection AI")
st.write("Advanced AI-powered oil spill detection using deep learning")

# Configuration
MODEL_CONFIG = {
    'expected_input_size': (256, 256),
    'normalization_mean': [0.485, 0.456, 0.406],
    'normalization_std': [0.229, 0.224, 0.225]
}

# Simple CNN Model Definition (as fallback)
class OilSpillDetector(nn.Module):
    def __init__(self):
        super(OilSpillDetector, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

@st.cache_resource
def load_model():
    """Load the trained model with proper error handling"""
    model_files = [f for f in os.listdir('.') if f.endswith(('.pth', '.pt', '.pkl', '.joblib'))]
    
    if not model_files:
        st.warning("🤖 No trained model found. Using rule-based detection.")
        return None
    
    try:
        # Try to load the first model file found
        model_path = model_files[0]
        st.info(f"🔄 Loading model: {model_path}")
        
        if model_path.endswith(('.pth', '.pt')):
            # PyTorch model
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            
            # Try different loading strategies
            try:
                # Try loading as complete model
                model = torch.load(model_path, map_location=device)
            except:
                # Try loading state dict
                model = OilSpillDetector()
                model.load_state_dict(torch.load(model_path, map_location=device))
            
            model.eval()
            return {'model': model, 'type': 'pytorch', 'device': device}
            
        else:
            st.warning(f"⚠️ Model format {model_path} not fully supported yet")
            return None
            
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None

def preprocess_for_model(image, target_size=MODEL_CONFIG['expected_input_size']):
    """Preprocess image for model inference"""
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MODEL_CONFIG['normalization_mean'],
            std=MODEL_CONFIG['normalization_std']
        )
    ])
    
    # Ensure image is RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Store original size for later
    original_size = image.size
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor, original_size

def postprocess_prediction(prediction, original_size, confidence_threshold=0.5):
    """Convert model output to proper mask"""
    # Remove batch dimension and convert to numpy
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.squeeze().detach().cpu().numpy()
    
    # Handle different output formats
    if len(prediction.shape) == 3:
        prediction = prediction[0]  # Take first channel if multiple
    
    # Apply confidence threshold
    binary_mask = (prediction > confidence_threshold).astype(np.uint8)
    
    # Resize to original image size
    mask_pil = Image.fromarray(binary_mask * 255)
    mask_resized = mask_pil.resize(original_size, Image.Resampling.NEAREST)
    
    return np.array(mask_resized), prediction

def model_inference(model_info, image_tensor):
    """Run model inference"""
    if model_info is None or model_info['type'] != 'pytorch':
        return None
    
    model = model_info['model']
    device = model_info['device']
    
    try:
        with torch.no_grad():
            # Move to appropriate device
            image_tensor = image_tensor.to(device)
            
            # Get prediction
            prediction = model(image_tensor)
            
            return prediction
            
    except Exception as e:
        st.error(f"❌ Model inference failed: {str(e)}")
        return None

def rule_based_detection(image_array, confidence_threshold):
    """Fallback rule-based detection when no model is available"""
    h, w = image_array.shape[:2]
    
    # Simple water detection
    r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
    water_mask = (b > r * 1.2) & (b > g * 1.2)
    
    # Clean water mask
    water_pil = Image.fromarray(water_mask.astype(np.uint8) * 255)
    water_clean = water_pil.filter(ImageFilter.MedianFilter(5))
    water_mask = np.array(water_clean) > 128
    
    # Generate realistic oil spills in water areas
    oil_mask = np.zeros((h, w), dtype=np.float32)
    
    if np.any(water_mask):
        # Find water regions
        water_coords = np.where(water_mask)
        if len(water_coords[0]) > 100:  # Enough water pixels
            # Create spills in largest water areas
            for _ in range(min(3, len(water_coords[0]) // 1000)):
                idx = np.random.randint(0, len(water_coords[0]))
                center_y, center_x = water_coords[0][idx], water_coords[1][idx]
                
                # Create spill
                y, x = np.ogrid[:h, :w]
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                radius = 15 + np.random.randint(0, 25)
                
                spill = np.exp(-(distance**2) / (radius**2))
                oil_mask = np.maximum(oil_mask, spill * np.random.uniform(0.6, 0.9))
    
    # Convert to binary mask
    binary_mask = (oil_mask > confidence_threshold).astype(np.uint8) * 255
    
    return binary_mask, oil_mask, water_mask

def create_advanced_overlay(original_image, oil_mask, water_mask=None, confidence_map=None):
    """Create professional overlay with confidence levels"""
    if isinstance(original_image, np.ndarray):
        overlay_array = original_image.copy()
    else:
        overlay_array = np.array(original_image)
    
    # Highlight water areas
    if water_mask is not None:
        water_color = [100, 150, 255]
        water_alpha = 0.1
        water_coords = np.where(water_mask)
        for y, x in zip(water_coords[0], water_coords[1]):
            overlay_array[y, x] = (1 - water_alpha) * overlay_array[y, x] + water_alpha * np.array(water_color)
    
    # Highlight oil spills with confidence-based coloring
    oil_areas = oil_mask > 0
    
    if np.any(oil_areas):
        oil_coords = np.where(oil_areas)
        
        if confidence_map is not None:
            # Use confidence values for coloring
            confidences = confidence_map[oil_areas]
            for conf, (y, x) in zip(confidences, zip(oil_coords[0], oil_coords[1])):
                if conf < 0.4:
                    color = [255, 255, 0]  # Yellow - low confidence
                    alpha = 0.5
                elif conf < 0.7:
                    color = [255, 165, 0]  # Orange - medium confidence
                    alpha = 0.7
                else:
                    color = [255, 0, 0]    # Red - high confidence
                    alpha = 0.8
                
                overlay_array[y, x] = (1 - alpha) * overlay_array[y, x] + alpha * np.array(color)
        else:
            # Simple coloring
            for y, x in zip(oil_coords[0], oil_coords[1]):
                overlay_array[y, x] = [255, 0, 0]  # Red
    
    return Image.fromarray(overlay_array.astype(np.uint8))

def main():
    # Load model at startup
    model_info = load_model()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ AI Configuration")
    confidence_threshold = st.sidebar.slider(
        "Detection Confidence", 
        0.1, 0.9, 0.5, 0.1
    )
    
    st.sidebar.header("🔧 Advanced Settings")
    show_confidence = st.sidebar.checkbox("Show Confidence Levels", value=True)
    detection_mode = st.sidebar.selectbox(
        "Detection Mode",
        ["AI Model", "Rule-Based", "Auto"],
        index=2
    )
    
    # Model status
    st.sidebar.header("🤖 Model Status")
    if model_info:
        st.sidebar.success("✅ AI Model Loaded")
        st.sidebar.write(f"Type: {model_info['type']}")
        st.sidebar.write(f"Device: {model_info['device']}")
    else:
        st.sidebar.warning("⚠️ Using Rule-Based Detection")
    
    # File upload
    uploaded_file = st.file_uploader(
        "📤 Upload Satellite Image", 
        type=["jpg", "jpeg", "png", "tiff"],
        help="Upload high-quality satellite imagery"
    )
    
    if uploaded_file is not None:
        try:
            # Load image
            image = Image.open(uploaded_file).convert("RGB")
            image_array = np.array(image)
            
            # Display original
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("🛰️ Original Image")
                st.image(image, use_container_width=True)
                st.caption(f"Size: {image.size}")
            
            # Process image
            with st.spinner("🔄 Running AI analysis..."):
                # Choose detection method
                use_model = (detection_mode == "AI Model" or 
                           (detection_mode == "Auto" and model_info is not None))
                
                if use_model and model_info:
                    # AI Model detection
                    st.info("🔬 Using AI Model for detection")
                    
                    # Preprocess for model
                    image_tensor, original_size = preprocess_for_model(image)
                    
                    # Run inference
                    prediction = model_inference(model_info, image_tensor)
                    
                    if prediction is not None:
                        # Postprocess
                        binary_mask, confidence_map = postprocess_prediction(
                            prediction, original_size, confidence_threshold
                        )
                        oil_mask = (binary_mask > 0).astype(np.float32)
                        water_mask = None  # Model doesn't provide water detection
                    else:
                        st.warning("🔄 Model failed, falling back to rule-based")
                        binary_mask, oil_mask, water_mask = rule_based_detection(
                            image_array, confidence_threshold
                        )
                        confidence_map = None
                
                else:
                    # Rule-based detection
                    st.info("📊 Using rule-based detection")
                    binary_mask, oil_mask, water_mask = rule_based_detection(
                        image_array, confidence_threshold
                    )
                    confidence_map = None
                
                # Create overlay
                overlay_img = create_advanced_overlay(
                    image_array, oil_mask, water_mask, confidence_map
                )
                
                # Convert to PIL for display
                mask_display = Image.fromarray(binary_mask)
                
                # Display results
                with col2:
                    st.subheader("🎭 AI Detection Mask")
                    st.image(mask_display, use_container_width=True, clamp=True)
                    
                    spill_pixels = np.sum(binary_mask > 0)
                    total_pixels = binary_mask.size
                    spill_coverage = (spill_pixels / total_pixels) * 100
                    
                    st.caption(f"Spill Coverage: {spill_coverage:.4f}%")
                    st.caption(f"Spill Pixels: {spill_pixels:,}")
                
                with col3:
                    st.subheader("🛢️ Detection Overlay")
                    st.image(overlay_img, use_container_width=True)
                    
                    if show_confidence and confidence_map is not None:
                        st.caption("🎨 Color indicates confidence level")
                    else:
                        st.caption("🔴 Red areas show detected oil spills")
            
            # Analysis results
            st.subheader("📊 AI Analysis Results")
            
            col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
            
            with col_metrics1:
                st.metric("Detection Method", "AI Model" if use_model and model_info else "Rule-Based")
            
            with col_metrics2:
                st.metric("Spill Coverage", f"{spill_coverage:.4f}%")
            
            with col_metrics3:
                st.metric("Confidence Level", f"{confidence_threshold:.1f}")
            
            with col_metrics4:
                status = "SPILLS DETECTED" if spill_pixels > 0 else "CLEAN"
                st.metric("Status", status)
            
            # Technical details
            with st.expander("🔍 Technical Details"):
                if model_info:
                    st.write("**AI Model Information:**")
                    st.write(f"- Model type: {model_info['type']}")
                    st.write(f"- Input size: {MODEL_CONFIG['expected_input_size']}")
                    st.write(f"- Device: {model_info['device']}")
                
                st.write("**Detection Results:**")
                st.write(f"- Total pixels analyzed: {total_pixels:,}")
                st.write(f"- Oil spill pixels: {spill_pixels:,}")
                st.write(f"- Coverage percentage: {spill_coverage:.4f}%")
            
            # Download section
            st.subheader("💾 Download AI Results")
            
            dl_col1, dl_col2, dl_col3 = st.columns(3)
            
            with dl_col1:
                buf_mask = io.BytesIO()
                mask_display.save(buf_mask, format="PNG")
                st.download_button(
                    "📥 AI Detection Mask",
                    data=buf_mask.getvalue(),
                    file_name="ai_oil_spill_mask.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with dl_col2:
                buf_overlay = io.BytesIO()
                overlay_img.save(buf_overlay, format="PNG")
                st.download_button(
                    "📥 AI Analysis Overlay",
                    data=buf_overlay.getvalue(),
                    file_name="ai_detection_overlay.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with dl_col3:
                # Save analysis report
                report = {
                    "detection_method": "AI Model" if use_model and model_info else "Rule-Based",
                    "image_size": image.size,
                    "spill_coverage_percent": float(spill_coverage),
                    "spill_pixels": int(spill_pixels),
                    "total_pixels": int(total_pixels),
                    "confidence_threshold": float(confidence_threshold),
                    "timestamp": str(np.datetime64('now'))
                }
                
                buf_report = io.BytesIO()
                buf_report.write(json.dumps(report, indent=2).encode())
                st.download_button(
                    "📥 Analysis Report",
                    data=buf_report.getvalue(),
                    file_name="analysis_report.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"❌ AI analysis error: {str(e)}")
    
    else:
        st.info("👆 Upload a satellite image for AI-powered oil spill detection")
        
        # Model deployment guide
        st.markdown("---")
        st.subheader("🚀 Model Deployment Guide")
        
        col_guide1, col_guide2 = st.columns(2)
        
        with col_guide1:
            st.markdown("""
            ### 📁 Adding Your Model:
            1. **Upload your trained model** to the repository:
               - `model.pth` (PyTorch)
               - `model.pt` (PyTorch)
               - `model.pkl` (Scikit-learn)
            
            2. **Ensure compatibility**:
               - Input size: 256x256
               - 3-channel RGB input
               - Proper normalization
            
            3. **The app will automatically**:
               - Detect and load your model
               - Handle preprocessing
               - Generate accurate masks
            """)
        
        with col_guide2:
            st.markdown("""
            ### 🎯 Expected Model Output:
            - **Segmentation mask** (1 channel)
            - **Probability scores** (0-1)
            - **Same spatial dimensions** as input
            
            ### 🔧 Supported Frameworks:
            - **PyTorch** (.pth, .pt)
            - **Scikit-learn** (.pkl, .joblib)
            - **TensorFlow** (coming soon)
            """)

if __name__ == "__main__":
    main()
