import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import numpy as np
import os
import gdown

# ------------------------
# Dropbox model download
# ------------------------
MODEL_PATH = "deeplabv3_oilspill.pth"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/yl9zmjt9eomfvefwgh559/deeplabv3_oilspill.pth?rlkey=c81oo107k1ir51zc3oj79bhrx&st=zr6j2qsx&dl=1"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Dropbox, please wait...")
    gdown.download(DROPBOX_URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully!")

# ------------------------
# Load model
# ------------------------
@st.cache_resource
def load_model():
    model = deeplabv3_resnet50(pretrained=False, aux_loss=True)
    
    # Modify main classifier for binary output
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1,1))
    
    # Modify auxiliary classifier if it exists
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256,1,kernel_size=(1,1))
    
    # Load state dict
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

model = load_model()

# ------------------------
# Preprocess image
# ------------------------
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    return preprocess(image).unsqueeze(0)

# ------------------------
# Postprocess output mask
# ------------------------
def postprocess_output(output_tensor):
    output = torch.sigmoid(output_tensor).detach().cpu().numpy()[0,0]
    mask = (output > 0.5).astype(np.uint8) * 255
    return mask

# ------------------------
# Streamlit interface
# ------------------------
st.set_page_config(page_title="Oil Spill Segmentation", layout="wide")
st.title("Oil Spill Segmentation System")
st.write("Upload a satellite image to detect oil spill regions.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if model is not None:
        with st.spinner("Detecting oil spill..."):
            input_tensor = preprocess_image(image)
            with torch.no_grad():
                output = model(input_tensor)['out']
            mask = postprocess_output(output)

        st.image(mask, caption="Detected Oil Spill", use_column_width=True)
        st.success("Segmentation completed!")

        if st.checkbox("Save the result image"):
            result_image = Image.fromarray(mask)
            result_image.save("result.png")
            st.write("Result image saved as result.png")
    else:
        st.error("Model not loaded. Check your model file.")
