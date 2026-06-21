import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import os
import torch.nn.functional as F
import cv2
import numpy as np

# --- Model Architecture ---
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

class AttentionCNN(nn.Module):
    def __init__(self):
        super(AttentionCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.sa = SpatialAttention(kernel_size=7)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.sa(x)
        x = self.classifier(x)
        return x

# --- Grad-CAM Class ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks and save handles to unregister cleanly afterwards
        self.h1 = self.target_layer.register_forward_hook(self.save_activation)
        self.h2 = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x):
        self.model.eval()
        output = self.model(x)
        self.model.zero_grad()
        output.backward()
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.shape[-1], x.shape[-2]))
        
        # Normalize
        cam_min = np.min(cam)
        cam_max = np.max(cam)
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        # Clean up hooks to prevent memory leaks or multiple callbacks!
        self.h1.remove()
        self.h2.remove()
            
        return cam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load Model ---
@st.cache_resource
def load_model():
    model = AttentionCNN()
    # Resolve the path relative to this script's directory
    model_path = os.path.join(os.path.dirname(__file__), 'pneumonia_model.pth')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    else:
        # Fallback to root directory check if not found in the script folder
        fallback_path = 'pneumonia_model.pth'
        if os.path.exists(fallback_path):
            model.load_state_dict(torch.load(fallback_path, map_location=device))
            model.to(device)
            model.eval()
            return model
        return None

# --- Web App UI ---
st.set_page_config(page_title="Pneumonia Detection AI", page_icon="🫁", layout="centered")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #0066cc; color: white; border-radius: 8px;}
    .stAlert {border-radius: 10px;}
    </style>
    """, unsafe_allow_html=True)

st.title("🫁 Pneumonia Detection from Chest X-Ray")
st.markdown("### Upload a Chest X-Ray image to instantly detect Pneumonia and see visual lung attention maps.")

model = load_model()

if model is None:
    st.error("⚠️ Model file `pneumonia_model.pth` not found! Please finish training your model in the notebook and save it first.")
else:
    uploaded_file = st.file_uploader("Choose a Chest X-Ray image (JPG/PNG)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        st.markdown("---")
        with st.spinner("Analyzing Image & Generating Grad-CAM Heatmap..."):
            # Preprocess Image
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            # Enable gradient calculation for Grad-CAM
            input_tensor = transform(image).unsqueeze(0).to(device)
            input_tensor.requires_grad = True
            
            # Predict
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
            
            # Generate Grad-CAM Heatmap
            try:
                cam_generator = GradCAM(model, model.conv3)
                heatmap = cam_generator.generate(input_tensor)
                
                # Superimpose heatmap
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                heatmap_colored = np.float32(heatmap_colored) / 255.0
                
                orig_resized = image.resize((224, 224))
                orig_np = np.float32(orig_resized) / 255.0
                
                cam_img = heatmap_colored + orig_np
                cam_img = cam_img / np.max(cam_img)
                cam_img = np.uint8(255 * cam_img)
                
                heatmap_image = Image.fromarray(cam_img)
                heatmap_available = True
            except Exception as e:
                st.warning(f"Grad-CAM generation failed: {e}")
                heatmap_available = False
                
        # Display side-by-side images
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Uploaded Chest X-Ray', use_container_width=True)
        with col2:
            if heatmap_available:
                st.image(heatmap_image, caption='Grad-CAM Attention Heatmap', use_container_width=True)
            else:
                st.info("Visual explainability heatmap is unavailable for this model checkpoint.")
        
        st.markdown("---")
        # Display Results
        if prob > 0.5:
            st.error(f"🚨 **Prediction: PNEUMONIA DETECTED**")
            st.markdown(f"**Confidence:** `{prob*100:.2f}%`")
            st.info("💡 *Recommendation: Please consult a Pulmonologist or Doctor immediately.*")
        else:
            st.success(f"✅ **Prediction: NORMAL (Clear Lungs)**")
            st.markdown(f"**Confidence:** `{(1-prob)*100:.2f}%`")
            st.info("💡 *No signs of pneumonia detected. Keep up the good health!*")
