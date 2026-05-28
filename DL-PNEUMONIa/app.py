import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import os
import torch.nn.functional as F

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load Model ---
@st.cache_resource
def load_model():
    model = AttentionCNN()
    model_path = 'pneumonia_model.pth'
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    else:
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
st.markdown("### Upload a Chest X-Ray image to instantly detect Pneumonia using Deep Learning.")

model = load_model()

if model is None:
    st.error("⚠️ Model file `pneumonia_model.pth` not found! Please finish training your model in the notebook and save it first.")
else:
    uploaded_file = st.file_uploader("Choose a Chest X-Ray image (JPG/PNG)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display Image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Chest X-Ray', use_column_width=True)
        
        st.markdown("---")
        with st.spinner("Analyzing Image..."):
            # Preprocess Image
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.sigmoid(output).item()
                
            # Display Results
            if prob > 0.5:
                st.error(f"🚨 **Prediction: PNEUMONIA DETECTED**")
                st.markdown(f"**Confidence:** `{prob*100:.2f}%`")
                st.info("💡 *Recommendation: Please consult a Pulmonologist or Doctor immediately.*")
            else:
                st.success(f"✅ **Prediction: NORMAL (Clear Lungs)**")
                st.markdown(f"**Confidence:** `{(1-prob)*100:.2f}%`")
                st.info("💡 *No signs of pneumonia detected. Keep up the good health!*")
