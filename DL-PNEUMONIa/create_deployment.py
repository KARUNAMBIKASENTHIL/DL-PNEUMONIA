import os
import shutil

deploy_dir = "C:/projects/DL-PNEUMONIA/Deployment"
os.makedirs(deploy_dir, exist_ok=True)
os.makedirs(f"{deploy_dir}/frontend", exist_ok=True)
os.makedirs(f"{deploy_dir}/model", exist_ok=True)

# 1. Copy Frontend Files
src_frontend = "C:/projects/DL-PNEUMONIA/frontend"
for file in ["index.html", "style.css", "script.js"]:
    if os.path.exists(f"{src_frontend}/{file}"):
        shutil.copy(f"{src_frontend}/{file}", f"{deploy_dir}/frontend/{file}")

# 2. Copy Model File
model_src = "C:/projects/DL-PNEUMONIA/DL-PNEUMONIa/pneumonia_model.pth"
if os.path.exists(model_src):
    shutil.copy(model_src, f"{deploy_dir}/model/pneumonia_model.pth")

# 3. Create requirements.txt
requirements = """Flask==3.0.0
flask-cors==4.0.0
torch==2.2.0
torchvision==0.17.0
Pillow==10.2.0
gunicorn==21.2.0
"""
with open(f"{deploy_dir}/requirements.txt", "w") as f:
    f.write(requirements)

# 4. Create Procfile
procfile = "web: gunicorn app:app\n"
with open(f"{deploy_dir}/Procfile", "w") as f:
    f.write(procfile)

# 5. Create app.py (Deployment Ready Backend)
flask_content = """from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import torch.nn as nn
import io
import os

app = Flask(__name__, static_folder='frontend')
CORS(app) # Allow frontend to communicate

# Define Model Architecture
class PneumoniaResNet(nn.Module):
    def __init__(self):
        super(PneumoniaResNet, self).__init__()
        self.resnet = models.resnet50(weights=None)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 1)
        )
    def forward(self, x):
        return self.resnet(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PneumoniaResNet()

# Load Model from the 'model' folder
model_path = os.path.join(os.path.dirname(__file__), 'model', 'pneumonia_model.pth')
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully!")
else:
    print(f"WARNING: Model file not found at {model_path}")

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Serve Frontend HTML
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

# Serve other static files (CSS, JS)
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
            
        if prob > 0.5:
            return jsonify({'prediction': 'Pneumonia', 'confidence': prob})
        else:
            return jsonify({'prediction': 'Normal', 'confidence': 1 - prob})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use PORT environment variable if available (required for Render/Heroku)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
"""
with open(f"{deploy_dir}/app.py", "w") as f:
    f.write(flask_content)

print(f"Deployment folder structure created at: {deploy_dir}")
