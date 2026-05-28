import os

frontend_dir = "C:/projects/DL-PNEUMONIA/frontend"
os.makedirs(frontend_dir, exist_ok=True)

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pneumonia Detection AI</title>
    <link rel="stylesheet" href="style.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
</head>
<body>
    <div class="container">
        <header>
            <h1><span class="icon">🫁</span> Pneumonia Detection AI</h1>
            <p>Upload a Chest X-Ray to check for Pneumonia</p>
        </header>

        <main>
            <div class="upload-section">
                <input type="file" id="imageInput" accept=".jpg, .jpeg, .png" hidden>
                <label for="imageInput" class="upload-box" id="uploadBox">
                    <div class="upload-content">
                        <span class="upload-icon">📤</span>
                        <p>Click or Drag X-Ray Image Here</p>
                        <small>Supports JPG, PNG</small>
                    </div>
                    <img id="imagePreview" class="hidden" alt="Preview">
                </label>
            </div>

            <div class="action-section">
                <button id="predictBtn" disabled>Run Analysis</button>
                <div id="loading" class="hidden">
                    <div class="spinner"></div>
                    <span>Analyzing Image...</span>
                </div>
            </div>

            <div id="resultBox" class="result-box hidden">
                <h2 id="predictionResult"></h2>
                <p id="confidenceScore"></p>
                <p id="recommendation"></p>
            </div>
        </main>
    </div>

    <script src="script.js"></script>
</body>
</html>
"""

css_content = """/* style.css */
:root {
    --primary-blue: #0056b3;
    --light-blue: #e6f2ff;
    --white: #ffffff;
    --text-dark: #333333;
    --text-gray: #666666;
    --danger-red: #dc3545;
    --success-green: #28a745;
    --bg-color: #f4f7f6;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family: 'Inter', sans-serif;
}

body {
    background-color: var(--bg-color);
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
}

.container {
    background: var(--white);
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
    width: 100%;
    max-width: 500px;
    text-align: center;
}

header {
    margin-bottom: 2rem;
}

h1 {
    color: var(--primary-blue);
    font-size: 1.8rem;
    margin-bottom: 0.5rem;
}

header p {
    color: var(--text-gray);
    font-size: 0.9rem;
}

.upload-section {
    margin-bottom: 1.5rem;
}

.upload-box {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 250px;
    border: 2px dashed var(--primary-blue);
    border-radius: 10px;
    background-color: var(--light-blue);
    cursor: pointer;
    transition: all 0.3s ease;
    overflow: hidden;
    position: relative;
}

.upload-box:hover {
    background-color: #d1e7fd;
}

.upload-icon {
    font-size: 3rem;
    margin-bottom: 10px;
}

.upload-content p {
    color: var(--primary-blue);
    font-weight: 600;
}

.upload-content small {
    color: var(--text-gray);
}

#imagePreview {
    width: 100%;
    height: 100%;
    object-fit: cover;
    position: absolute;
    top: 0;
    left: 0;
    background: #000;
}

.hidden {
    display: none !important;
}

button {
    background-color: var(--primary-blue);
    color: white;
    border: none;
    padding: 12px 24px;
    font-size: 1rem;
    border-radius: 8px;
    cursor: pointer;
    width: 100%;
    font-weight: 600;
    transition: background 0.3s ease;
}

button:hover:not(:disabled) {
    background-color: #004494;
}

button:disabled {
    background-color: #a0c4e8;
    cursor: not-allowed;
}

#loading {
    margin-top: 1rem;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 10px;
    color: var(--primary-blue);
    font-weight: 600;
}

.spinner {
    width: 20px;
    height: 20px;
    border: 3px solid rgba(0, 86, 179, 0.3);
    border-top: 3px solid var(--primary-blue);
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.result-box {
    margin-top: 1.5rem;
    padding: 1.5rem;
    border-radius: 8px;
    animation: fadeIn 0.5s ease-in-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}

.result-pneumonia {
    background-color: #ffe6e6;
    border: 2px solid var(--danger-red);
}

.result-pneumonia h2 { color: var(--danger-red); }

.result-normal {
    background-color: #e6ffe6;
    border: 2px solid var(--success-green);
}

.result-normal h2 { color: var(--success-green); }

.result-box h2 {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
}

.result-box p {
    color: var(--text-dark);
    font-weight: 600;
}
"""

js_content = """// script.js
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const uploadContent = document.querySelector('.upload-content');
const predictBtn = document.getElementById('predictBtn');
const loading = document.getElementById('loading');
const resultBox = document.getElementById('resultBox');
const predictionResult = document.getElementById('predictionResult');
const confidenceScore = document.getElementById('confidenceScore');
const recommendation = document.getElementById('recommendation');

let selectedFile = null;

// Handle File Selection
imageInput.addEventListener('change', function(event) {
    const file = event.target.files[0];
    
    if (file) {
        if (!file.type.match('image/jpeg') && !file.type.match('image/png')) {
            alert('Please select a valid JPG or PNG image.');
            return;
        }

        selectedFile = file;
        const reader = new FileReader();

        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove('hidden');
            uploadContent.classList.add('hidden');
            predictBtn.disabled = false;
        }

        reader.readAsDataURL(file);
        
        // Reset previous results
        resultBox.classList.add('hidden');
    }
});

// Handle Predict Button
predictBtn.addEventListener('click', async function() {
    if (!selectedFile) {
        alert("Please upload an image first.");
        return;
    }

    // UI State: Loading
    predictBtn.disabled = true;
    loading.classList.remove('hidden');
    resultBox.classList.add('hidden');

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        
        // UI State: Done Loading
        loading.classList.add('hidden');
        predictBtn.disabled = false;
        
        // Display Result
        showResult(data.prediction, data.confidence);

    } catch (error) {
        console.error("Error during prediction:", error);
        alert("Failed to connect to the server. Please ensure the backend API is running.");
        loading.classList.add('hidden');
        predictBtn.disabled = false;
    }
});

function showResult(prediction, confidence) {
    resultBox.classList.remove('hidden', 'result-pneumonia', 'result-normal');
    
    predictionResult.innerText = prediction.toUpperCase();
    confidenceScore.innerText = `Confidence: ${(confidence * 100).toFixed(2)}%`;

    if (prediction.toLowerCase() === 'pneumonia') {
        resultBox.classList.add('result-pneumonia');
        recommendation.innerText = "Consult a doctor immediately.";
    } else {
        resultBox.classList.add('result-normal');
        recommendation.innerText = "No signs of pneumonia detected.";
    }
}
"""

flask_content = """from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import torch.nn as nn
import io
import os

app = Flask(__name__)
CORS(app) # Required to allow frontend to access the API

class PneumoniaResNet(nn.Module):
    def __init__(self):
        super(PneumoniaResNet, self).__init__()
        self.resnet = models.resnet18(weights=None)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 1)
        )

    def forward(self, x):
        return self.resnet(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PneumoniaResNet()

model_path = 'pneumonia_model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully!")
else:
    print("WARNING: Model file not found. Ensure 'pneumonia_model.pth' exists.")

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

@app.route('/predict', methods=['POST'])
def predict():
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

if __name__ == '__main__':
    app.run(port=5000, debug=True)
"""

with open(f"{frontend_dir}/index.html", "w", encoding='utf-8') as f:
    f.write(html_content)
with open(f"{frontend_dir}/style.css", "w", encoding='utf-8') as f:
    f.write(css_content)
with open(f"{frontend_dir}/script.js", "w", encoding='utf-8') as f:
    f.write(js_content)
with open("C:/projects/DL-PNEUMONIA/DL-PNEUMONIa/app_flask.py", "w", encoding='utf-8') as f:
    f.write(flask_content)
