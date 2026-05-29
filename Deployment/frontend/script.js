// script.js
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
        const response = await fetch('https://pneumonia-prediction-chest-xray.onrender.com/predict', {
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
