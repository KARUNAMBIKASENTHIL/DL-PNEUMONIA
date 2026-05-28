import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.model_selection import train_test_split
import cv2
import os
import torch.nn.functional as F

# ==========================================
# 1. Configuration & Setup
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define hyperparams
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

base_path = "C:/projects/DL-PNEUMONIA/Radiography"

# Dynamic dataset augmentations and preprocessing
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets and split train dynamically for stable validation metrics
try:
    full_train_dataset = datasets.ImageFolder(root=os.path.join(base_path, "train"))
    targets = full_train_dataset.targets
    
    # Stratified Split (85% train, 15% validation) to handle class imbalance stably
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=0.15,
        stratify=targets,
        random_state=42
    )
    
    train_dataset = datasets.ImageFolder(root=os.path.join(base_path, "train"), transform=transform_train)
    val_dataset = datasets.ImageFolder(root=os.path.join(base_path, "train"), transform=transform_test)
    
    # Create Subsets using the indices
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)
    
    # Standard Holdout Test Set
    test_dataset = datasets.ImageFolder(root=os.path.join(base_path, "test"), transform=transform_test)
    
    # Set up DataLoaders (num_workers=0 is safest on Windows to prevent multiprocessing issues)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Dataset successfully loaded and split!")
    print(f"Training samples: {len(train_subset)} | Validation samples: {len(val_subset)} | Test samples: {len(test_dataset)}")
    
    # Calculate exact Class Weights for Imbalance-Aware Loss
    train_targets = np.array(targets)[train_idx]
    neg_count = np.sum(train_targets == 0) # Normal
    pos_count = np.sum(train_targets == 1) # Pneumonia
    pos_weight = neg_count / pos_count
    print(f"Calculated pos_weight for loss balancing: {pos_weight:.4f}")

except Exception as e:
    print(f"Error: Dataset could not be loaded/split. Error: {e}")
    pos_weight = 1.0

# ==========================================
# 2. Custom scratch-built AttentionCNN Model
# ==========================================

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

# ==========================================
# 3. Training and Evaluation Loops
# ==========================================

def train_and_evaluate(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, pos_w=pos_weight):
    model = model.to(device)
    # Using class-imbalance weighted BCE loss with logits for numerical stability
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_prec': [], 'val_rec': [], 'val_auc': []}

    for epoch in range(epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation Phase
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds)
                all_probs.extend(probs)

        # Average losses
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate Validation Metrics
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        auc = roc_auc_score(all_labels, all_probs)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(acc)
        history['val_prec'].append(prec)
        history['val_rec'].append(rec)
        history['val_auc'].append(auc)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Metrics -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}\n")
        
        scheduler.step(avg_val_loss)

        # Save Best Model Weights
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save locally
            torch.save(model.state_dict(), 'pneumonia_model.pth')
            # Save in deployment folder as well
            deployment_dir = 'C:/projects/DL-PNEUMONIA/Deployment/model'
            os.makedirs(deployment_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(deployment_dir, 'pneumonia_model.pth'))
            print("[SUCCESS] Saved new best model checkpoint!")

    return model, history

# ==========================================
# 4. Generate Final Plots & Report Metrics
# ==========================================

def evaluate_on_test_and_plot(model, test_loader):
    print("\nEvaluating model on the Independent Test Set...")
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds).squeeze()
    all_probs = np.array(all_probs).squeeze()

    # Calculate Test Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)

    print("\n" + "="*50)
    print("FINAL TEST SET PERFORMANCE METRICS")
    print("="*50)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("="*50)

    # 1. Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.title('Confusion Matrix on Test Set')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()

    # 2. ROC Curve Plot
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300)
    plt.close()
    
    print("Saved evaluation plots: 'confusion_matrix.png', 'roc_curve.png'")

# ==========================================
# 5. Grad-CAM Interpretability
# ==========================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

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
            
        return cam

def plot_gradcam(model, image_tensor, original_image, filename='gradcam_sample.png'):
    # Hook to conv3 (deep conv features of our custom AttentionCNN)
    target_layer = model.conv3
    cam_generator = GradCAM(model, target_layer)
    
    # Generate map
    heatmap = cam_generator.generate(image_tensor.unsqueeze(0).to(device))
    
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = np.float32(heatmap_colored) / 255.0
    
    # Unnormalize original image for visualization
    original_np = original_image.permute(1, 2, 0).numpy()
    original_np = original_np * 0.5 + 0.5
    original_np = np.clip(original_np, 0, 1)
    
    # Superimpose heatmap
    cam_img = heatmap_colored + original_np
    cam_img = cam_img / np.max(cam_img)
    
    # Plot and save
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_np)
    axes[0].set_title('Original Chest X-Ray')
    axes[0].axis('off')
    
    axes[1].imshow(cam_img)
    axes[1].set_title('Grad-CAM Attention Heatmap')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved Grad-CAM visualization output to '{filename}'")

# ==========================================
# 6. Execution Block
# ==========================================

if __name__ == '__main__':
    print("Initializing Custom AttentionCNN Architecture...")
    model = AttentionCNN()
    
    print("\nStarting Imbalance-Aware Stratified Model Training...")
    trained_model, history = train_and_evaluate(model, train_loader, val_loader)
    
    print("\nLoading best model weights for final evaluation...")
    trained_model.load_state_dict(torch.load('pneumonia_model.pth', map_location=device))
    
    # Evaluate on holdout test set and save evaluation plots
    evaluate_on_test_and_plot(trained_model, test_loader)
    
    # Generate Grad-CAM Heatmap for validation sample
    print("\nGenerating Grad-CAM attention heatmap for a sample X-ray...")
    try:
        sample_batch, sample_labels = next(iter(val_loader))
        pneumonia_indices = (sample_labels == 1).nonzero(as_tuple=True)[0]
        idx = pneumonia_indices[0].item() if len(pneumonia_indices) > 0 else 0
        
        plot_gradcam(trained_model, sample_batch[idx], sample_batch[idx], filename='gradcam_sample.png')
    except Exception as e:
        print(f"Failed to generate Grad-CAM visualization. Error: {e}")
        
    print("\nAll training and evaluation tasks completed with perfection!")
