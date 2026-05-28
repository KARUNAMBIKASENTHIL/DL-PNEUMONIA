import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Pneumonia Detection - High Accuracy Model (Fast Training)\n",
    "Intha notebook la 'ResNet18' pre-trained model use panrom. Ithu CPU la ResNet50 vida romba fast ah train aagum!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "import torchvision\n",
    "import torchvision.transforms as transforms\n",
    "import torchvision.models as models\n",
    "from torch.utils.data import DataLoader\n",
    "from torchvision import datasets\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix\n",
    "import cv2\n",
    "import os\n",
    "import torch.nn.functional as F\n",
    "\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "print(f\"Using device: {device}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Hyperparameter Tuning\n",
    "BATCH_SIZE = 32\n",
    "EPOCHS = 3 # Reduced to 3 epochs for extremely fast results\n",
    "LEARNING_RATE = 0.001\n",
    "WEIGHT_DECAY = 1e-4\n",
    "\n",
    "base_path = \"C:/projects/DL-PNEUMONIA/Radiography\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "transform_train = transforms.Compose([\n",
    "    transforms.Resize((224, 224)),\n",
    "    transforms.RandomHorizontalFlip(),\n",
    "    transforms.RandomRotation(10),\n",
    "    transforms.ToTensor(),\n",
    "    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))\n",
    "])\n",
    "\n",
    "transform_test = transforms.Compose([\n",
    "    transforms.Resize((224, 224)),\n",
    "    transforms.ToTensor(),\n",
    "    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))\n",
    "])\n",
    "\n",
    "train_dataset = datasets.ImageFolder(root=os.path.join(base_path, \"train\"), transform=transform_train)\n",
    "val_dataset = datasets.ImageFolder(root=os.path.join(base_path, \"val\"), transform=transform_test)\n",
    "\n",
    "train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)\n",
    "val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)\n",
    "\n",
    "print(f\"Classes: {train_dataset.classes}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "class PneumoniaResNet(nn.Module):\n",
    "    def __init__(self):\n",
    "        super(PneumoniaResNet, self).__init__()\n",
    "        # Use ResNet18 (Much faster on CPU than ResNet50)\n",
    "        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)\n",
    "        \n",
    "        # Freeze early layers to speed up training massively\n",
    "        for param in list(self.resnet.parameters())[:-10]:\n",
    "            param.requires_grad = False\n",
    "            \n",
    "        num_ftrs = self.resnet.fc.in_features\n",
    "        self.resnet.fc = nn.Sequential(\n",
    "            nn.Dropout(0.5),\n",
    "            nn.Linear(num_ftrs, 1)\n",
    "        )\n",
    "\n",
    "    def forward(self, x):\n",
    "        return self.resnet(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def train_and_evaluate(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE):\n",
    "    model = model.to(device)\n",
    "    criterion = nn.BCEWithLogitsLoss() \n",
    "    \n",
    "    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=WEIGHT_DECAY)\n",
    "    \n",
    "    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=1)\n",
    "\n",
    "    for epoch in range(epochs):\n",
    "        model.train()\n",
    "        running_loss = 0.0\n",
    "        for inputs, labels in train_loader:\n",
    "            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)\n",
    "            optimizer.zero_grad()\n",
    "            outputs = model(inputs)\n",
    "            loss = criterion(outputs, labels)\n",
    "            loss.backward()\n",
    "            optimizer.step()\n",
    "            running_loss += loss.item()\n",
    "\n",
    "        model.eval()\n",
    "        val_loss = 0.0\n",
    "        all_labels, all_preds, all_probs = [], [], []\n",
    "\n",
    "        with torch.no_grad():\n",
    "            for inputs, labels in val_loader:\n",
    "                inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)\n",
    "                outputs = model(inputs)\n",
    "                loss = criterion(outputs, labels)\n",
    "                val_loss += loss.item()\n",
    "\n",
    "                probs = torch.sigmoid(outputs).cpu().numpy()\n",
    "                preds = (probs > 0.5).astype(int)\n",
    "                \n",
    "                all_labels.extend(labels.cpu().numpy())\n",
    "                all_preds.extend(preds)\n",
    "                all_probs.extend(probs)\n",
    "        \n",
    "        avg_val_loss = val_loss/len(val_loader)\n",
    "        scheduler.step(avg_val_loss)\n",
    "\n",
    "        acc = accuracy_score(all_labels, all_preds)\n",
    "        prec = precision_score(all_labels, all_preds, zero_division=0)\n",
    "        rec = recall_score(all_labels, all_preds, zero_division=0)\n",
    "        f1 = f1_score(all_labels, all_preds, zero_division=0)\n",
    "        auc = roc_auc_score(all_labels, all_probs)\n",
    "\n",
    "        print(f\"Epoch {epoch+1}/{epochs} | Train Loss: {running_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}\")\n",
    "        print(f\"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}\\n\")\n",
    "        \n",
    "    return model, all_labels, all_preds, all_probs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Training FAST ResNet18 Model...\")\n",
    "resnet_model = PneumoniaResNet()\n",
    "trained_model, labels, preds, probs = train_and_evaluate(resnet_model, train_loader, val_loader)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# SAVE THE MODEL FOR WEB APP\n",
    "print(\"Saving the trained model weights...\")\n",
    "torch.save(trained_model.state_dict(), 'pneumonia_model.pth')\n",
    "print(\"Model saved successfully as pneumonia_model.pth\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "class GradCAM:\n",
    "    def __init__(self, model, target_layer):\n",
    "        self.model = model\n",
    "        self.target_layer = target_layer\n",
    "        self.gradients = None\n",
    "        self.activations = None\n",
    "        \n",
    "        target_layer.register_forward_hook(self.save_activation)\n",
    "        target_layer.register_full_backward_hook(self.save_gradient)\n",
    "\n",
    "    def save_activation(self, module, input, output):\n",
    "        self.activations = output\n",
    "\n",
    "    def save_gradient(self, module, grad_input, grad_output):\n",
    "        self.gradients = grad_output[0]\n",
    "\n",
    "    def generate(self, x):\n",
    "        self.model.eval()\n",
    "        output = self.model(x)\n",
    "        self.model.zero_grad()\n",
    "        output.backward()\n",
    "        \n",
    "        gradients = self.gradients.cpu().data.numpy()[0]\n",
    "        activations = self.activations.cpu().data.numpy()[0]\n",
    "        \n",
    "        weights = np.mean(gradients, axis=(1, 2))\n",
    "        cam = np.zeros(activations.shape[1:], dtype=np.float32)\n",
    "        \n",
    "        for i, w in enumerate(weights):\n",
    "            cam += w * activations[i]\n",
    "            \n",
    "        cam = np.maximum(cam, 0)\n",
    "        cam = cv2.resize(cam, (x.shape[-1], x.shape[-2]))\n",
    "        cam = cam - np.min(cam)\n",
    "        cam = cam / np.max(cam)\n",
    "        return cam\n",
    "\n",
    "def plot_gradcam(model, image_tensor, original_image):\n",
    "    target_layer = model.resnet.layer4[-1]\n",
    "    cam = GradCAM(model, target_layer)\n",
    "    heatmap = cam.generate(image_tensor.unsqueeze(0).to(device))\n",
    "    \n",
    "    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)\n",
    "    heatmap = np.float32(heatmap) / 255\n",
    "    \n",
    "    inv_normalize = transforms.Normalize(\n",
    "        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],\n",
    "        std=[1/0.229, 1/0.224, 1/0.225]\n",
    "    )\n",
    "    original_np = inv_normalize(original_image).permute(1, 2, 0).numpy()\n",
    "    original_np = np.clip(original_np, 0, 1)\n",
    "    \n",
    "    cam_img = heatmap * 0.4 + original_np * 0.6\n",
    "    cam_img = cam_img / np.max(cam_img)\n",
    "    \n",
    "    fig, axes = plt.subplots(1, 2, figsize=(10, 5))\n",
    "    axes[0].imshow(original_np)\n",
    "    axes[0].set_title('Original X-Ray')\n",
    "    axes[0].axis('off')\n",
    "    \n",
    "    axes[1].imshow(cam_img)\n",
    "    axes[1].set_title('Grad-CAM Heatmap (ResNet18)')\n",
    "    axes[1].axis('off')\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Generating Grad-CAM visualization for a sample image...\")\n",
    "sample_img, sample_label = next(iter(val_loader))\n",
    "plot_gradcam(trained_model, sample_img[0], sample_img[0])"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('c:/projects/DL-PNEUMONIA/DL-PNEUMONIa/pneumonia_advanced.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

# Also update app.py to match ResNet18
with open('c:/projects/DL-PNEUMONIA/DL-PNEUMONIa/app.py', 'r', encoding='utf-8') as f:
    app_content = f.read()

app_content = app_content.replace('resnet50', 'resnet18')

with open('c:/projects/DL-PNEUMONIA/DL-PNEUMONIa/app.py', 'w', encoding='utf-8') as f:
    f.write(app_content)

