import json

with open('c:/projects/DL-PNEUMONIA/DL-PNEUMONIa/pneumonia_advanced.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

confusion_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# CONFUSION MATRIX\n",
        "from sklearn.metrics import confusion_matrix\n",
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "\n",
        "flat_labels = [int(l) for l in labels]\n",
        "flat_preds  = [int(p) for p in preds]\n",
        "\n",
        "cm = confusion_matrix(flat_labels, flat_preds)\n",
        "\n",
        "plt.figure(figsize=(7, 5))\n",
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n",
        "            xticklabels=['Normal', 'Pneumonia'],\n",
        "            yticklabels=['Normal', 'Pneumonia'],\n",
        "            linewidths=1, linecolor='black')\n",
        "plt.title('Confusion Matrix - Pneumonia Detection', fontsize=14, fontweight='bold')\n",
        "plt.ylabel('Actual Label', fontsize=12)\n",
        "plt.xlabel('Predicted Label', fontsize=12)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "print('TN:', cm[0][0], '| FP:', cm[0][1])\n",
        "print('FN:', cm[1][0], '| TP:', cm[1][1])"
    ]
}

roc_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ROC CURVE\n",
        "from sklearn.metrics import roc_curve, auc\n",
        "\n",
        "flat_probs = [float(p) for p in probs]\n",
        "\n",
        "fpr, tpr, thresholds = roc_curve(flat_labels, flat_probs)\n",
        "roc_auc = auc(fpr, tpr)\n",
        "\n",
        "plt.figure(figsize=(7, 5))\n",
        "plt.plot(fpr, tpr, color='#0056b3', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')\n",
        "plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')\n",
        "plt.fill_between(fpr, tpr, alpha=0.1, color='#0056b3')\n",
        "plt.xlim([0.0, 1.0])\n",
        "plt.ylim([0.0, 1.05])\n",
        "plt.xlabel('False Positive Rate', fontsize=12)\n",
        "plt.ylabel('True Positive Rate', fontsize=12)\n",
        "plt.title('ROC Curve - Pneumonia Detection (ResNet18)', fontsize=14, fontweight='bold')\n",
        "plt.legend(loc='lower right', fontsize=11)\n",
        "plt.grid(alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "print(f'AUC Score: {roc_auc:.4f}')"
    ]
}

# Find the SAVE MODEL cell index and insert after it
save_cell_index = None
for i, cell in enumerate(notebook['cells']):
    if any('SAVE THE MODEL' in line for line in cell.get('source', [])):
        save_cell_index = i
        break

if save_cell_index is not None:
    notebook['cells'].insert(save_cell_index + 1, roc_cell)
    notebook['cells'].insert(save_cell_index + 1, confusion_cell)
    print(f"Inserted Confusion Matrix and ROC Curve cells after cell index {save_cell_index}")
else:
    print("Save cell not found! Appending at end.")
    notebook['cells'].append(confusion_cell)
    notebook['cells'].append(roc_cell)

with open('c:/projects/DL-PNEUMONIA/DL-PNEUMONIa/pneumonia_advanced.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated successfully!")
