"""
Complete Test Set Evaluation for TinyDefectNet
Calculates accuracy, precision, recall, F1-score, and confusion matrix
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import json

from model import create_student_model
from dataset import DefectDataset, get_transforms


def evaluate_test_set(
    test_loader,
    class_names,
    device='cpu',
    save_dir='results'
):
    """
    Evaluate model on test set
    
    Args:
        test_loader: Test data loader
        class_names: List of class names
        device: Device to run on
        save_dir: Directory to save results
    
    Returns:
        Dictionary with all metrics
    """

    import onnxruntime as ort

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
    if torch.cuda.is_available() else ['CPUExecutionProvider']

    session = ort.InferenceSession(
    "/kaggle/working/tinydefectnet_student/checkpoints/student_model.onnx",
    providers=providers
    )

    input_name = session.get_inputs()[0].name


    all_preds = []
    all_labels = []
    all_probs = []

    print("Evaluating ONNX model on test set...")

    for images, labels in tqdm(test_loader, desc="Testing (ONNX)"):

        # PyTorch → NumPy
        images_np = images.numpy()  # shape: [B, 3, 224, 224]

        # ONNX inference
        outputs = session.run(None, {input_name: images_np})
        logits = outputs[0]  # shape: [B, num_classes]

        # Softmax (NumPy)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # --- metrics (unchanged) ---
    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "confusion_matrix": cm.tolist()
    }


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_normalized,
        annot=cm,  # Show raw counts
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'}
    )
    
    plt.title('Confusion Matrix\n(annotations show raw counts)', fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()


def plot_per_class_metrics(class_names, precision, recall, f1, save_path):
    """Plot per-class metrics"""
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=8
            )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Per-class metrics plot saved to {save_path}")
    plt.close()


# ============================================================================
# MAIN USAGE
# ============================================================================

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # Configuration
    CHECKPOINT_PATH = "/kaggle/working/tinydefectnet_student/checkpoints/student_model.onnx"
    TEST_DIR = Path("/kaggle/input/neu-metal-surface-defects-data/NEU Metal Surface Defects Data/test")
    NUM_CLASSES = 6
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    RESULTS_DIR = "/kaggle/working/test_results"
    
    # Class names (should match your dataset)
    CLASS_NAMES = [
        'Crazing',
        'Inclusion',
        'Patches',
        'Pitted_surface',
        'Rolled-in_scale',
        'Scratches'
    ]
    
    print("="*80)
    print("TinyDefectNet - Test Set Evaluation")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Test directory: {TEST_DIR}")
    print(f"  Device: {DEVICE}")
    print(f"  Batch size: {BATCH_SIZE}")
    
    # Load model
    print(f"\nLoading model...")
    model = create_student_model(num_classes=NUM_CLASSES)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Training epoch: {checkpoint['epoch']}")
    print(f"  Validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Create test dataset and loader
    print(f"\nLoading test dataset...")
    test_transform = get_transforms('val')  # Use validation transforms (no augmentation)
    test_dataset = DefectDataset(
        root_dir=TEST_DIR,
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    print(f"✓ Test dataset loaded")
    print(f"  Total samples: {len(test_dataset)}")
    print(f"  Classes: {test_dataset.classes}")
    print(f"  Class distribution:")
    distribution = test_dataset.get_class_distribution()
    for cls_name, count in distribution.items():
        print(f"    {cls_name}: {count} samples")
    
    # Evaluate
    results = evaluate_test_set(
        model=model,
        test_loader=test_loader,
        class_names=CLASS_NAMES,
        device=DEVICE,
        save_dir=RESULTS_DIR
    )
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"  - test_metrics.json")
    print(f"  - classification_report.txt")
    print(f"  - confusion_matrix.png")
    print(f"  - per_class_metrics.png")