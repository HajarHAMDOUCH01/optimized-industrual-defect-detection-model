---
license: apache-2.0
metrics:
- accuracy
- recall
- f1
- precision
base_model:
- google/efficientnet-b0
pipeline_tag: image-classification
tags:
- defect-detection
- industry
- knowledge distillation
---
# Industrial Defect Detection (Teacher → Student)

A lightweight, production-ready computer vision system for real-time industrial surface defect detection under strict deployment constraints (CPU-only, low memory -612KB-, low latency -3.2ms-).

![student model architecture](https://github.com/HajarHAMDOUCH01/optimized-industrual-defect-detection-model/blob/0f5078ab24c7632442465ac2ebaeaa4fd13379b3/model.png)


## Detailed Performance Metrics

The student model achieves excellent performance across all evaluation metrics:

### Overall Statistics (72 test samples)

| Metric | Value |
|--------|-------|
| Accuracy | 95.83% |
| Macro Precision | 96.15% |
| Macro Recall | 95.83% |
| Macro F1-Score | 95.62% |
| Weighted Precision | 96.15% |
| Weighted Recall | 95.83% |
| Weighted F1-Score | 95.62% |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Crazing | 92.31% | 100.0% | 96.00% |
| Inclusion | 92.31% | 100.0% | 96.00% |
| Patches | 100.0% | 100.0% | 100.0% |
| Pitted Surface | 100.0% | 75.00% | 85.71% |
| Rolled-in Scale | 92.31% | 100.0% | 96.00% |
| Scratches | 100.0% | 100.0% | 100.0% |




## Overview

This repository implements a **teacher–student knowledge distillation pipeline** where a strong CNN teacher is first trained, then a tiny student network is distilled and pruned for real-world deployment. The resulting model achieves high accuracy while being optimized for resource-constrained environments.

**Key Features:**
- Real-time inference on CPU
- Compact model size (<5MB)
- Deterministic and stable predictions
- Robust to noise, texture, and lighting variations
- Optimized via knowledge distillation + structured pruning

## Dataset

The model is trained on the **NEU Surface Defect Dataset**, which contains texture-based grayscale steel surface images with six defect classes:


| Class | Description |
|-------|-------------|
| **Crazing** | Fine network of cracks |
| **Inclusion** | Embedded foreign particles |
| **Patches** | Surface patches |
| **Pitted Surface** | Small holes or pits |
| **Rolled-in Scale** | Oxide scale rolled into surface |
| **Scratches** | Linear surface scratches |

**Dataset Characteristics:**
- 6 defect classes
- Texture-based grayscale images
- Small dataset size with high intra-class variation
- Industrial-relevant surface defects

## Model Architecture

### Teacher Model
- **Backbone**: EfficientNet-B0 (ImageNet pretrained)
- **Training Strategy**: Two-stage fine-tuning
  1. Freeze backbone, train classifier head
  2. Unfreeze final blocks for domain adaptation
- **Output**: Stable logits used for distillation

### Student Model 
A custom TinyCNN designed specifically for CPU deployment:

**Training Pipeline:**
1. **Knowledge Distillation**: Student learns from both ground-truth labels and teacher's soft probabilities
   ```
   L = α·ℓ(y, pₛ) + (1-α)·KL(pₜᵗ / pₛᵗ)
   ```
   where τ is temperature scaling

2. **Structured Channel Pruning**: Remove low-importance convolution channels using L1-norm ranking

3. **Fine-tuning**: Retrain pruned model to recover accuracy

## Quick Start: ONNX Inference

### Installation
```bash
pip install onnxruntime pillow torchvision numpy
```

### Inference Code
```python
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image
import torchvision.transforms as T
import json

# Load configuration
config_file_path = hf_hub_download(
    repo_id="hajar001/optimized-industrual-defect-detection",
    filename="config.json"
)

with open(config_file_path) as f:
    cfg = json.load(f)

# Configuration parameters
REPO_ID = cfg["repo_id"]
ONNX_FILE = cfg["onnx_file"]
IMG_SIZE = cfg["img_size"]
MEAN = cfg["mean"]
STD = cfg["std"]
CLASS_NAMES = cfg["class_names"]
providers = cfg["providers"]

# Download and load ONNX model
onnx_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=ONNX_FILE
)

session = ort.InferenceSession(onnx_path, providers=providers)
input_name = session.get_inputs()[0].name

# Preprocessing transform
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

def preprocess(image_path):
    """Preprocess image for model inference."""
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)  # (1, 3, 224, 224)
    return x.numpy().astype(np.float32)

def predict(image_path):
    """Run inference on a single image."""
    x = preprocess(image_path)
    
    outputs = session.run(
        None,
        {input_name: x}
    )
    
    logits = outputs[0]
    pred = np.argmax(logits, axis=1)[0]
    
    return pred

# Example usage
if __name__ == "__main__":
    image_path = "/kaggle/input/neu-metal-surface-defects-data/NEU Metal Surface Defects Data/test/Rolled/RS_1.bmp"
    pred_idx = predict(image_path)
    
    print(f"Prediction: {CLASS_NAMES[pred_idx]}")
```

## Model Performance

| Metric | Teacher Model | Student Model (Pruned) |
|--------|---------------|------------------------|
| **Size** | ~46.1MB | **612KB** |
| **Inference Speed** | ~50.0ms (GPU) | **~3.2 ms (CPU)** |
| **Accuracy** | 98.4% | 98.33% |
| **Parameters** | 2,000,000 | **144,342** |

## Training Pipeline

1. **Teacher Training**: Train EfficientNet-B0 on NEU dataset
2. **Knowledge Distillation**: Transfer knowledge to TinyCNN
3. **Pruning**: Remove redundant channels
4. **Fine-tuning**: Recover accuracy loss from pruning

## License

Apache 2.0 License - See [LICENSE](LICENSE) file for details.


## Support

For issues, questions, or contributions:
- Open an issue on this GitHub repository

---

**Note**: This model is optimized for industrial surface defect detection and may require fine-tuning for other defect detection tasks.
