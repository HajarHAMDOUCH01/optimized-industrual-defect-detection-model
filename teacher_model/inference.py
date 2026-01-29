#!/usr/bin/env python3
"""
Inference script for trained TinyDefectNet Teacher Model

Usage:
    python inference.py --checkpoint checkpoints/best_model.pth --image test.jpg
    python inference.py --checkpoint checkpoints/best_model.pth --image_dir test_images/
"""

import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import json

from config import Config
from model import TeacherModel, load_checkpoint
from dataset import get_transforms


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load trained teacher model"""
    # Create model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = checkpoint['config']['num_classes']
    
    model = TeacherModel(
        num_classes=num_classes,
        pretrained=False,
        dropout_rate=Config.DROPOUT_RATE,
        freeze_backbone=False
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Acc: {checkpoint.get('val_acc', 0):.2f}%")
    
    return model, num_classes


def predict_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: list,
    device: str = 'cpu',
    temperature: float = 1.0
) -> dict:
    """
    Predict single image
    
    Args:
        model: Trained model
        image_path: Path to image
        class_names: List of class names
        device: Device to run on
        temperature: Temperature for softmax
    
    Returns:
        Dictionary with predictions
    """
    # Load and preprocess image
    transform = get_transforms('val')
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(image_tensor)
        
        # Standard probabilities (T=1)
        probs = F.softmax(logits, dim=1)[0]
        
        # Soft probabilities (T>1 for KD)
        soft_logits = logits / temperature
        soft_probs = F.softmax(soft_logits, dim=1)[0]
        
        # Get prediction
        pred_idx = torch.argmax(probs).item()
        pred_class = class_names[pred_idx]
        confidence = probs[pred_idx].item()
    
    # Format results
    results = {
        'predicted_class': pred_class,
        'predicted_idx': pred_idx,
        'confidence': confidence,
        'probabilities': {
            class_names[i]: probs[i].item()
            for i in range(len(class_names))
        },
        'soft_probabilities': {
            class_names[i]: soft_probs[i].item()
            for i in range(len(class_names))
        },
        'logits': logits[0].cpu().tolist()
    }
    
    return results


def main(args):
    """Main inference function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, num_classes = load_model(args.checkpoint, device)
    
    # Load class names
    if args.class_names:
        with open(args.class_names, 'r') as f:
            class_names = json.load(f)
    else:
        # Default class names
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    print(f"Classes: {class_names}")
    
    # Single image inference
    if args.image:
        print(f"\nPredicting: {args.image}")
        results = predict_image(
            model=model,
            image_path=args.image,
            class_names=class_names,
            device=device,
            temperature=args.temperature
        )
        
        # Print results
        print("\n" + "="*60)
        print(f"Prediction: {results['predicted_class']}")
        print(f"Confidence: {results['confidence']:.2%}")
        print("\nProbabilities:")
        for cls, prob in results['probabilities'].items():
            bar = '█' * int(prob * 50)
            print(f"  {cls:20s} {prob:6.2%} {bar}")
        
        if args.temperature > 1.0:
            print(f"\nSoft Probabilities (T={args.temperature}):")
            for cls, prob in results['soft_probabilities'].items():
                bar = '█' * int(prob * 50)
                print(f"  {cls:20s} {prob:6.2%} {bar}")
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
    
    # Directory inference
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        results_all = []
        
        for img_path in image_dir.iterdir():
            if img_path.suffix.lower() in valid_extensions:
                print(f"\nPredicting: {img_path.name}")
                
                results = predict_image(
                    model=model,
                    image_path=str(img_path),
                    class_names=class_names,
                    device=device,
                    temperature=args.temperature
                )
                
                results['image_path'] = str(img_path)
                results_all.append(results)
                
                print(f"  → {results['predicted_class']} ({results['confidence']:.2%})")
        
        # Save batch results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results_all, f, indent=2)
            print(f"\nBatch results saved to {args.output}")
        
        # Print summary
        print("\n" + "="*60)
        print(f"Processed {len(results_all)} images")
        
        # Class distribution
        pred_counts = {}
        for r in results_all:
            pred_cls = r['predicted_class']
            pred_counts[pred_cls] = pred_counts.get(pred_cls, 0) + 1
        
        print("\nPrediction distribution:")
        for cls, count in sorted(pred_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(results_all)
            print(f"  {cls:20s}: {count:3d} ({pct:5.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference with TinyDefectNet Teacher Model"
    )
    
    # Model arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--class_names',
        type=str,
        default=None,
        help='Path to JSON file with class names'
    )
    
    # Input arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--image',
        type=str,
        help='Path to single image'
    )
    group.add_argument(
        '--image_dir',
        type=str,
        help='Path to directory of images'
    )
    
    # Output arguments
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save results (JSON)'
    )
    
    # Inference arguments
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Temperature for softmax (>1 for softer predictions)'
    )
    
    args = parser.parse_args()
    main(args)