"""
Utility functions for TinyDefectNet Teacher Model training
"""

import torch
import torch.nn as nn
import numpy as np
import random
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from config import Config


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if Config.DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class MetricsTracker:
    """Track training and validation metrics"""
    
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
    
    def update(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float
    ):
        """Update metrics for current epoch"""
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['learning_rates'].append(lr)
    
    def save(self, filename: str = "metrics.json"):
        """Save metrics to JSON file"""
        filepath = self.save_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Metrics saved to {filepath}")
    
    def get_best_epoch(self, metric: str = 'val_acc') -> int:
        """Get epoch with best metric value"""
        if metric not in self.history or len(self.history[metric]) == 0:
            return 0
        
        if 'acc' in metric:
            return np.argmax(self.history[metric])
        else:
            return np.argmin(self.history[metric])


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 1e-4,
        mode: str = 'max'
    ):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'max':
            self.is_better = lambda new, best: new > best + min_delta
        else:
            self.is_better = lambda new, best: new < best - min_delta
    
    def __call__(self, metric: float) -> bool:
        """
        Check if training should stop
        
        Args:
            metric: Current metric value
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = metric
            return False
        
        if self.is_better(metric, self.best_score):
            self.best_score = metric
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    save_path: Path,
    is_best: bool = False
):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        epoch: Current epoch
        train_loss: Training loss
        train_acc: Training accuracy
        val_loss: Validation loss
        val_acc: Validation accuracy
        save_path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'config': {
            'num_classes': Config.NUM_CLASSES,
            'image_size': Config.IMAGE_SIZE,
            'backbone': Config.BACKBONE
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save checkpoint
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    
    # Save best model separately
    if is_best:
        best_path = save_path.parent / 'best_model.pth'
        torch.save(checkpoint, best_path)
        print(f"✓ Best model saved to {best_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load checkpoint on
    
    Returns:
        Checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Acc: {checkpoint.get('val_acc', 0):.4f}")
    
    return checkpoint


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate classification accuracy
    
    Args:
        outputs: Model outputs (logits) [B, num_classes]
        targets: Ground truth labels [B]
    
    Returns:
        Accuracy as percentage
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    accuracy = 100.0 * correct / total
    return accuracy


class GradScalerWrapper:
    """Wrapper for gradient scaler with mixed precision training"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        if enabled:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def scale(self, loss):
        if self.enabled:
            return self.scaler.scale(loss)
        return loss
    
    def step(self, optimizer):
        if self.enabled:
            self.scaler.step(optimizer)
        else:
            optimizer.step()
    
    def update(self):
        if self.enabled:
            self.scaler.update()


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def print_training_header(phase: str, epoch: int, total_epochs: int):
    """Print formatted training header"""
    print("\n" + "="*80)
    print(f"{phase} - Epoch [{epoch}/{total_epochs}]")
    print("="*80)


def print_epoch_summary(
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    lr: float,
    epoch_time: float
):
    """Print formatted epoch summary"""
    print("\n" + "-"*80)
    print(f"Epoch {epoch} Summary:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    print(f"  LR: {lr:.6f} | Time: {epoch_time:.2f}s")
    print("-"*80)


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test seed setting
    set_seed(42)
    print(f"✓ Seed set to {Config.SEED}")
    
    # Test AverageMeter
    meter = AverageMeter("test_loss")
    meter.update(1.5, 32)
    meter.update(1.3, 32)
    meter.update(1.1, 32)
    print(f"✓ AverageMeter: {meter}")
    
    # Test MetricsTracker
    tracker = MetricsTracker(Config.LOGS_DIR)
    tracker.update(1, 1.5, 85.0, 1.2, 87.5, 0.001)
    tracker.update(2, 1.3, 88.0, 1.1, 89.0, 0.0008)
    tracker.save("test_metrics.json")
    print(f"✓ MetricsTracker saved to {Config.LOGS_DIR}")
    
    # Test EarlyStopping
    early_stopping = EarlyStopping(patience=3, mode='max')
    for val_acc in [87.5, 89.0, 88.5, 88.3, 88.0]:
        should_stop = early_stopping(val_acc)
        if should_stop:
            print(f"✓ EarlyStopping triggered at val_acc={val_acc:.2f}")
            break
    
    print("\nAll utility tests passed!")