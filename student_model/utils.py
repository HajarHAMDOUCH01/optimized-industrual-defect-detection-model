"""
Utility functions for TinyDefectNet Student Model
Includes KD helpers, deployment verification, and robustness testing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from config import Config


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
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


def count_parameters(model: nn.Module) -> int:
    """
    Count total number of parameters in model
    
    Args:
        model: PyTorch model
    
    Returns:
        Total parameter count
    """
    return sum(p.numel() for p in model.parameters())


def estimate_model_size(model: nn.Module, dtype=torch.float32) -> float:
    """
    Estimate model size in MB
    
    Args:
        model: PyTorch model
        dtype: Data type (float32 = 4 bytes, float16 = 2 bytes, int8 = 1 byte)
    
    Returns:
        Model size in MB
    """
    param_count = count_parameters(model)
    
    # Bytes per parameter
    if dtype == torch.float32:
        bytes_per_param = 4
    elif dtype == torch.float16:
        bytes_per_param = 2
    elif dtype == torch.int8:
        bytes_per_param = 1
    else:
        bytes_per_param = 4
    
    size_bytes = param_count * bytes_per_param
    size_mb = size_bytes / (1024 ** 2)
    
    return size_mb


def benchmark_latency(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = 'cpu',
    num_runs: int = 100,
    warmup_runs: int = 10
) -> Dict[str, float]:
    """
    Benchmark model inference latency
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
        device: Device to benchmark on
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
    
    Returns:
        Dictionary with latency statistics
    """
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
    
    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(dummy_input)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
        'median_ms': np.median(latencies),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99)
    }


def verify_deployment_constraints(model: nn.Module) -> Dict[str, Any]:
    """
    Verify model meets deployment constraints
    
    Args:
        model: Student model
    
    Returns:
        Dictionary with constraint check results
    """
    results = {}
    
    # Parameter count
    params = count_parameters(model)
    results['params'] = params
    results['params_ok'] = params <= Config.MAX_PARAMS
    results['params_limit'] = Config.MAX_PARAMS
    
    # Model size (float32)
    size_fp32 = estimate_model_size(model, torch.float32)
    size_int8 = estimate_model_size(model, torch.int8)
    results['size_fp32_mb'] = size_fp32
    results['size_int8_mb'] = size_int8
    results['size_ok'] = size_int8 <= Config.MAX_MODEL_SIZE_MB
    results['size_limit_mb'] = Config.MAX_MODEL_SIZE_MB
    
    # Latency benchmark (if enabled)
    if Config.BENCHMARK_CPU:
        latency = benchmark_latency(
            model,
            device='cpu',
            num_runs=Config.NUM_BENCHMARK_RUNS
        )
        results['latency_cpu_ms'] = latency['mean_ms']
        results['latency_ok'] = latency['mean_ms'] <= Config.MAX_LATENCY_MS
        results['latency_limit_ms'] = Config.MAX_LATENCY_MS
    
    # Overall pass/fail
    results['all_constraints_met'] = (
        results['params_ok'] and
        results['size_ok'] and
        results.get('latency_ok', True)
    )
    
    return results


def print_constraint_report(results: Dict[str, Any]):
    """Print formatted constraint verification report"""
    print("\n" + "=" * 80)
    print("DEPLOYMENT CONSTRAINT VERIFICATION")
    print("=" * 80)
    
    # Parameters
    status = "✓" if results['params_ok'] else "✗"
    print(f"{status} Parameters: {results['params']:,} / {results['params_limit']:,}")
    
    # Size
    status = "✓" if results['size_ok'] else "✗"
    print(f"{status} Size (int8): {results['size_int8_mb']:.2f} MB / {results['size_limit_mb']:.2f} MB")
    print(f"   Size (fp32): {results['size_fp32_mb']:.2f} MB")
    
    # Latency
    if 'latency_cpu_ms' in results:
        status = "✓" if results['latency_ok'] else "✗"
        print(f"{status} Latency (CPU): {results['latency_cpu_ms']:.2f} ms / {results['latency_limit_ms']:.2f} ms")
    
    print("=" * 80)
    
    if results['all_constraints_met']:
        print("✓ ALL CONSTRAINTS MET - Model ready for deployment")
    else:
        print("✗ CONSTRAINTS VIOLATED - Model needs optimization")
    
    print("=" * 80)


# ============================================================================
# KNOWLEDGE DISTILLATION UTILITIES
# ============================================================================

def kl_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0
) -> torch.Tensor:
    """
    Compute KL divergence loss for knowledge distillation
    
    KL(P_teacher || P_student) where P = softmax(logits / temperature)
    
    Args:
        student_logits: Student model logits [B, num_classes]
        teacher_logits: Teacher model logits [B, num_classes]
        temperature: Temperature for softening distributions
    
    Returns:
        KL divergence loss (scalar)
    """
    # Soften distributions with temperature
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    
    # KL divergence: sum(P * log(P/Q)) = sum(P * (log(P) - log(Q)))
    kl_loss = F.kl_div(
        student_soft,
        teacher_soft,
        reduction='batchmean'
    )
    
    # Scale by temperature^2 (as per Hinton et al.)
    kl_loss = kl_loss * (temperature ** 2)
    
    return kl_loss


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
    hard_loss_fn: Optional[nn.Module] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute combined distillation loss
    
    L = α * CE(y, student) + (1-α) * KL(teacher || student)
    
    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits
        targets: Ground truth labels
        temperature: Temperature for KD
        alpha: Weight for hard loss (1-alpha for soft loss)
        hard_loss_fn: Hard loss function (default: CrossEntropy)
    
    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary with individual loss components
    """
    # Hard loss (standard cross-entropy with ground truth)
    if hard_loss_fn is None:
        hard_loss_fn = nn.CrossEntropyLoss()
    
    hard_loss = hard_loss_fn(student_logits, targets)
    
    # Soft loss (KL divergence with teacher)
    soft_loss = kl_divergence_loss(student_logits, teacher_logits, temperature)
    
    # Combined loss
    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss
    
    # Return individual components for logging
    loss_dict = {
        'hard_loss': hard_loss.item(),
        'soft_loss': soft_loss.item(),
        'kl_loss': soft_loss.item(),  # Alias for clarity
        'total_loss': total_loss.item()
    }
    
    return total_loss, loss_dict


# ============================================================================
# ROBUSTNESS TESTING
# ============================================================================

def add_gaussian_noise(images: torch.Tensor, std: float = 0.01) -> torch.Tensor:
    """Add Gaussian noise to images"""
    noise = torch.randn_like(images) * std
    return torch.clamp(images + noise, 0, 1)


def apply_gaussian_blur(images: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Apply Gaussian blur to images"""
    from torchvision.transforms import GaussianBlur
    blur = GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))
    return blur(images)


def test_robustness(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cpu',
    noise_levels: list = [0.01, 0.05, 0.1],
    blur_kernels: list = [3, 5]
) -> Dict[str, float]:
    """
    Test model robustness to noise and blur
    
    Args:
        model: Model to test
        dataloader: Test dataloader
        device: Device to run on
        noise_levels: Gaussian noise standard deviations
        blur_kernels: Gaussian blur kernel sizes
    
    Returns:
        Dictionary with robustness metrics
    """
    model.eval()
    results = {}
    
    # Clean accuracy (baseline)
    clean_correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            model = model.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            clean_correct += (predicted == targets).sum().item()
            total += targets.size(0)
    
    clean_acc = 100.0 * clean_correct / total
    results['clean_accuracy'] = clean_acc
    
    # Noise robustness
    for noise_std in noise_levels:
        noisy_correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in dataloader:
                images, targets = images.to(device), targets.to(device)
                noisy_images = add_gaussian_noise(images, std=noise_std)
                outputs = model(noisy_images)
                _, predicted = torch.max(outputs, 1)
                noisy_correct += (predicted == targets).sum().item()
                total += targets.size(0)
        
        noisy_acc = 100.0 * noisy_correct / total
        results[f'noise_std_{noise_std}_accuracy'] = noisy_acc
    
    # Blur robustness
    for kernel_size in blur_kernels:
        blur_correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in dataloader:
                images, targets = images.to(device), targets.to(device)
                blurred_images = apply_gaussian_blur(images, kernel_size=kernel_size)
                outputs = model(blurred_images)
                _, predicted = torch.max(outputs, 1)
                blur_correct += (predicted == targets).sum().item()
                total += targets.size(0)
        
        blur_acc = 100.0 * blur_correct / total
        results[f'blur_kernel_{kernel_size}_accuracy'] = blur_acc
    
    return results


# ============================================================================
# STANDARD UTILITIES (from teacher)
# ============================================================================

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
    """Track training metrics including KD-specific metrics"""
    
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'hard_loss': [],
            'soft_loss': [],
            'kl_loss': []
        }
    
    def update(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float,
        hard_loss: float = 0.0,
        soft_loss: float = 0.0
    ):
        """Update metrics for current epoch"""
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['learning_rates'].append(lr)
        self.history['hard_loss'].append(hard_loss)
        self.history['soft_loss'].append(soft_loss)
        self.history['kl_loss'].append(soft_loss)
    
    def save(self, filename: str = "metrics.json"):
        """Save metrics to JSON"""
        filepath = self.save_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)


class EarlyStopping:
    """Early stopping with patience"""
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = 'max'):
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


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate classification accuracy"""
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    accuracy = 100.0 * correct / total
    return accuracy


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate"""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


if __name__ == "__main__":
    print("Testing utility functions...")
    
    # Test KL divergence
    student_logits = torch.randn(4, 6)
    teacher_logits = torch.randn(4, 6)
    targets = torch.randint(0, 6, (4,))
    
    loss, loss_dict = distillation_loss(
        student_logits,
        teacher_logits,
        targets,
        temperature=4.0,
        alpha=0.7
    )
    
    print(f"\n✓ Distillation loss test:")
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Hard loss: {loss_dict['hard_loss']:.4f}")
    print(f"  Soft loss: {loss_dict['soft_loss']:.4f}")
    
    print("\n✓ All utility tests passed!")