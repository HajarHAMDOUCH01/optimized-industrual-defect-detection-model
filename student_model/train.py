"""
Training script for TinyDefectNet Student Model

Implements knowledge distillation from teacher model with deployment constraints

Usage:
    python train.py --data_dir /path/to/data --teacher_checkpoint teacher_best.pth
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import sys

from config import Config
from dataset import create_dataloaders
from model import create_student_model
from trainer import DistillationTrainer
from utils import (
    set_seed,
    EarlyStopping,
    verify_deployment_constraints,
    print_constraint_report,
    test_robustness,
    benchmark_latency
)


def load_teacher_model(checkpoint_path: Path, num_classes: int, device: str):
    """Load teacher model from checkpoint"""
    print(f"\nLoading teacher model from {checkpoint_path}...")
    
    try:
        # Import teacher model architecture
        # NOTE: This assumes teacher model is in parent directory
        # Adjust path as needed for your setup
        sys.path.insert(0, str(Path(__file__).parent.parent / "teacher"))
        from model import TeacherModel
        
        # Create teacher model
        teacher = TeacherModel(
            num_classes=num_classes,
            pretrained=False,
            dropout_rate=0.2,
            freeze_backbone=False
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        teacher.load_state_dict(checkpoint['model_state_dict'])
        teacher.to(device)
        teacher.eval()
        
        print(f"✓ Teacher model loaded successfully")
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val accuracy: {checkpoint.get('val_acc', 0):.2f}%")
        
        return teacher
    
    except Exception as e:
        print(f"✗ Failed to load teacher model: {e}")
        print("  Training will continue without knowledge distillation")
        return None


def create_optimizer(model: nn.Module, lr: float, weight_decay: float):
    """Create optimizer"""
    if Config.OPTIMIZER.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=Config.BETAS,
            eps=Config.EPS,
            weight_decay=weight_decay
        )
    elif Config.OPTIMIZER.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=Config.BETAS,
            eps=Config.EPS,
            weight_decay=weight_decay
        )
    elif Config.OPTIMIZER.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True
        )
    else:
        raise ValueError(f"Unknown optimizer: {Config.OPTIMIZER}")
    
    return optimizer


def create_scheduler(optimizer, num_epochs: int):
    """Create learning rate scheduler"""
    if Config.SCHEDULER.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=Config.LR_MIN
        )
    elif Config.SCHEDULER.lower() == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=num_epochs // 3,
            gamma=0.1
        )
    else:
        scheduler = None
    
    return scheduler


def export_to_onnx(model, save_path, input_size=(1, 3, 224, 224)):
    """Export model to ONNX (fixed version)"""
    import onnx
    
    # IMPORTANT: Move model to CPU for ONNX export
    model = model.cpu()
    model.eval()
    
    dummy_input = torch.randn(input_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"✓ ONNX model exported to {save_path}")




def main(args):
    """Main training function"""
    
    # Set seed
    set_seed(Config.SEED)
    
    # Display configuration
    Config.display()
    
    # Create directories
    Config.create_dirs()
    
    # Update config from args
    if args.num_classes:
        Config.NUM_CLASSES = args.num_classes
    if args.data_dir:
        Config.DATA_ROOT = Path(args.data_dir)
        Config.TRAIN_DIR = Config.DATA_ROOT / "train"
        Config.VAL_DIR = Config.DATA_ROOT / "val"
    if args.teacher_checkpoint:
        Config.TEACHER_CHECKPOINT = Path(args.teacher_checkpoint)
    
    print(f"\nLoading data from: {Config.DATA_ROOT}")
    
    # Create dataloaders
    train_loader, val_loader, class_to_idx = create_dataloaders(
        train_dir=Config.TRAIN_DIR,
        val_dir=Config.VAL_DIR,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        val_split=Config.VAL_SPLIT
    )
    
    print(f"\nClass mapping: {class_to_idx}")
    
    # Create student model
    print(f"\n{'='*80}")
    print("Creating Student Model")
    print(f"{'='*80}")
    
    student = create_student_model(num_classes=Config.NUM_CLASSES)
    
    # Verify deployment constraints
    print(f"\nPre-training Constraint Check:")
    results = verify_deployment_constraints(student)
    print_constraint_report(results)
    
    if not results['all_constraints_met']:
        print("\n⚠ WARNING: Model does not meet deployment constraints!")
        if not args.force:
            print("Use --force to train anyway")
            sys.exit(1)
    
    # Load teacher model (if using KD)
    teacher = None
    if Config.USE_KD and Config.TEACHER_CHECKPOINT.exists():
        teacher = load_teacher_model(
            Config.TEACHER_CHECKPOINT,
            Config.NUM_CLASSES,
            Config.DEVICE
        )
    elif Config.USE_KD:
        print(f"\n⚠ Teacher checkpoint not found: {Config.TEACHER_CHECKPOINT}")
        print("  Training without knowledge distillation")
        Config.USE_KD = False
    
    # Create optimizer
    optimizer = create_optimizer(
        model=student,
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Create scheduler
    scheduler = create_scheduler(
        optimizer=optimizer,
        num_epochs=Config.EPOCHS
    )
    
    # Create trainer
    trainer = DistillationTrainer(
        student_model=student,
        teacher_model=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        checkpoint_dir=Config.CHECKPOINTS_DIR,
        log_dir=Config.LOGS_DIR,
        use_kd=Config.USE_KD,
        temperature=Config.KD_TEMPERATURE,
        alpha=Config.KD_ALPHA,
        mixed_precision=Config.MIXED_PRECISION
    )
    
    # Early stopping
    early_stopping = None
    if Config.EARLY_STOPPING:
        early_stopping = EarlyStopping(
            patience=Config.EARLY_STOPPING_PATIENCE,
            min_delta=Config.EARLY_STOPPING_MIN_DELTA,
            mode='max'
        )
    
    # Train
    print(f"\n{'='*80}")
    print("Starting Training")
    print(f"{'='*80}")
    
    best_metrics = trainer.train(
        num_epochs=Config.EPOCHS,
        early_stopping=early_stopping,
        verify_constraints=Config.VERIFY_CONSTRAINTS
    )
    
    # Save metrics
    trainer.metrics_tracker.save()
    
    # Post-training evaluation
    print(f"\n{'='*80}")
    print("POST-TRAINING EVALUATION")
    print(f"{'='*80}")
    
    # Load best model
    best_checkpoint = Config.CHECKPOINTS_DIR / 'best_student.pth'
    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location=Config.DEVICE)
        student.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded best model (Epoch {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.2f}%)")
    
    # Final constraint verification
    print(f"\nFinal Deployment Constraints:")
    results = verify_deployment_constraints(student)
    print_constraint_report(results)
    
    # Benchmark latency
    if Config.BENCHMARK_CPU:
        print(f"\nBenchmarking CPU Latency:")
        latency = benchmark_latency(
            student,
            device='cpu',
            num_runs=Config.NUM_BENCHMARK_RUNS
        )
        print(f"  Mean: {latency['mean_ms']:.2f} ms")
        print(f"  Median: {latency['median_ms']:.2f} ms")
        print(f"  P95: {latency['p95_ms']:.2f} ms")
        print(f"  P99: {latency['p99_ms']:.2f} ms")
        
        if latency['mean_ms'] <= Config.MAX_LATENCY_MS:
            print(f"  ✓ Within latency limit ({Config.MAX_LATENCY_MS} ms)")
        else:
            print(f"  ✗ Exceeds latency limit ({Config.MAX_LATENCY_MS} ms)")
    
    # Test robustness
    if Config.TEST_ROBUSTNESS:
        print(f"\nTesting Robustness:")
        robustness = test_robustness(
            student,
            val_loader,
            device=Config.DEVICE,
            noise_levels=Config.NOISE_LEVELS,
            blur_kernels=Config.BLUR_KERNELS
        )
        
        print(f"  Clean: {robustness['clean_accuracy']:.2f}%")
        for key, val in robustness.items():
            if 'noise' in key or 'blur' in key:
                print(f"  {key}: {val:.2f}%")
    
    # Export to ONNX
    if Config.EXPORT_ONNX:
        onnx_path = Config.CHECKPOINTS_DIR / 'student_model.onnx'
        print(f"\nExporting to ONNX:")
        export_to_onnx(student, onnx_path)
    
    # Summary
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Best Validation Accuracy: {best_metrics['best_val_acc']:.2f}%")
    print(f"Best Epoch: {best_metrics['best_epoch']}")
    print(f"\nModel saved to: {Config.CHECKPOINTS_DIR / 'best_student.pth'}")
    print(f"Logs saved to: {Config.LOGS_DIR}")
    
    if results['all_constraints_met']:
        print(f"\n✓ Model meets all deployment constraints!")
        print(f"  Parameters: {results['params']:,} / {results['params_limit']:,}")
        print(f"  Size (int8): {results['size_int8_mb']:.2f} MB / {results['size_limit_mb']:.2f} MB")
        if 'latency_cpu_ms' in results:
            print(f"  Latency: {results['latency_cpu_ms']:.2f} ms / {results['latency_limit_ms']:.2f} ms")
        print(f"\n Ready for deployment!")
    else:
        print(f"\n⚠ Model needs optimization to meet deployment constraints")
        print(f"  Consider: pruning, quantization, or architecture changes")
    
    print(f"{'='*80}")
    
    return best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TinyDefectNet Student Model with Knowledge Distillation"
    )
    
    # Data arguments
    parser.add_argument(
        '--data_dir',
        type=str,
        default="/kaggle/input/neu-metal-surface-defects-data/NEU Metal Surface Defects Data",
        help='Path to data directory'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=None,
        help='Number of classes'
    )
    
    # Teacher arguments
    parser.add_argument(
        '--teacher_checkpoint',
        type=str,
        default="/kaggle/working/optimized-industrual-defect-detection-model/teacher_model/checkpoints/teacher_final.pth",
        help='Path to teacher model checkpoint'
    )
    parser.add_argument(
        '--no_kd',
        action='store_true',
        help='Disable knowledge distillation'
    )
    
    # Training arguments
    parser.add_argument(
        '--force',
        action='store_true',
        help='Train even if constraints are violated'
    )
    
    args = parser.parse_args()
    
    # Disable KD if requested
    if args.no_kd:
        Config.USE_KD = False
    
    try:
        metrics = main(args)
        print("\n✓ Training completed successfully!")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)