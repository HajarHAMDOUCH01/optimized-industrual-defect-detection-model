#!/usr/bin/env python3
"""
Training script for TinyDefectNet Teacher Model

Two-phase training strategy:
    Phase 1: Train classification head with frozen backbone
    Phase 2: Fine-tune with unfrozen last blocks

Usage:
    python train.py --data_dir /path/to/data --num_classes 6
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import sys

from config import Config
from dataset import create_dataloaders
from model import create_teacher_model
from trainer import Trainer
from utils import set_seed, EarlyStopping


def create_optimizer(model: nn.Module, lr: float, weight_decay: float):
    """Create optimizer"""
    if Config.OPTIMIZER.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            betas=Config.BETAS,
            eps=Config.EPS,
            weight_decay=weight_decay
        )
    elif Config.OPTIMIZER.lower() == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            betas=Config.BETAS,
            eps=Config.EPS,
            weight_decay=weight_decay
        )
    elif Config.OPTIMIZER.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True
        )
    else:
        raise ValueError(f"Unknown optimizer: {Config.OPTIMIZER}")
    
    return optimizer


def create_scheduler(optimizer, num_epochs: int, warmup_epochs: int = 0):
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
    elif Config.SCHEDULER.lower() == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {Config.SCHEDULER}")
    
    return scheduler


def create_criterion(class_weights=None):
    """Create loss function"""
    if Config.LOSS_FN.lower() == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=Config.LABEL_SMOOTHING
        )
    else:
        raise ValueError(f"Unknown loss function: {Config.LOSS_FN}")
    
    return criterion


def main(args):
    """Main training function"""
    
    # Set seed for reproducibility
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
    
    print(f"\nLoading data from: {Config.DATA_ROOT}")
    
    # Create dataloaders
    train_loader, val_loader, class_to_idx = create_dataloaders(
        train_dir=Config.TRAIN_DIR,
        val_dir=Config.VAL_DIR,
        batch_size=Config.PHASE1_BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        val_split=Config.VAL_SPLIT
    )
    
    print(f"\nClass mapping: {class_to_idx}")
    
    # Compute class weights if needed
    class_weights = None
    if Config.CLASS_WEIGHTS is not None or args.compute_class_weights:
        print("\nComputing class weights for imbalanced dataset...")
        # Get class weights from training dataset
        class_weights = train_loader.dataset.compute_class_weights()
        class_weights = class_weights.to(Config.DEVICE)
        print(f"Class weights: {class_weights}")
    
    # =========================================================================
    # PHASE 1: TRAIN HEAD ONLY (FROZEN BACKBONE)
    # =========================================================================
    
    print("\n" + "="*80)
    print("PHASE 1: Training classification head (frozen backbone)")
    print("="*80)
    
    # Create model with frozen backbone
    model = create_teacher_model(
        num_classes=Config.NUM_CLASSES,
        pretrained=Config.PRETRAINED,
        dropout_rate=Config.DROPOUT_RATE,
        freeze_backbone=True
    )
    
    # Create optimizer for phase 1
    optimizer_phase1 = create_optimizer(
        model=model,
        lr=Config.PHASE1_LR,
        weight_decay=Config.PHASE1_WEIGHT_DECAY
    )
    
    # Create scheduler for phase 1
    scheduler_phase1 = create_scheduler(
        optimizer=optimizer_phase1,
        num_epochs=Config.PHASE1_EPOCHS,
        warmup_epochs=Config.WARMUP_EPOCHS
    )
    
    # Create criterion
    criterion = create_criterion(class_weights)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_phase1,
        scheduler=scheduler_phase1,
        device=Config.DEVICE,
        checkpoint_dir=Config.CHECKPOINTS_DIR,
        log_dir=Config.LOGS_DIR,
        mixed_precision=Config.MIXED_PRECISION
    )
    
    # Train phase 1
    early_stopping_phase1 = None
    if Config.EARLY_STOPPING:
        early_stopping_phase1 = EarlyStopping(
            patience=Config.EARLY_STOPPING_PATIENCE,
            min_delta=Config.EARLY_STOPPING_MIN_DELTA,
            mode='max'
        )
    
    phase1_metrics = trainer.train_phase(
        phase_name="Phase 1",
        num_epochs=Config.PHASE1_EPOCHS,
        early_stopping=early_stopping_phase1,
        start_epoch=1
    )
    
    # =========================================================================
    # PHASE 2: FINE-TUNE WITH UNFROZEN LAST BLOCKS
    # =========================================================================
    
    print("\n" + "="*80)
    print("PHASE 2: Fine-tuning with unfrozen last blocks")
    print("="*80)
    
    # Unfreeze last blocks
    model.unfreeze_backbone(from_block=Config.PHASE2_UNFREEZE_FROM_BLOCK)
    
    # Update dataloader batch size if different
    if Config.PHASE2_BATCH_SIZE != Config.PHASE1_BATCH_SIZE:
        train_loader, val_loader, _ = create_dataloaders(
            train_dir=Config.TRAIN_DIR,
            val_dir=Config.VAL_DIR,
            batch_size=Config.PHASE2_BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
            val_split=Config.VAL_SPLIT
        )
        trainer.train_loader = train_loader
        trainer.val_loader = val_loader
    
    # Create optimizer for phase 2 (lower learning rate)
    optimizer_phase2 = create_optimizer(
        model=model,
        lr=Config.PHASE2_LR,
        weight_decay=Config.PHASE2_WEIGHT_DECAY
    )
    
    # Create scheduler for phase 2
    scheduler_phase2 = create_scheduler(
        optimizer=optimizer_phase2,
        num_epochs=Config.PHASE2_EPOCHS,
        warmup_epochs=0
    )
    
    # Update trainer with new optimizer and scheduler
    trainer.optimizer = optimizer_phase2
    trainer.scheduler = scheduler_phase2
    
    # Train phase 2
    early_stopping_phase2 = None
    if Config.EARLY_STOPPING:
        early_stopping_phase2 = EarlyStopping(
            patience=Config.EARLY_STOPPING_PATIENCE,
            min_delta=Config.EARLY_STOPPING_MIN_DELTA,
            mode='max'
        )
    
    phase2_metrics = trainer.train_phase(
        phase_name="Phase 2",
        num_epochs=Config.PHASE2_EPOCHS,
        early_stopping=early_stopping_phase2,
        start_epoch=Config.PHASE1_EPOCHS + 1
    )
    
    # =========================================================================
    # TRAINING COMPLETE
    # =========================================================================
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    # Save final checkpoint
    trainer.save_final_checkpoint()
    
    # Save metrics
    trainer.metrics_tracker.save()
    
    # Print summary
    summary = trainer.get_summary()
    print(f"\nTraining Summary:")
    print(f"  Total Epochs: {summary['total_epochs']}")
    print(f"  Best Epoch: {summary['best_epoch']}")
    print(f"  Best Val Acc: {summary['best_val_acc']:.2f}%")
    print(f"  Final Train Acc: {summary['final_train_acc']:.2f}%")
    print(f"  Final Val Acc: {summary['final_val_acc']:.2f}%")
    
    print(f"\nPhase 1 Best Val Acc: {phase1_metrics['best_val_acc']:.2f}%")
    print(f"Phase 2 Best Val Acc: {phase2_metrics['best_val_acc']:.2f}%")
    
    print(f"\nCheckpoints saved to: {Config.CHECKPOINTS_DIR}")
    print(f"Logs saved to: {Config.LOGS_DIR}")
    print(f"Best model: {Config.CHECKPOINTS_DIR / 'best_model.pth'}")
    
    print("\n" + "="*80)
    print("Teacher model ready for knowledge distillation!")
    print("="*80)
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TinyDefectNet Teacher Model"
    )
    
    # Data arguments
    parser.add_argument(
        '--data_dir',
        type=str,
        default="/kaggle/input/neu-metal-surface-defects-data/NEU Metal Surface Defects Data",
        help='Path to data directory containing train/val folders'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=6,
        help='Number of classes in dataset'
    )
    
    # Training arguments
    parser.add_argument(
        '--compute_class_weights',
        action='store_true',
        help='Compute class weights for imbalanced datasets'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    args = parser.parse_args()
    
    try:
        summary = main(args)
        print("\n✓ Training completed successfully!")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)