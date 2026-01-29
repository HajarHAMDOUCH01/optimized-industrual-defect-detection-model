"""
Trainer module for TinyDefectNet Teacher Model
Handles training loop, validation, checkpointing, and logging
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from pathlib import Path
from typing import Optional, Dict, Tuple
import time
from tqdm import tqdm

from config import Config
from utils import (
    AverageMeter,
    MetricsTracker,
    EarlyStopping,
    save_checkpoint,
    calculate_accuracy,
    GradScalerWrapper,
    get_learning_rate,
    print_training_header,
    print_epoch_summary
)


class Trainer:
    """
    Trainer for teacher model with two-phase training strategy
    
    Phase 1: Train classification head with frozen backbone
    Phase 2: Fine-tune with unfrozen last blocks
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        device: str,
        checkpoint_dir: Path,
        log_dir: Path,
        mixed_precision: bool = True
    ):
        """
        Args:
            model: Teacher model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
            mixed_precision: Use mixed precision training
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.mixed_precision = mixed_precision
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking
        self.metrics_tracker = MetricsTracker(self.log_dir)
        self.grad_scaler = GradScalerWrapper(enabled=mixed_precision)
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Mixed precision: {mixed_precision}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
    
    def train_epoch(self, epoch: int, phase: str = "Phase 1") -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            phase: Training phase name
        
        Returns:
            avg_loss: Average training loss
            avg_acc: Average training accuracy
        """
        self.model.train()
        
        loss_meter = AverageMeter("Loss")
        acc_meter = AverageMeter("Acc")
        
        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"{phase} Epoch {epoch} [Train]",
            disable=not Config.VERBOSE
        )
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass with mixed precision
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                self.grad_scaler.scale(loss).backward()
                
                # Gradient clipping
                if Config.GRADIENT_CLIP_NORM > 0:
                    self.grad_scaler.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        Config.GRADIENT_CLIP_NORM
                    )
                
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if Config.GRADIENT_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        Config.GRADIENT_CLIP_NORM
                    )
                
                self.optimizer.step()
            
            # Calculate accuracy
            acc = calculate_accuracy(outputs, targets)
            
            # Update meters
            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc, batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_meter.avg:.4f}",
                'acc': f"{acc_meter.avg:.2f}%"
            })
        
        return loss_meter.avg, acc_meter.avg
    
    @torch.no_grad()
    def validate(self, epoch: int, phase: str = "Phase 1") -> Tuple[float, float]:
        """
        Validate model
        
        Args:
            epoch: Current epoch number
            phase: Training phase name
        
        Returns:
            avg_loss: Average validation loss
            avg_acc: Average validation accuracy
        """
        self.model.eval()
        
        loss_meter = AverageMeter("Loss")
        acc_meter = AverageMeter("Acc")
        
        # Progress bar
        pbar = tqdm(
            self.val_loader,
            desc=f"{phase} Epoch {epoch} [Val]",
            disable=not Config.VERBOSE
        )
        
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Calculate accuracy
            acc = calculate_accuracy(outputs, targets)
            
            # Update meters
            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc, batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_meter.avg:.4f}",
                'acc': f"{acc_meter.avg:.2f}%"
            })
        
        return loss_meter.avg, acc_meter.avg
    
    def train_phase(
        self,
        phase_name: str,
        num_epochs: int,
        early_stopping: Optional[EarlyStopping] = None,
        start_epoch: int = 1
    ) -> Dict[str, float]:
        """
        Train for multiple epochs (one phase)
        
        Args:
            phase_name: Name of training phase
            num_epochs: Number of epochs to train
            early_stopping: Optional early stopping
            start_epoch: Starting epoch number
        
        Returns:
            Best metrics from this phase
        """
        print(f"\n{'='*80}")
        print(f"Starting {phase_name}")
        print(f"{'='*80}")
        
        phase_best_acc = 0.0
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            epoch_start_time = time.time()
            
            print_training_header(phase_name, epoch, start_epoch + num_epochs - 1)
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch, phase_name)
            
            # Validate
            val_loss, val_acc = self.validate(epoch, phase_name)
            
            # Learning rate
            current_lr = get_learning_rate(self.optimizer)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Track metrics
            self.metrics_tracker.update(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=current_lr
            )
            
            # Epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print summary
            print_epoch_summary(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=current_lr,
                epoch_time=epoch_time
            )
            
            # Check if best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                print(f"✓ New best model! Val Acc: {val_acc:.2f}%")
            
            # Update phase best
            phase_best_acc = max(phase_best_acc, val_acc)
            
            # Save checkpoint
            if Config.SAVE_INTERVAL > 0 and epoch % Config.SAVE_INTERVAL == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    save_path=checkpoint_path,
                    is_best=is_best
                )
            
            # Early stopping check
            if early_stopping is not None:
                if early_stopping(val_acc):
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break
        
        print(f"\n{phase_name} completed. Best Val Acc: {phase_best_acc:.2f}%")
        
        return {
            'best_val_acc': phase_best_acc,
            'final_train_loss': train_loss,
            'final_train_acc': train_acc,
            'final_val_loss': val_loss,
            'final_val_acc': val_acc
        }
    
    def save_final_checkpoint(self, filename: str = "teacher_final.pth"):
        """Save final model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        
        # Get latest metrics
        if len(self.metrics_tracker.history['val_acc']) > 0:
            latest_idx = -1
            train_loss = self.metrics_tracker.history['train_loss'][latest_idx]
            train_acc = self.metrics_tracker.history['train_acc'][latest_idx]
            val_loss = self.metrics_tracker.history['val_loss'][latest_idx]
            val_acc = self.metrics_tracker.history['val_acc'][latest_idx]
            epoch = len(self.metrics_tracker.history['val_acc'])
        else:
            train_loss = train_acc = val_loss = val_acc = 0.0
            epoch = 0
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            save_path=checkpoint_path,
            is_best=False
        )
        
        print(f"\n✓ Final checkpoint saved to {checkpoint_path}")
    
    def get_summary(self) -> Dict[str, any]:
        """Get training summary"""
        return {
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'total_epochs': len(self.metrics_tracker.history['val_acc']),
            'final_train_acc': self.metrics_tracker.history['train_acc'][-1] if self.metrics_tracker.history['train_acc'] else 0,
            'final_val_acc': self.metrics_tracker.history['val_acc'][-1] if self.metrics_tracker.history['val_acc'] else 0
        }


if __name__ == "__main__":
    print("Trainer module - use via train.py")