"""
Trainer module for TinyDefectNet Student Model
Implements knowledge distillation training with teacher supervision
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
    calculate_accuracy,
    get_learning_rate,
    distillation_loss,
    verify_deployment_constraints,
    print_constraint_report
)


class DistillationTrainer:
    """
    Trainer for student model with knowledge distillation
    
    Implements:
    - Knowledge distillation from teacher
    - Combined hard + soft loss
    - Constraint monitoring
    - Checkpointing
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: Optional[nn.Module],
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        device: str,
        checkpoint_dir: Path,
        log_dir: Path,
        use_kd: bool = True,
        temperature: float = 4.0,
        alpha: float = 0.7,
        mixed_precision: bool = True
    ):
        """
        Args:
            student_model: Student model to train
            teacher_model: Teacher model for distillation (optional)
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
            use_kd: Whether to use knowledge distillation
            temperature: Temperature for softening distributions
            alpha: Weight for hard loss (1-alpha for soft loss)
            mixed_precision: Use mixed precision training
        """
        self.student = student_model.to(device)
        self.teacher = teacher_model
        if self.teacher is not None:
            self.teacher = self.teacher.to(device)
            self.teacher.eval()  # Teacher always in eval mode
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        
        # KD parameters
        self.use_kd = use_kd and (teacher_model is not None)
        self.temperature = temperature
        self.alpha = alpha
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function (hard loss)
        self.hard_loss_fn = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
        
        # Mixed precision
        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        
        # Tracking
        self.metrics_tracker = MetricsTracker(self.log_dir)
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        print(f"DistillationTrainer initialized:")
        print(f"  Device: {device}")
        print(f"  Knowledge Distillation: {self.use_kd}")
        if self.use_kd:
            print(f"  Temperature (τ): {self.temperature}")
            print(f"  Alpha (α): {self.alpha}")
        print(f"  Mixed precision: {mixed_precision}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
    
    def train_epoch(self, epoch: int) -> Tuple[float, float, Dict[str, float]]:
        """
        Train for one epoch with knowledge distillation
        
        Args:
            epoch: Current epoch number
        
        Returns:
            avg_loss: Average training loss
            avg_acc: Average training accuracy
            loss_components: Dictionary with loss breakdown
        """
        self.student.train()
        if self.teacher is not None:
            self.teacher.eval()
        
        loss_meter = AverageMeter("Loss")
        acc_meter = AverageMeter("Acc")
        hard_loss_meter = AverageMeter("Hard")
        soft_loss_meter = AverageMeter("Soft")
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [Train]",
            disable=not Config.VERBOSE
        )
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass with mixed precision
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    student_logits = self.student(images)
                    
                    if self.use_kd:
                        # Get teacher logits (no grad needed)
                        with torch.no_grad():
                            teacher_logits = self.teacher(images)
                        
                        # Distillation loss
                        loss, loss_dict = distillation_loss(
                            student_logits,
                            teacher_logits,
                            targets,
                            temperature=self.temperature,
                            alpha=self.alpha,
                            hard_loss_fn=self.hard_loss_fn
                        )
                        hard_loss_val = loss_dict['hard_loss']
                        soft_loss_val = loss_dict['soft_loss']
                    else:
                        # Standard cross-entropy
                        loss = self.hard_loss_fn(student_logits, targets)
                        hard_loss_val = loss.item()
                        soft_loss_val = 0.0
            else:
                student_logits = self.student(images)
                
                if self.use_kd:
                    with torch.no_grad():
                        teacher_logits = self.teacher(images)
                    
                    loss, loss_dict = distillation_loss(
                        student_logits,
                        teacher_logits,
                        targets,
                        temperature=self.temperature,
                        alpha=self.alpha,
                        hard_loss_fn=self.hard_loss_fn
                    )
                    hard_loss_val = loss_dict['hard_loss']
                    soft_loss_val = loss_dict['soft_loss']
                else:
                    loss = self.hard_loss_fn(student_logits, targets)
                    hard_loss_val = loss.item()
                    soft_loss_val = 0.0
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                
                if Config.GRADIENT_CLIP_NORM > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(),
                        Config.GRADIENT_CLIP_NORM
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                if Config.GRADIENT_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(),
                        Config.GRADIENT_CLIP_NORM
                    )
                
                self.optimizer.step()
            
            # Calculate accuracy
            acc = calculate_accuracy(student_logits, targets)
            
            # Update meters
            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc, batch_size)
            hard_loss_meter.update(hard_loss_val, batch_size)
            soft_loss_meter.update(soft_loss_val, batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_meter.avg:.4f}",
                'acc': f"{acc_meter.avg:.2f}%",
                'hard': f"{hard_loss_meter.avg:.4f}",
                'soft': f"{soft_loss_meter.avg:.4f}"
            })
        
        loss_components = {
            'hard_loss': hard_loss_meter.avg,
            'soft_loss': soft_loss_meter.avg
        }
        
        return loss_meter.avg, acc_meter.avg, loss_components
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, float]:
        """
        Validate model
        
        Args:
            epoch: Current epoch number
        
        Returns:
            avg_loss: Average validation loss
            avg_acc: Average validation accuracy
        """
        self.student.eval()
        
        loss_meter = AverageMeter("Loss")
        acc_meter = AverageMeter("Acc")
        
        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch} [Val]",
            disable=not Config.VERBOSE
        )
        
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.student(images)
            loss = self.hard_loss_fn(outputs, targets)
            
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
    
    def train(
        self,
        num_epochs: int,
        early_stopping: Optional[EarlyStopping] = None,
        verify_constraints: bool = True
    ) -> Dict[str, float]:
        """
        Train for multiple epochs
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping: Optional early stopping
            verify_constraints: Verify deployment constraints
        
        Returns:
            Best metrics
        """
        print(f"\n{'='*80}")
        print("Starting Knowledge Distillation Training")
        print(f"{'='*80}")
        
        # Verify deployment constraints before training
        if verify_constraints:
            results = verify_deployment_constraints(self.student)
            print_constraint_report(results)
            
            if not results['all_constraints_met']:
                print("\n⚠ WARNING: Model does not meet deployment constraints!")
                print("  Training will continue, but model needs optimization.")
        
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_acc, loss_components = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
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
                lr=current_lr,
                hard_loss=loss_components['hard_loss'],
                soft_loss=loss_components['soft_loss']
            )
            
            # Epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print summary
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            if self.use_kd:
                print(f"  Hard Loss: {loss_components['hard_loss']:.4f} | Soft Loss: {loss_components['soft_loss']:.4f}")
            print(f"  LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")
            print(f"{'='*80}")
            
            # Check if best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                print(f"✓ New best model! Val Acc: {val_acc:.2f}%")
                
                # Save best checkpoint
                self._save_checkpoint(
                    epoch, train_loss, train_acc, val_loss, val_acc,
                    is_best=True
                )
            
            # Save periodic checkpoint
            if epoch % Config.SAVE_INTERVAL == 0:
                self._save_checkpoint(
                    epoch, train_loss, train_acc, val_loss, val_acc,
                    is_best=False
                )
            
            # Early stopping
            if early_stopping is not None:
                if early_stopping(val_acc):
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break
        
        print(f"\n{'='*80}")
        print("Training Complete")
        print(f"{'='*80}")
        print(f"Best Val Acc: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        
        # Final constraint check
        if verify_constraints:
            print(f"\nFinal Deployment Constraint Check:")
            results = verify_deployment_constraints(self.student)
            print_constraint_report(results)
        
        return {
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }
    
    def _save_checkpoint(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        is_best: bool = False
    ):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'config': {
                'num_classes': Config.NUM_CLASSES,
                'temperature': self.temperature,
                'alpha': self.alpha
            }
        }
        
        if is_best:
            save_path = self.checkpoint_dir / 'best_student.pth'
            torch.save(checkpoint, save_path)
            print(f"  Saved to {save_path}")
        else:
            save_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, save_path)


if __name__ == "__main__":
    print("Trainer module - use via train.py")