"""
Teacher model architecture for TinyDefectNet
EfficientNet-B0 with custom classifier and freeze/unfreeze utilities
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional

from config import Config


class TeacherModel(nn.Module):
    """
    Teacher model based on EfficientNet-B0
    
    Architecture:
        - EfficientNet-B0 backbone (pretrained on ImageNet)
        - Custom classification head
        - Dropout for regularization
    
    Training strategy:
        Phase 1: Freeze backbone, train head only
        Phase 2: Unfreeze last blocks, fine-tune end-to-end
    """
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        freeze_backbone: bool = True
    ):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Use ImageNet pretrained weights
            dropout_rate: Dropout probability
            freeze_backbone: Whether to freeze backbone initially
        """
        super(TeacherModel, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Load EfficientNet-B0 pretrained on ImageNet
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        
        # Get feature dimension from backbone
        # EfficientNet-B0: 1280 features before classifier
        in_features = self.backbone.classifier[1].in_features
        
        # Replace classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()
        
        print(f"TeacherModel initialized:")
        print(f"  Backbone: EfficientNet-B0 (pretrained={pretrained})")
        print(f"  Num classes: {num_classes}")
        print(f"  Dropout: {dropout_rate}")
        print(f"  Backbone frozen: {freeze_backbone}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, 3, H, W]
        
        Returns:
            logits: Raw logits [B, num_classes]
        """
        logits = self.backbone(x)
        return logits
    
    def freeze_backbone(self):
        """Freeze all backbone parameters"""
        for name, param in self.backbone.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        
        print("Backbone frozen (classifier trainable)")
        self._print_trainable_params()
    
    def unfreeze_backbone(self, from_block: Optional[int] = None):
        """
        Unfreeze backbone parameters
        
        Args:
            from_block: If specified, unfreeze only from this block onwards.
                       EfficientNet-B0 has blocks 0-6 in features.
                       If None, unfreezes entire backbone.
        """
        if from_block is None:
            # Unfreeze all
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("Entire backbone unfrozen")
        else:
            # Unfreeze from specific block
            # EfficientNet structure: features[0-8] (blocks 0-6 are in features)
            for name, param in self.backbone.named_parameters():
                # Check if in features and which block
                if 'features' in name:
                    # Extract block number from parameter name
                    # E.g., 'features.5.0.weight' -> block 5
                    parts = name.split('.')
                    if len(parts) > 1 and parts[1].isdigit():
                        block_num = int(parts[1])
                        if block_num >= from_block:
                            param.requires_grad = True
                else:
                    # Classifier always trainable
                    param.requires_grad = True
            
            print(f"Backbone unfrozen from block {from_block} onwards")
        
        self._print_trainable_params()
    
    def _print_trainable_params(self):
        """Print number of trainable parameters"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get raw logits (for knowledge distillation)
        
        Args:
            x: Input tensor [B, 3, H, W]
        
        Returns:
            logits: Raw logits [B, num_classes]
        """
        return self.forward(x)
    
    def get_soft_predictions(
        self,
        x: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Get softened predictions for knowledge distillation
        
        Args:
            x: Input tensor [B, 3, H, W]
            temperature: Temperature for softmax (T > 1 softens distribution)
        
        Returns:
            soft_probs: Softened probability distribution [B, num_classes]
        """
        logits = self.forward(x)
        soft_logits = logits / temperature
        soft_probs = torch.softmax(soft_logits, dim=1)
        return soft_probs
    
    def count_parameters(self) -> dict:
        """Count model parameters"""
        return {
            'total': sum(p.numel() for p in self.parameters()),
            'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'frozen': sum(p.numel() for p in self.parameters() if not p.requires_grad)
        }


def create_teacher_model(
    num_classes: int = Config.NUM_CLASSES,
    pretrained: bool = Config.PRETRAINED,
    dropout_rate: float = Config.DROPOUT_RATE,
    freeze_backbone: bool = True
) -> TeacherModel:
    """
    Factory function to create teacher model
    
    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        dropout_rate: Dropout probability
        freeze_backbone: Whether to freeze backbone initially
    
    Returns:
        Teacher model instance
    """
    model = TeacherModel(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone
    )
    
    return model


def load_checkpoint(
    model: TeacherModel,
    checkpoint_path: str,
    device: str = 'cpu'
) -> dict:
    """
    Load model from checkpoint
    
    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
    
    Returns:
        Checkpoint metadata (epoch, metrics, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Acc: {checkpoint.get('val_acc', 'N/A'):.4f}")
    
    return checkpoint


if __name__ == "__main__":
    # Test model creation
    print("Testing TeacherModel...")
    
    # Phase 1: Frozen backbone
    print("\n" + "="*80)
    print("PHASE 1: Frozen backbone")
    print("="*80)
    model_phase1 = create_teacher_model(
        num_classes=6,
        pretrained=True,
        freeze_backbone=True
    )
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    logits = model_phase1(dummy_input)
    print(f"\nOutput shape: {logits.shape}")
    
    param_counts = model_phase1.count_parameters()
    print(f"\nParameter counts:")
    for key, value in param_counts.items():
        print(f"  {key}: {value:,}")
    
    # Phase 2: Unfreeze last blocks
    print("\n" + "="*80)
    print("PHASE 2: Unfrozen from block 5")
    print("="*80)
    model_phase1.unfreeze_backbone(from_block=5)
    
    param_counts = model_phase1.count_parameters()
    print(f"\nParameter counts:")
    for key, value in param_counts.items():
        print(f"  {key}: {value:,}")
    
    # Test soft predictions
    print("\n" + "="*80)
    print("Testing soft predictions (for KD)")
    print("="*80)
    
    soft_probs_T1 = model_phase1.get_soft_predictions(dummy_input, temperature=1.0)
    soft_probs_T4 = model_phase1.get_soft_predictions(dummy_input, temperature=4.0)
    
    print(f"Soft predictions (T=1.0): {soft_probs_T1[0]}")
    print(f"Soft predictions (T=4.0): {soft_probs_T4[0]}")
    print(f"T=4.0 is smoother (lower max prob): {soft_probs_T4[0].max():.4f} < {soft_probs_T1[0].max():.4f}")