"""
TinyNet Student Model for TinyDefectNet
Lightweight CNN optimized for deployment: <5MB, <200k params, <50ms CPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from config import Config


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution
    Significantly reduces parameters and computation compared to standard conv
    
    Standard conv params: C_in * C_out * k * k
    Depthwise separable: C_in * k * k + C_in * C_out
    Reduction: ~8-9x for 3x3 kernels
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False
    ):
        super().__init__()
        
        # Depthwise: apply single filter per input channel
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # Key: groups = in_channels
            bias=bias
        )
        
        # Pointwise: 1x1 conv to combine channels
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ConvBlock(nn.Module):
    """
    Basic convolutional block with normalization and activation
    Uses depthwise separable convs for efficiency
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_depthwise: bool = True,
        normalization: str = "group_norm",
        activation: str = "relu",
        dropout_rate: float = 0.0
    ):
        super().__init__()
        
        # Convolution
        if use_depthwise:
            self.conv = DepthwiseSeparableConv(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            )
        
        # Normalization
        if normalization == "batch_norm":
            self.norm = nn.BatchNorm2d(out_channels)
        elif normalization == "group_norm":
            # Group norm with 8 groups (better for small batches)
            num_groups = min(8, out_channels)
            self.norm = nn.GroupNorm(num_groups, out_channels)
        else:
            self.norm = nn.Identity()
        
        # Activation
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "relu6":
            self.act = nn.ReLU6(inplace=True)
        elif activation == "hard_swish":
            self.act = nn.Hardswish(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)
        
        # Dropout
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x
from huggingface_hub import PyTorchModelHubMixin

class TinyNet(nn.Module, PyTorchModelHubMixin):
    """
    TinyNet: Lightweight CNN for industrial defect detection
    
    Design principles:
    - Fully convolutional (no large dense layers)
    - Depthwise separable convolutions
    - Group normalization (better for deployment)
    - Minimal parameters (<200k)
    - Fast inference (<50ms on CPU)
    
    Architecture:
        Input (224x224x3)
        ↓
        Stem (initial conv + pool)
        ↓
        Stage 1: 112x112 → 56x56
        Stage 2: 56x56 → 28x28
        Stage 3: 28x28 → 14x14
        Stage 4: 14x14 → 7x7
        ↓
        Global Average Pooling
        ↓
        Classifier (1x1 conv)
        ↓
        Output logits
    """
    
    def __init__(
        self,
        num_classes: int = Config.NUM_CLASSES,
        initial_channels: int = Config.INITIAL_CHANNELS,
        channel_multipliers: List[int] = Config.CHANNEL_MULTIPLIERS,
        num_blocks_per_stage: List[int] = Config.NUM_BLOCKS_PER_STAGE,
        use_depthwise: bool = Config.USE_DEPTHWISE_SEPARABLE,
        normalization: str = Config.NORMALIZATION,
        activation: str = Config.ACTIVATION,
        dropout_rate: float = Config.DROPOUT_RATE
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Stem: initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, initial_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, initial_channels), initial_channels),
            nn.ReLU(inplace=True)
        )
        # After stem: 224 → 112
        
        # Build stages
        self.stages = nn.ModuleList()
        in_channels = initial_channels
        
        for stage_idx, (mult, num_blocks) in enumerate(zip(channel_multipliers, num_blocks_per_stage)):
            out_channels = initial_channels * mult
            
            # First block in stage: stride=2 for downsampling
            stage = nn.ModuleList()
            stage.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    stride=2,
                    use_depthwise=use_depthwise,
                    normalization=normalization,
                    activation=activation,
                    dropout_rate=0.0  # No dropout in downsampling
                )
            )
            
            # Remaining blocks: stride=1
            for _ in range(num_blocks - 1):
                stage.append(
                    ConvBlock(
                        out_channels,
                        out_channels,
                        stride=1,
                        use_depthwise=use_depthwise,
                        normalization=normalization,
                        activation=activation,
                        dropout_rate=dropout_rate
                    )
                )
            
            self.stages.append(nn.Sequential(*stage))
            in_channels = out_channels
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier: 1x1 conv instead of FC (fully convolutional)
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
        
        # Print model info
        print(f"TinyNet initialized:")
        print(f"  Classes: {num_classes}")
        print(f"  Initial channels: {initial_channels}")
        print(f"  Stages: {len(channel_multipliers)}")
        print(f"  Depthwise separable: {use_depthwise}")
        self._print_model_size()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, 3, H, W]
        
        Returns:
            logits: Raw logits [B, num_classes]
        """
        # Stem
        x = self.stem(x)  # [B, C, H/2, W/2]
        
        # Stages
        for stage in self.stages:
            x = stage(x)  # Progressive downsampling
        
        # Global pooling
        x = self.gap(x)  # [B, C, 1, 1]
        
        # Classifier
        x = self.classifier(x)  # [B, num_classes, 1, 1]
        
        # Flatten
        x = x.view(x.size(0), -1)  # [B, num_classes]
        
        return x
    
    def get_soft_predictions(
        self,
        x: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Get softened predictions (for knowledge distillation)
        
        Args:
            x: Input tensor [B, 3, H, W]
            temperature: Temperature for softmax
        
        Returns:
            soft_probs: Softened probabilities [B, num_classes]
        """
        logits = self.forward(x)
        soft_logits = logits / temperature
        soft_probs = F.softmax(soft_logits, dim=1)
        return soft_probs
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _print_model_size(self):
        """Print model size information"""
        from utils import count_parameters, estimate_model_size
        
        params = count_parameters(self)
        size_mb = estimate_model_size(self)
        
        print(f"  Parameters: {params:,}")
        print(f"  Model size: {size_mb:.2f} MB")
        
        # Check constraints
        if params > Config.MAX_PARAMS:
            print(f"  ⚠ WARNING: Exceeds parameter limit ({Config.MAX_PARAMS:,})")
        else:
            print(f"  ✓ Within parameter limit ({Config.MAX_PARAMS:,})")
        
        if size_mb > Config.MAX_MODEL_SIZE_MB:
            print(f"  ⚠ WARNING: Exceeds size limit ({Config.MAX_MODEL_SIZE_MB} MB)")
        else:
            print(f"  ✓ Within size limit ({Config.MAX_MODEL_SIZE_MB} MB)")
    
    def count_parameters(self) -> dict:
        """Count model parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }


def create_student_model(
    num_classes: int = Config.NUM_CLASSES,
    **kwargs
) -> TinyNet:
    """
    Factory function to create student model
    
    Args:
        num_classes: Number of output classes
        **kwargs: Additional model arguments
    
    Returns:
        TinyNet model instance
    """
    model = TinyNet(num_classes=num_classes, **kwargs)
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing TinyNet student model...")
    print("=" * 80)
    
    # Create model
    model = create_student_model(num_classes=6)
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    logits = model(dummy_input)
    
    print(f"\nForward pass test:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {logits.shape}")
    
    # Test soft predictions
    soft_probs = model.get_soft_predictions(dummy_input, temperature=4.0)
    print(f"  Soft probs shape: {soft_probs.shape}")
    print(f"  Soft probs sum: {soft_probs.sum(dim=1)[0]:.4f}")
    
    # Parameter breakdown
    print(f"\nParameter breakdown:")
    total_params = 0
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        total_params += params
        print(f"  {name:15s}: {params:>8,} params")
    print(f"  {'TOTAL':15s}: {total_params:>8,} params")
    
    # Test depthwise separable efficiency
    print(f"\n" + "=" * 80)
    print("Depthwise Separable Convolution Efficiency:")
    print("=" * 80)
    
    # Standard conv
    standard_conv = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    standard_params = sum(p.numel() for p in standard_conv.parameters())
    
    # Depthwise separable
    ds_conv = DepthwiseSeparableConv(32, 64)
    ds_params = sum(p.numel() for p in ds_conv.parameters())
    
    print(f"Standard 3x3 conv (32→64): {standard_params:,} params")
    print(f"Depthwise separable (32→64): {ds_params:,} params")
    print(f"Reduction: {standard_params / ds_params:.2f}x")
    
    print("\n✓ TinyNet tests passed!")