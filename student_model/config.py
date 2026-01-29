"""
Configuration module for TinyDefectNet Student Model

Contains all hyperparameters, paths, and deployment constraints for training
a lightweight defect detection model with knowledge distillation.

Usage:
    from config import Config
    Config.display()
"""

from pathlib import Path
import torch


class Config:
    """
    Central configuration for TinyDefectNet training
    
    Sections:
    - Model Architecture
    - Data Processing
    - Training Hyperparameters
    - Knowledge Distillation
    - Deployment Constraints
    - Paths and Directories
    - Logging and Checkpointing
    """
    
    # =========================================================================
    # MODEL ARCHITECTURE
    # =========================================================================
    
    # Number of output classes
    NUM_CLASSES = 6  # NEU dataset has 6 defect types
    
    # Student model architecture type
    # Options: 'efficientnet_lite', 'mobilenet_v3', 'shufflenet', 'custom', 'tinynet'
    STUDENT_ARCHITECTURE = 'mobilenet_v3'
    
    # Model variant (for standard architectures)
    # MobileNetV3: 'small', 'large'
    # EfficientNet: 'lite0', 'lite1', 'lite2'
    MODEL_VARIANT = 'small'
    
    # Input image size (height, width)
    IMAGE_SIZE = (224, 224)
    
    # Dropout rate for regularization
    DROPOUT_RATE = 0.2
    
    # Use pretrained weights (ImageNet initialization)
    USE_PRETRAINED = True
    
    # -------------------------------------------------------------------------
    # Custom TinyNet Architecture Parameters
    # -------------------------------------------------------------------------
    
    # Initial number of channels (for custom TinyNet)
    INITIAL_CHANNELS = 24
    
    # Channel expansion stages (multipliers for each stage)
    CHANNEL_STAGES = [1, 2, 4, 8]  # Results in: 24, 48, 96, 192
    
    # Channel multipliers (alternative parameterization)
    CHANNEL_MULTIPLIERS = [1, 2, 4, 8]  # Same as CHANNEL_STAGES
    
    # Number of blocks per stage
    NUM_BLOCKS_PER_STAGE = [2, 3, 4, 3]
    
    # Use squeeze-and-excitation blocks
    USE_SE = True
    
    # SE reduction ratio
    SE_REDUCTION = 4
    
    # Activation function for custom models
    # Options: 'relu', 'relu6', 'swish', 'hardswish', 'gelu'
    ACTIVATION = 'relu6'
    
    # Use batch normalization
    USE_BATCH_NORM = True
    
    # Batch norm momentum
    BN_MOMENTUM = 0.1
    
    # Batch norm epsilon
    BN_EPS = 1e-5
    
    # Width multiplier (for scaling model width)
    WIDTH_MULTIPLIER = 1.0
    
    # Depth multiplier (for scaling model depth)
    DEPTH_MULTIPLIER = 1.0
    
    # =========================================================================
    # DATA PROCESSING
    # =========================================================================
    
    # Data paths
    DATA_ROOT = Path("/kaggle/input/neu-metal-surface-defects-data/NEU Metal Surface Defects Data")
    TRAIN_DIR = DATA_ROOT / "train"
    VAL_DIR = DATA_ROOT / "val"
    
    # Batch size
    BATCH_SIZE = 32
    
    # Validation split (if no separate val folder)
    VAL_SPLIT = 0.2
    
    # Number of workers for data loading
    NUM_WORKERS = 4
    
    # Pin memory for faster GPU transfer
    PIN_MEMORY = True
    
    # Data augmentation settings
    TRAIN_AUGMENTATION = {
        'random_resized_crop': True,
        'horizontal_flip': True,
        'vertical_flip': False,
        'rotation_degrees': 15,
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.1,
            'hue': 0.05
        },
        'random_erasing': {
            'probability': 0.1,
            'scale': (0.02, 0.15),
            'ratio': (0.3, 3.3)
        }
    }
    
    # Normalization (ImageNet stats)
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # =========================================================================
    # TRAINING HYPERPARAMETERS
    # =========================================================================
    
    # Number of training epochs
    EPOCHS = 100
    
    # Optimizer settings
    OPTIMIZER = 'adamw'  # Options: 'adamw', 'adam', 'sgd'
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    BETAS = (0.9, 0.999)
    EPS = 1e-8
    
    # Learning rate scheduler
    SCHEDULER = 'cosine'  # Options: 'cosine', 'step', None
    LR_MIN = 1e-6  # Minimum learning rate for cosine annealing
    
    # Gradient clipping
    GRADIENT_CLIP_NORM = 1.0  # Set to 0 to disable
    
    # Label smoothing
    LABEL_SMOOTHING = 0.1
    
    # Mixed precision training
    MIXED_PRECISION = True
    
    # Random seed for reproducibility
    SEED = 42
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # =========================================================================
    # KNOWLEDGE DISTILLATION
    # =========================================================================
    
    # Enable knowledge distillation
    USE_KD = True
    
    # Teacher model checkpoint path
    TEACHER_CHECKPOINT = Path("/kaggle/working/optimized-industrual-defect-detection-model/teacher_model/checkpoints/teacher_final.pth")
    
    # Distillation temperature (τ)
    # Higher temperature = softer probability distributions
    # Typical range: 1-10, recommended: 3-5
    KD_TEMPERATURE = 4.0
    
    # Distillation loss weight (α)
    # alpha * hard_loss + (1-alpha) * soft_loss
    # Higher alpha = more weight on hard labels
    # Typical range: 0.3-0.9, recommended: 0.5-0.7
    KD_ALPHA = 0.7
    
    # =========================================================================
    # DEPLOYMENT CONSTRAINTS
    # =========================================================================
    
    # Maximum model parameters (for edge deployment)
    MAX_PARAMS = 2_000_000  # 2M parameters
    
    # Maximum model size (int8 quantized)
    MAX_MODEL_SIZE_MB = 5.0  # 5 MB
    
    # Maximum inference latency (CPU, single image)
    MAX_LATENCY_MS = 50.0  # 50 milliseconds
    
    # Target hardware platform
    TARGET_PLATFORM = 'cpu'  # Options: 'cpu', 'edge_tpu', 'coral'
    
    # Verify constraints during training
    VERIFY_CONSTRAINTS = True
    
    # Constraint verification frequency (epochs)
    CONSTRAINT_CHECK_INTERVAL = 10
    
    # =========================================================================
    # EARLY STOPPING
    # =========================================================================
    
    # Enable early stopping
    EARLY_STOPPING = True
    
    # Patience (epochs without improvement)
    EARLY_STOPPING_PATIENCE = 15
    
    # Minimum improvement to count as better
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # =========================================================================
    # PATHS AND DIRECTORIES
    # =========================================================================
    
    # Working directory
    WORK_DIR = Path("/kaggle/working")
    
    # Project directory
    PROJECT_DIR = WORK_DIR / "tinydefectnet_student"
    
    # Checkpoints directory
    CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"
    
    # Logs directory
    LOGS_DIR = PROJECT_DIR / "logs"
    
    # Results directory
    RESULTS_DIR = PROJECT_DIR / "results"
    
    # =========================================================================
    # LOGGING AND CHECKPOINTING
    # =========================================================================
    
    # Save checkpoint every N epochs
    SAVE_INTERVAL = 10
    
    # Keep only best N checkpoints
    KEEP_TOP_N_CHECKPOINTS = 3
    
    # Verbose logging
    VERBOSE = True
    
    # Log metrics to file
    LOG_METRICS = True
    
    # Metrics log file
    METRICS_LOG_FILE = LOGS_DIR / "metrics.json"
    
    # =========================================================================
    # EVALUATION AND TESTING
    # =========================================================================
    
    # Benchmark CPU latency
    BENCHMARK_CPU = True
    
    # Number of benchmark runs for latency measurement
    NUM_BENCHMARK_RUNS = 100
    
    # Test robustness to noise and blur
    TEST_ROBUSTNESS = True
    
    # Noise levels for robustness testing (Gaussian noise std)
    NOISE_LEVELS = [0.01, 0.03, 0.05, 0.1]
    
    # Blur kernel sizes for robustness testing
    BLUR_KERNELS = [3, 5, 7]
    
    # =========================================================================
    # EXPORT SETTINGS
    # =========================================================================
    
    # Export to ONNX format
    EXPORT_ONNX = True
    
    # ONNX opset version
    ONNX_OPSET_VERSION = 11
    
    # Export to TorchScript
    EXPORT_TORCHSCRIPT = False
    
    # Export to TFLite
    EXPORT_TFLITE = False
    
    # =========================================================================
    # CLASS METHODS
    # =========================================================================
    
    @classmethod
    def display(cls):
        """Display current configuration"""
        print("\n" + "="*80)
        print("TinyDefectNet Configuration")
        print("="*80)
        
        print("\n[MODEL ARCHITECTURE]")
        print(f"  Student Architecture: {cls.STUDENT_ARCHITECTURE}")
        print(f"  Model Variant: {cls.MODEL_VARIANT}")
        print(f"  Number of Classes: {cls.NUM_CLASSES}")
        print(f"  Image Size: {cls.IMAGE_SIZE}")
        print(f"  Dropout Rate: {cls.DROPOUT_RATE}")
        print(f"  Use Pretrained: {cls.USE_PRETRAINED}")
        
        print("\n[DATA PROCESSING]")
        print(f"  Data Root: {cls.DATA_ROOT}")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Val Split: {cls.VAL_SPLIT}")
        print(f"  Num Workers: {cls.NUM_WORKERS}")
        
        print("\n[TRAINING]")
        print(f"  Epochs: {cls.EPOCHS}")
        print(f"  Optimizer: {cls.OPTIMIZER}")
        print(f"  Learning Rate: {cls.LEARNING_RATE}")
        print(f"  Weight Decay: {cls.WEIGHT_DECAY}")
        print(f"  Scheduler: {cls.SCHEDULER}")
        print(f"  Mixed Precision: {cls.MIXED_PRECISION}")
        print(f"  Label Smoothing: {cls.LABEL_SMOOTHING}")
        print(f"  Gradient Clip: {cls.GRADIENT_CLIP_NORM}")
        print(f"  Device: {cls.DEVICE}")
        
        print("\n[KNOWLEDGE DISTILLATION]")
        print(f"  Use KD: {cls.USE_KD}")
        if cls.USE_KD:
            print(f"  Teacher Checkpoint: {cls.TEACHER_CHECKPOINT}")
            print(f"  Temperature (τ): {cls.KD_TEMPERATURE}")
            print(f"  Alpha (α): {cls.KD_ALPHA}")
        
        print("\n[DEPLOYMENT CONSTRAINTS]")
        print(f"  Max Parameters: {cls.MAX_PARAMS:,}")
        print(f"  Max Model Size: {cls.MAX_MODEL_SIZE_MB} MB")
        print(f"  Max Latency: {cls.MAX_LATENCY_MS} ms")
        print(f"  Target Platform: {cls.TARGET_PLATFORM}")
        print(f"  Verify Constraints: {cls.VERIFY_CONSTRAINTS}")
        
        print("\n[EARLY STOPPING]")
        print(f"  Enabled: {cls.EARLY_STOPPING}")
        if cls.EARLY_STOPPING:
            print(f"  Patience: {cls.EARLY_STOPPING_PATIENCE}")
            print(f"  Min Delta: {cls.EARLY_STOPPING_MIN_DELTA}")
        
        print("\n[PATHS]")
        print(f"  Project Dir: {cls.PROJECT_DIR}")
        print(f"  Checkpoints: {cls.CHECKPOINTS_DIR}")
        print(f"  Logs: {cls.LOGS_DIR}")
        
        print("\n[EVALUATION]")
        print(f"  Benchmark CPU: {cls.BENCHMARK_CPU}")
        print(f"  Test Robustness: {cls.TEST_ROBUSTNESS}")
        
        print("\n[EXPORT]")
        print(f"  Export ONNX: {cls.EXPORT_ONNX}")
        print(f"  Export TorchScript: {cls.EXPORT_TORCHSCRIPT}")
        print(f"  Export TFLite: {cls.EXPORT_TFLITE}")
        
        print("\n" + "="*80 + "\n")
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        dirs = [
            cls.PROJECT_DIR,
            cls.CHECKPOINTS_DIR,
            cls.LOGS_DIR,
            cls.RESULTS_DIR
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Created directories:")
        for dir_path in dirs:
            print(f"  {dir_path}")
    
    @classmethod
    def update(cls, **kwargs):
        """
        Update configuration parameters
        
        Args:
            **kwargs: Key-value pairs to update
        
        Example:
            Config.update(EPOCHS=150, LEARNING_RATE=1e-4)
        """
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
                print(f"✓ Updated {key} = {value}")
            else:
                print(f"⚠ Warning: {key} is not a valid config parameter")
    
    @classmethod
    def save(cls, filepath: Path):
        """
        Save configuration to JSON file
        
        Args:
            filepath: Path to save configuration
        """
        import json
        
        config_dict = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        print(f"✓ Configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path):
        """
        Load configuration from JSON file
        
        Args:
            filepath: Path to configuration file
        """
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        for key, value in config_dict.items():
            if hasattr(cls, key):
                # Convert string paths back to Path objects
                if 'DIR' in key or 'PATH' in key or key == 'DATA_ROOT':
                    value = Path(value)
                setattr(cls, key, value)
        
        print(f"✓ Configuration loaded from {filepath}")
    
    @classmethod
    def get_constraint_dict(cls) -> dict:
        """
        Get deployment constraints as dictionary
        
        Returns:
            Dictionary of constraints
        """
        return {
            'max_params': cls.MAX_PARAMS,
            'max_size_mb': cls.MAX_MODEL_SIZE_MB,
            'max_latency_ms': cls.MAX_LATENCY_MS,
            'target_platform': cls.TARGET_PLATFORM
        }
    
    @classmethod
    def get_kd_config(cls) -> dict:
        """
        Get knowledge distillation configuration
        
        Returns:
            Dictionary of KD settings
        """
        return {
            'use_kd': cls.USE_KD,
            'teacher_checkpoint': str(cls.TEACHER_CHECKPOINT),
            'temperature': cls.KD_TEMPERATURE,
            'alpha': cls.KD_ALPHA
        }
    
    @classmethod
    def get_training_config(cls) -> dict:
        """
        Get training configuration
        
        Returns:
            Dictionary of training settings
        """
        return {
            'epochs': cls.EPOCHS,
            'batch_size': cls.BATCH_SIZE,
            'optimizer': cls.OPTIMIZER,
            'learning_rate': cls.LEARNING_RATE,
            'weight_decay': cls.WEIGHT_DECAY,
            'scheduler': cls.SCHEDULER,
            'mixed_precision': cls.MIXED_PRECISION,
            'label_smoothing': cls.LABEL_SMOOTHING,
            'gradient_clip_norm': cls.GRADIENT_CLIP_NORM
        }
    
    @classmethod
    def get_model_config(cls) -> dict:
        """
        Get model architecture configuration
        
        Returns:
            Dictionary of model settings
        """
        return {
            'architecture': cls.STUDENT_ARCHITECTURE,
            'variant': cls.MODEL_VARIANT,
            'num_classes': cls.NUM_CLASSES,
            'image_size': cls.IMAGE_SIZE,
            'dropout_rate': cls.DROPOUT_RATE,
            'use_pretrained': cls.USE_PRETRAINED,
            'initial_channels': cls.INITIAL_CHANNELS,
            'channel_stages': cls.CHANNEL_STAGES,
            'channel_multipliers': cls.CHANNEL_MULTIPLIERS,
            'num_blocks_per_stage': cls.NUM_BLOCKS_PER_STAGE,
            'use_se': cls.USE_SE,
            'se_reduction': cls.SE_REDUCTION,
            'activation': cls.ACTIVATION,
            'width_multiplier': cls.WIDTH_MULTIPLIER,
            'depth_multiplier': cls.DEPTH_MULTIPLIER
        }


# Presets for different use cases
class QuickTestConfig(Config):
    """Configuration for quick testing"""
    EPOCHS = 5
    BATCH_SIZE = 16
    VERIFY_CONSTRAINTS = False
    BENCHMARK_CPU = False
    TEST_ROBUSTNESS = False
    EXPORT_ONNX = False
    SAVE_INTERVAL = 1


class HighAccuracyConfig(Config):
    """Configuration optimized for high accuracy"""
    EPOCHS = 200
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 5e-5
    LABEL_SMOOTHING = 0.15
    EARLY_STOPPING_PATIENCE = 25
    KD_ALPHA = 0.5  # More weight on soft targets


class FastInferenceConfig(Config):
    """Configuration optimized for fast inference"""
    MAX_PARAMS = 1_000_000  # 1M parameters
    MAX_LATENCY_MS = 30.0  # 30ms
    MODEL_VARIANT = 'small'
    DROPOUT_RATE = 0.1


class EdgeDeviceConfig(Config):
    """Configuration for edge device deployment"""
    MAX_PARAMS = 500_000  # 500K parameters
    MAX_MODEL_SIZE_MB = 2.0  # 2 MB
    MAX_LATENCY_MS = 20.0  # 20ms
    TARGET_PLATFORM = 'edge_tpu'
    BATCH_SIZE = 16
    IMAGE_SIZE = (192, 192)  # Smaller images for edge


if __name__ == "__main__":
    # Display default configuration
    Config.display()
    
    # Example: Create directories
    Config.create_dirs()
    
    # Example: Save configuration
    Config.save(Config.LOGS_DIR / "config.json")
    
    print("\n✓ Config module test complete")