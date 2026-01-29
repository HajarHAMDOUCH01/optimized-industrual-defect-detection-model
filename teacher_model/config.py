"""
Configuration for TinyDefectNet Teacher Model Training
Optimized for knowledge distillation and industrial defect detection
"""

import torch
from pathlib import Path


class Config:
    """Teacher model training configuration"""
    
    # ============================================================================
    # PATHS
    # ============================================================================
    PROJECT_ROOT = Path(__file__).parent
    DATA_ROOT = PROJECT_ROOT / "data"
    CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    TRAIN_DIR = DATA_ROOT / "train"
    VAL_DIR = DATA_ROOT / "val"
    
    # ============================================================================
    # DEVICE & RUNTIME
    # ============================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    PIN_MEMORY = True if DEVICE == "cuda" else False
    
    # ============================================================================
    # MODEL ARCHITECTURE
    # ============================================================================
    BACKBONE = "efficientnet_b0"  # EfficientNet-B0 pretrained on ImageNet
    NUM_CLASSES = 6  
    PRETRAINED = True
    DROPOUT_RATE = 0.2  # Light dropout for stability
    
    # ============================================================================
    # DATA AUGMENTATION
    # ============================================================================
    # Input size for EfficientNet-B0
    IMAGE_SIZE = 224
    
    # Augmentation parameters (conservative for industrial defects)
    # Avoid semantic distortion - preserve texture integrity
    AUG_ROTATION = 15  # degrees
    AUG_BRIGHTNESS = 0.1
    AUG_CONTRAST = 0.1
    AUG_SATURATION = 0.05
    AUG_HUE = 0.02
    AUG_HORIZONTAL_FLIP = True
    AUG_VERTICAL_FLIP = False  # Usually False for industrial surfaces
    
    # Normalization (ImageNet stats for pretrained backbone)
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # ============================================================================
    # TRAINING HYPERPARAMETERS - PHASE 1: HEAD ONLY
    # ============================================================================
    # Phase 1: Train only the classification head with frozen backbone
    PHASE1_EPOCHS = 10
    PHASE1_BATCH_SIZE = 32
    PHASE1_LR = 1e-3
    PHASE1_WEIGHT_DECAY = 1e-4
    PHASE1_FREEZE_BACKBONE = True
    
    # ============================================================================
    # TRAINING HYPERPARAMETERS - PHASE 2: FINE-TUNING
    # ============================================================================
    # Phase 2: Unfreeze last blocks and fine-tune end-to-end
    PHASE2_EPOCHS = 20
    PHASE2_BATCH_SIZE = 32
    PHASE2_LR = 1e-4  # Lower LR for fine-tuning
    PHASE2_WEIGHT_DECAY = 1e-4
    PHASE2_UNFREEZE_FROM_BLOCK = 5  # Unfreeze from block 5 onwards (EfficientNet has 7 blocks)
    
    # ============================================================================
    # OPTIMIZER & SCHEDULER
    # ============================================================================
    OPTIMIZER = "adamw"  # AdamW for better generalization
    BETAS = (0.9, 0.999)
    EPS = 1e-8
    
    # Learning rate scheduler
    SCHEDULER = "cosine"  # cosine annealing
    LR_MIN = 1e-6  # Minimum LR for cosine annealing
    WARMUP_EPOCHS = 2  # Warmup for phase 1
    
    # ============================================================================
    # LOSS & REGULARIZATION
    # ============================================================================
    # CrossEntropy for teacher (NO label smoothing - we need clean logits for distillation)
    LOSS_FN = "cross_entropy"
    LABEL_SMOOTHING = 0.0  # CRITICAL: No smoothing for KD teacher
    
    # Class weights for imbalanced datasets (set to None for balanced)
    CLASS_WEIGHTS = None  # Will be computed from dataset if needed
    
    # ============================================================================
    # TRAINING STABILITY
    # ============================================================================
    GRADIENT_CLIP_NORM = 1.0  # Gradient clipping for stability
    MIXED_PRECISION = True if DEVICE == "cuda" else False  # AMP for faster training
    
    # ============================================================================
    # CHECKPOINTING & EARLY STOPPING
    # ============================================================================
    SAVE_BEST_ONLY = True
    CHECKPOINT_METRIC = "val_acc"  # Monitor validation accuracy
    CHECKPOINT_MODE = "max"  # Maximize validation accuracy
    
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 7
    EARLY_STOPPING_MIN_DELTA = 1e-4
    
    # ============================================================================
    # LOGGING
    # ============================================================================
    LOG_INTERVAL = 10  # Log every N batches
    SAVE_INTERVAL = 1  # Save checkpoint every N epochs
    VERBOSE = True
    
    # ============================================================================
    # VALIDATION
    # ============================================================================
    VAL_SPLIT = 0.2  # If using single directory, split into train/val
    STRATIFY = True  # Stratified split for balanced validation
    
    # ============================================================================
    # REPRODUCIBILITY
    # ============================================================================
    SEED = 42
    DETERMINISTIC = True
    
    # ============================================================================
    # KNOWLEDGE DISTILLATION PREPARATION
    # ============================================================================
    # These are not used during teacher training but documented for reference
    KD_TEMPERATURE = 4.0  # Temperature for softening logits (used later)
    KD_ALPHA = 0.7  # Balance between hard and soft targets (used later)
    
    @classmethod
    def display(cls):
        """Display configuration"""
        print("=" * 80)
        print("TinyDefectNet Teacher Model Configuration")
        print("=" * 80)
        print(f"Device: {cls.DEVICE}")
        print(f"Backbone: {cls.BACKBONE}")
        print(f"Num Classes: {cls.NUM_CLASSES}")
        print(f"Image Size: {cls.IMAGE_SIZE}")
        print("\nPhase 1 (Head Only):")
        print(f"  Epochs: {cls.PHASE1_EPOCHS}")
        print(f"  Batch Size: {cls.PHASE1_BATCH_SIZE}")
        print(f"  Learning Rate: {cls.PHASE1_LR}")
        print(f"  Freeze Backbone: {cls.PHASE1_FREEZE_BACKBONE}")
        print("\nPhase 2 (Fine-tuning):")
        print(f"  Epochs: {cls.PHASE2_EPOCHS}")
        print(f"  Batch Size: {cls.PHASE2_BATCH_SIZE}")
        print(f"  Learning Rate: {cls.PHASE2_LR}")
        print(f"  Unfreeze From Block: {cls.PHASE2_UNFREEZE_FROM_BLOCK}")
        print("=" * 80)
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        cls.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATA_ROOT.mkdir(parents=True, exist_ok=True)