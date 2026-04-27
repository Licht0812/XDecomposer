"""Configuration for XRD datasets."""

from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class OnlineMixingConfig:
    """Configuration for Online Mixing XRD Dataset."""
    
    # Dataset Generation
    MIN_K: int = 2
    MAX_K: int = 4
    MIN_WEIGHT: float = 0.15
    K_DISTRIBUTION: str = "weighted"  # "uniform" or "weighted"
    K_WEIGHTS: Tuple[float, ...] = (0.4, 0.3, 0.15, 0.1, 0.05)
    WEIGHT_DISTRIBUTION: str = "random"  # "random" (Dirichlet) or "equal"
    
    # Signal Processing
    XRD_LENGTH: int = 3500
    NORM_METHOD: str = "max"  # "max", "minmax", or "none"
    AUGMENT: bool = False
    NOISE_LEVEL: float = 0.001
    
    # Optimization
    RANDOM_SINGLE_SAMPLE: bool = True
    
    # Splitting
    TRAIN_RATIO: float = 0.8
    VAL_RATIO: float = 0.1
    TEST_RATIO: float = 0.1
    SEED: int = 42
