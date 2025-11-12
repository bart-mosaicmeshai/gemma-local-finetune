"""
Configuration for Gemma fine-tuning on Apple Silicon
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Training configuration parameters"""

    # Model settings
    model_name: str = "google/gemma-3-270m"  # Options: gemma-3-270m, gemma-3-1b, gemma-3-2b, gemma-3-7b

    # Dataset settings
    dataset_name: str = "bebechien/MobileGameNPC"
    dataset_config: Optional[str] = "martian"  # Options: "martian", "venusian"
    max_seq_length: int = 512

    # Training hyperparameters
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    warmup_steps: int = 10
    logging_steps: int = 5
    save_steps: int = 50

    # Optimization
    optim: str = "adamw_torch"  # Use torch's AdamW for MPS compatibility
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Device settings
    device: Optional[str] = None
    use_mps: bool = True  # Enable MPS for Apple Silicon

    # Output settings
    output_dir: str = "./outputs"
    logging_dir: str = "./logs"
    save_total_limit: int = 2

    # Memory optimization
    gradient_checkpointing: bool = True
    fp16: bool = False  # MPS doesn't support fp16
    bf16: bool = False  # Use with caution on Apple Silicon

    # Hugging Face token
    hf_token: Optional[str] = None

    def __post_init__(self):
        """Set device automatically"""
        if self.device is None:
            if torch.backends.mps.is_available() and self.use_mps:
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

    def get_device_info(self) -> str:
        """Get device information string"""
        if self.device == "mps":
            return "Apple Silicon (MPS)"
        elif self.device == "cuda":
            return f"CUDA GPU"
        else:
            return "CPU"


def get_default_config() -> TrainingConfig:
    """Get default training configuration optimized for Apple Silicon"""
    return TrainingConfig()
