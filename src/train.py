"""
Training script for fine-tuning Gemma models on Apple Silicon
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from config import TrainingConfig, get_default_config
from data_utils import prepare_dataset_for_training
import warnings

warnings.filterwarnings('ignore')


def setup_model_and_tokenizer(config: TrainingConfig):
    """
    Load model and tokenizer with MPS optimizations

    Args:
        config: Training configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {config.model_name}")
    print(f"Target device: {config.get_device_info()}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        token=config.hf_token,
        trust_remote_code=True
    )

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        token=config.hf_token,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # MPS works best with float32
    )

    # Enable gradient checkpointing for memory efficiency
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Move model to device
    if config.device == "mps":
        # MPS-specific optimizations
        model = model.to(config.device)
        print("✓ Model loaded on Apple Silicon (MPS)")
    elif config.device == "cuda":
        model = model.to(config.device)
        print("✓ Model loaded on CUDA GPU")
    else:
        print("⚠ Running on CPU (this will be slow)")

    return model, tokenizer


def create_training_arguments(config: TrainingConfig) -> TrainingArguments:
    """
    Create training arguments optimized for Apple Silicon

    Args:
        config: Training configuration

    Returns:
        TrainingArguments object
    """
    # Note: use_mps_device is deprecated, device is handled automatically
    args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_dir=config.logging_dir,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        optim=config.optim,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        fp16=False,  # MPS doesn't support fp16
        bf16=False,  # Be cautious with bf16 on Apple Silicon
        report_to="tensorboard",
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Use 0 for MPS compatibility
    )

    return args


def train(config: TrainingConfig = None):
    """
    Main training function

    Args:
        config: Training configuration (uses default if None)
    """
    if config is None:
        config = get_default_config()

    print("=" * 60)
    print("Gemma Fine-Tuning on Apple Silicon")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Device: {config.get_device_info()}")
    print(f"Batch size: {config.per_device_train_batch_size}")
    print(f"Epochs: {config.num_train_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print("=" * 60)

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)

    # Prepare dataset
    print("\nPreparing dataset...")
    train_dataset, _ = prepare_dataset_for_training(
        config.dataset_name,
        config.model_name,
        config.max_seq_length,
        config.hf_token,
        config.dataset_config
    )

    print(f"Training samples: {len(train_dataset)}")

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal language modeling
    )

    # Create training arguments
    training_args = create_training_arguments(config)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Start training
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    trainer.train()

    # Save final model
    final_output_dir = os.path.join(config.output_dir, "final_model")
    print(f"\n✓ Training complete! Saving final model to {final_output_dir}")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {final_output_dir}")
    print(f"Logs saved to: {config.logging_dir}")
    print("\nTo view training logs, run:")
    print(f"  tensorboard --logdir {config.logging_dir}")


if __name__ == "__main__":
    # Check MPS availability
    if torch.backends.mps.is_available():
        print("✓ MPS (Apple Silicon GPU) is available")
    else:
        print("⚠ MPS not available, will use CPU")

    # Run training with default config
    train()
