"""
Training script for fine-tuning Gemma on Bluey personality dataset
"""

import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import TrainingConfig
import warnings

warnings.filterwarnings('ignore')


def load_bluey_dataset(jsonl_path: str):
    """
    Load Bluey training dataset from JSONL file

    Format expected:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    print(f"Loading Bluey dataset from: {jsonl_path}")

    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Extract user and assistant messages
            messages = entry['messages']
            user_msg = next((m['content'] for m in messages if m['role'] == 'user'), None)
            assistant_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), None)

            if user_msg and assistant_msg:
                data.append({
                    'user': user_msg,
                    'assistant': assistant_msg
                })

    print(f"Loaded {len(data)} conversation pairs")
    return Dataset.from_list(data)


def format_bluey_conversation(example, tokenizer, max_length=512):
    """Format conversation in Gemma chat format"""
    user_text = example['user']
    assistant_text = example['assistant']

    # Format using Gemma's chat template
    text = f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n{assistant_text}<end_of_turn>"

    # Tokenize
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )

    return tokenized


def setup_model_and_tokenizer(model_name: str, device: str):
    """Load model and tokenizer"""
    print(f"Loading model: {model_name}")
    print(f"Target device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # MPS works best with float32
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Move to device
    model = model.to(device)
    print(f"✓ Model loaded on {device}")

    return model, tokenizer


def train_bluey_model(
    jsonl_path: str = "../../datasets/bluey_training.jsonl.txt",
    model_name: str = "google/gemma-3-270m",
    output_dir: str = "../outputs/bluey_270m",
    num_epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
):
    """Main training function for Bluey model"""

    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
        print("✓ MPS (Apple Silicon GPU) is available")
    elif torch.cuda.is_available():
        device = "cuda"
        print("✓ CUDA GPU is available")
    else:
        device = "cpu"
        print("⚠ Using CPU")

    print("=" * 60)
    print("Bluey Fine-Tuning on Apple Silicon")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: {jsonl_path}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 60)

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name, device)

    # Load dataset
    dataset = load_bluey_dataset(jsonl_path)

    # Tokenize dataset
    print("\nTokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: format_bluey_conversation(x, tokenizer),
        remove_columns=dataset.column_names
    )

    print(f"Training samples: {len(tokenized_dataset)}")

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=10,
        logging_dir="../logs/bluey",
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=False,
        bf16=False,
        report_to="tensorboard",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Start training
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    trainer.train()

    # Save final model
    final_output = os.path.join(output_dir, "final_model")
    print(f"\n✓ Training complete! Saving final model to {final_output}")
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {final_output}")
    print(f"Logs saved to: ../logs/bluey")


if __name__ == "__main__":
    # Train with default settings (270M model)
    train_bluey_model()
