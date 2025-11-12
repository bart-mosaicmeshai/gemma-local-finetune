"""
Data loading and preprocessing utilities for Gemma fine-tuning
"""

from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, Any, Optional


def load_and_prepare_dataset(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    max_seq_length: int = 512,
    split: str = "train"
) -> Any:
    """
    Load and prepare dataset for training

    Args:
        dataset_name: Name of the dataset on Hugging Face Hub
        tokenizer: Tokenizer to use for preprocessing
        max_seq_length: Maximum sequence length
        split: Dataset split to load

    Returns:
        Prepared dataset
    """
    # Load dataset
    dataset = load_dataset(dataset_name, split=split)

    print(f"Loaded {len(dataset)} examples from {dataset_name}")
    print(f"Dataset columns: {dataset.column_names}")

    return dataset


def format_instruction_prompt(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format example into instruction-following format

    This function should be adapted based on your dataset structure.
    Default assumes 'instruction' and 'response' fields.

    Args:
        example: Dictionary containing the example data

    Returns:
        Formatted example with 'text' field
    """
    # Check for common field names
    if 'instruction' in example and 'response' in example:
        instruction = example['instruction']
        response = example['response']
        text = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"
    elif 'prompt' in example and 'completion' in example:
        prompt = example['prompt']
        completion = example['completion']
        text = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{completion}<end_of_turn>"
    elif 'text' in example:
        text = example['text']
    else:
        raise ValueError(f"Unknown dataset format. Available fields: {example.keys()}")

    return {"text": text}


def tokenize_function(
    examples: Dict[str, Any],
    tokenizer: AutoTokenizer,
    max_seq_length: int = 512
) -> Dict[str, Any]:
    """
    Tokenize examples

    Args:
        examples: Batch of examples
        tokenizer: Tokenizer to use
        max_seq_length: Maximum sequence length

    Returns:
        Tokenized examples
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
        return_tensors=None,
    )


def prepare_dataset_for_training(
    dataset_name: str,
    model_name: str,
    max_seq_length: int = 512,
    hf_token: Optional[str] = None
) -> tuple:
    """
    Complete pipeline to prepare dataset for training

    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model (for tokenizer)
        max_seq_length: Maximum sequence length
        hf_token: Hugging Face token for authentication

    Returns:
        Tuple of (dataset, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True
    )

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_and_prepare_dataset(
        dataset_name,
        tokenizer,
        max_seq_length
    )

    # Format dataset
    print("Formatting dataset...")
    dataset = dataset.map(format_instruction_prompt)

    # Tokenize dataset
    print("Tokenizing dataset...")
    dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_seq_length),
        batched=True,
        remove_columns=dataset.column_names
    )

    return dataset, tokenizer
