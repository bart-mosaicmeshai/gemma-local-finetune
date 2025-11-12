# Gemma Local Fine-Tuning on Apple Silicon

This project adapts the [Google Gemma Hugging Face fine-tuning tutorial](https://ai.google.dev/gemma/docs/core/huggingface_text_full_finetune) to run locally on Apple Silicon (M4 Max).

## System Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS with Metal support
- Python 3.10+
- 16GB+ RAM (64GB+ recommended for larger models)
- 20GB+ free disk space

## Project Overview

Fine-tune Google Gemma models (270M, 1B, 2B, 7B variants) on custom datasets using:
- PyTorch with MPS (Metal Performance Shaders) backend
- Hugging Face Transformers & TRL
- Local execution (no cloud GPU required)

## Setup

1. Clone this repository
2. Create a virtual environment
3. Install dependencies
4. Configure Hugging Face access
5. Run training scripts

Detailed setup instructions coming soon.

## Dataset

Default: [bebechien/MobileGameNPC](https://huggingface.co/datasets/bebechien/MobileGameNPC)
Custom datasets can be used by following the format guidelines.

## Training Configuration

- Model: Gemma 270M (configurable)
- Epochs: 5
- Batch Size: 4
- Learning Rate: 5e-5
- Max Sequence Length: 512 tokens

## License

This project follows the Gemma model license terms.
