# Gemma Local Fine-Tuning on Apple Silicon

This project adapts the [Google Gemma Hugging Face fine-tuning tutorial](https://ai.google.dev/gemma/docs/core/huggingface_text_full_finetune) to run locally on Apple Silicon using MPS (Metal Performance Shaders).

## Why Local Fine-Tuning?

Running on Apple Silicon offers several advantages over cloud notebooks:
- **No time limits**: Train as long as you need
- **Better hardware**: M4 Max with 128GB unified memory
- **Privacy**: Your data stays local
- **Cost**: Free vs paid cloud GPU access
- **Flexibility**: Full control over your environment

## System Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS with Metal support
- Python 3.10+
- 16GB+ RAM (64GB+ recommended for larger models)
- 20GB+ free disk space

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/bart-mosaicmeshai/gemma-local-finetune.git
cd gemma-local-finetune

# Run setup script
./scripts/setup_environment.sh

# Activate virtual environment
source venv/bin/activate
```

### 2. Get Hugging Face Access (Required)

Gemma models require authentication:

1. Create account at [huggingface.co](https://huggingface.co)
2. Accept Gemma license at [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m)
3. Create access token at [Settings > Tokens](https://huggingface.co/settings/tokens)
4. Login via CLI:
```bash
huggingface-cli login
```

### 3. Test the Base Model (Optional but Recommended)

Before fine-tuning, interact with the base model to see its behavior:

**Interactive Chat Mode:**
```bash
cd src
python test_base_model.py
```

Type your questions and see how the untrained model responds. Type `quit` to exit.

**Test with Predefined Prompts:**
```bash
cd src
python test_base_model.py --test
```

**Note:** The base model may show odd behavior like repetition loops or incoherent responses. This demonstrates why fine-tuning is valuable!

### 4. Run Training

**Option A: Using Python Script**
```bash
cd src
python train.py
```

**Option B: Using Jupyter Notebook**
```bash
jupyter notebook notebooks/quickstart.ipynb
```

**Option C: Custom Configuration**
```python
from src.config import TrainingConfig
from src.train import train

config = TrainingConfig(
    model_name="google/gemma-3-270m",
    dataset_name="bebechien/MobileGameNPC",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
)

train(config)
```

### 5. Test Your Fine-tuned Model

After training completes, test your fine-tuned model:

**Quick Test (Single Prompt):**
```bash
python src/inference.py outputs/final_model "Hello! Tell me about yourself."
```

**Interactive Chat Mode:**
Create an interactive script to chat with your fine-tuned model:
```bash
cd src
python -c "
from inference import GemmaInference

# Load fine-tuned model
inference = GemmaInference('../outputs/final_model')

print('Chat with your fine-tuned model. Type quit to exit.\\n')

while True:
    prompt = input('You: ').strip()
    if prompt.lower() in ['quit', 'exit', 'q']:
        break
    if prompt:
        response = inference.generate(prompt, max_new_tokens=150)
        print(f'Model: {response}\\n')
"
```

**Compare Base vs Fine-tuned:**
Test the same prompts on both models to see the improvement!

## Project Structure

```
gemma-local-finetune/
├── src/
│   ├── config.py           # Training configuration
│   ├── data_utils.py       # Dataset loading and preprocessing
│   ├── train.py            # Main training script
│   └── inference.py        # Inference script
├── notebooks/
│   └── quickstart.ipynb    # Interactive notebook
├── scripts/
│   └── setup_environment.sh # Environment setup
├── datasets/               # Custom datasets (optional)
├── outputs/               # Trained models
└── logs/                  # Training logs
```

## Configuration Options

Key parameters in `TrainingConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `google/gemma-3-270m` | Model to fine-tune |
| `dataset_name` | `bebechien/MobileGameNPC` | Dataset to use |
| `num_train_epochs` | `5` | Number of training epochs |
| `per_device_train_batch_size` | `4` | Batch size per device |
| `learning_rate` | `5e-5` | Learning rate |
| `max_seq_length` | `512` | Maximum sequence length |
| `use_mps` | `True` | Use Apple Silicon GPU |
| `gradient_checkpointing` | `True` | Enable for memory efficiency |

## Available Models

Start with smaller models and scale up:

- `google/gemma-3-270m` - 270M parameters (recommended for testing)
- `google/gemma-3-1b` - 1B parameters
- `google/gemma-3-2b` - 2B parameters
- `google/gemma-3-7b` - 7B parameters (requires significant RAM)

## Using Custom Datasets

Your dataset should have one of these formats:

**Format 1: Instruction-Response**
```json
{
  "instruction": "User message here",
  "response": "Model response here"
}
```

**Format 2: Prompt-Completion**
```json
{
  "prompt": "User prompt here",
  "completion": "Model completion here"
}
```

Update `src/data_utils.py` to adapt the `format_instruction_prompt` function for your format.

## Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir logs
```

Then open http://localhost:6006 in your browser.

## Tips for Apple Silicon

1. **Start small**: Begin with `gemma-3-270m` before trying larger models
2. **Monitor memory**: Use Activity Monitor to watch RAM usage
3. **Reduce batch size**: If you run out of memory, lower `per_device_train_batch_size`
4. **Enable gradient checkpointing**: Saves memory at cost of speed
5. **Use float32**: MPS works best with float32 precision (not fp16/bf16)
6. **Close other apps**: Free up RAM for training
7. **Keep plugged in**: Training is GPU-intensive

## Troubleshooting

### Out of Memory
- Reduce `per_device_train_batch_size` (try 2 or 1)
- Enable `gradient_checkpointing=True`
- Use smaller model variant
- Reduce `max_seq_length`

### MPS Not Available
- Check macOS version (requires macOS 12.3+)
- Update PyTorch: `pip install --upgrade torch`
- Falls back to CPU automatically

### Slow Training
- Ensure you're using MPS (check logs for "Apple Silicon (MPS)")
- Close memory-intensive applications
- Try smaller batch size with gradient accumulation

### Model Access Denied
- Ensure you've accepted the Gemma license on Hugging Face
- Run `huggingface-cli login` with a valid token
- Check token has read permissions

## Performance Expectations

On M4 Max (128GB RAM):

| Model | Batch Size | Time/Epoch (approx) |
|-------|-----------|---------------------|
| gemma-3-270m | 4 | ~5-10 minutes |
| gemma-3-1b | 4 | ~15-25 minutes |
| gemma-3-2b | 2 | ~30-45 minutes |
| gemma-3-7b | 1 | ~1-2 hours |

*Times vary based on dataset size and sequence length*

## Comparison: Local vs Colab

| Feature | Local (Apple Silicon) | Google Colab |
|---------|---------------------|--------------|
| Cost | Free | Free tier limited, Pro $10-50/mo |
| Time limits | None | Disconnect after idle / 24h max |
| RAM | 128GB (your system) | 12-40GB |
| Storage | Persistent | Temporary |
| Privacy | Complete | Cloud-based |
| Setup | One-time | Per session |

## Next Steps

1. **Experiment with hyperparameters**: Adjust learning rate, batch size, epochs
2. **Try different models**: Start small, scale up
3. **Use your own data**: Adapt dataset loading for custom formats
4. **Evaluate results**: Test with various prompts
5. **Optimize for your use case**: Tune for your specific task

## Resources

- [Original Tutorial](https://ai.google.dev/gemma/docs/core/huggingface_text_full_finetune)
- [Gemma Models](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)

## License

This project follows the Gemma model license terms. See individual model pages on Hugging Face for details.
