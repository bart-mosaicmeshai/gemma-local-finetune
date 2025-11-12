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
    dataset_config="martian",  # Options: "martian" or "venusian"
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

**Interactive Chat Mode (Recommended):**
Use the built-in chat script for a better experience:
```bash
cd src
python chat_with_model.py
```

This provides a friendly chat interface where you can have conversations with your fine-tuned model.

**Compare Base vs Fine-tuned:**
- Base model: May show repetitive loops or incoherent responses
- Fine-tuned model: Responds with learned personality (e.g., Martian dialect)
- The improvement demonstrates successful fine-tuning!

## Project Structure

```
gemma-local-finetune/
├── src/
│   ├── core/              # Core modules
│   │   ├── config.py      # Training configuration
│   │   ├── data_utils.py  # Dataset loading and preprocessing
│   │   └── inference.py   # Inference wrapper
│   ├── train/             # Training scripts
│   │   ├── train.py       # General training script
│   │   ├── train_bluey.py # Bluey-specific training
│   │   └── train_bluey_1b_it.py # 1B instruction-tuned training
│   ├── chat/              # Interactive chat interfaces
│   │   ├── chat_bluey.py  # Bluey chatbot
│   │   └── chat_with_model.py # Generic chat interface
│   ├── test/              # Testing scripts
│   │   ├── test_base_model.py # Test untrained models
│   │   ├── test_generation_params.py # Parameter optimization
│   │   └── test_params_it.py # IT model parameter testing
│   ├── outputs/           # Trained models
│   └── logs/              # Training logs
├── datasets/              # Training datasets
├── notebooks/             # Jupyter notebooks
│   └── quickstart.ipynb
└── scripts/               # Setup scripts
    └── setup_environment.sh
```

## Configuration Options

Key parameters in `TrainingConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `google/gemma-3-270m` | Model to fine-tune |
| `dataset_name` | `bebechien/MobileGameNPC` | Dataset to use |
| `dataset_config` | `"martian"` | Dataset configuration (e.g., "martian", "venusian") |
| `num_train_epochs` | `5` | Number of training epochs |
| `per_device_train_batch_size` | `4` | Batch size per device |
| `learning_rate` | `5e-5` | Learning rate |
| `max_seq_length` | `512` | Maximum sequence length |
| `use_mps` | `True` | Use Apple Silicon GPU |
| `gradient_checkpointing` | `True` | Enable for memory efficiency |

## Available Models

Start with smaller models and scale up:

### Base Models (-pt suffix)
- `google/gemma-3-270m-pt` - 270M parameters (fast training, good for testing)
- `google/gemma-3-1b-pt` - 1B parameters (pre-trained, requires more data for personality learning)
- `google/gemma-3-2b-pt` - 2B parameters (excellent quality/speed balance)
- `google/gemma-3-7b-pt` - 7B parameters (best quality, requires significant RAM)

### Instruction-Tuned Models (-it suffix) ⭐ Recommended for Personality/Style Learning
- `google/gemma-3-270m-it` - 270M parameters (fast, good for testing)
- `google/gemma-3-1b-it` - 1B parameters (better at learning conversational styles)
- `google/gemma-3-2b-it` - 2B parameters (excellent quality)
- `google/gemma-3-7b-it` - 7B parameters (best quality)

**Key Differences:**
- **Pre-trained (-pt)**: Better for general completion, requires more training data
- **Instruction-tuned (-it)**: Better for learning personalities, speaking styles, and conversational patterns from fewer examples
- **For personality/character fine-tuning**: Use `-it` models for better results with limited training data

**Note:** Smaller models may generate shorter responses. For production use, consider 1B or larger models.

## Example: Training a Bluey Personality Model

This project includes scripts for fine-tuning a Bluey character chatbot as a complete example:

### Training a Character Personality

```bash
cd src/train
python train_bluey_1b_it.py
```

This trains a `google/gemma-3-1b-it` model on Bluey conversation examples (~111 pairs) in ~5 minutes on M4 Max.

### Chatting with Bluey

```bash
cd src/chat
python chat_bluey.py ../outputs/bluey_1b_it/final_model
```

### Lessons Learned from Bluey Experiment

**What Worked:**
- ✅ Instruction-tuned models (`-it`) work much better for personality/style learning than pre-trained (`-pt`)
- ✅ 111 conversation examples sufficient for basic personality capture
- ✅ Training completes quickly: ~5 min for 1B model, 5 epochs
- ✅ MPS (Apple Silicon GPU) provides excellent performance
- ✅ Loss reduction from 5.0 → 0.1 indicates successful learning

**Challenges:**
- ⚠️ Models learned to generate short responses matching training data length (52-76 words average)
- ⚠️ Early stopping issues: models hit EOS token prematurely even with high max_new_tokens
- ⚠️ Generation parameters (temperature, top_p, top_k) have significant impact on coherence
- ⚠️ Base models (-pt) struggled more with personality consistency than instruction-tuned models

**Key Insights:**
1. **Use instruction-tuned models** for personality/character fine-tuning
2. **Training data matters**: Response lengths in training data affect generation length
3. **More examples help**: 111 examples sufficient for basic personality, but 500-1000+ would improve generalization
4. **Generation parameters are critical**: Use `min_new_tokens` to prevent early stopping
5. **Temperature sweet spot**: 0.7-0.8 works well for balanced creativity vs coherence

**Parameter Recommendations for Character Models:**
```python
response = inference.generate(
    prompt,
    max_new_tokens=400,
    min_new_tokens=50,      # Prevent early stopping
    temperature=0.8,         # Balanced creativity
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.1   # Reduce repetition
)
```

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

On M4 Max (128GB RAM) with 111 training examples:

| Model | Type | Batch Size | Total Time (5 epochs) | Time/Step |
|-------|------|-----------|----------------------|-----------|
| gemma-3-270m-pt | Pre-trained | 4 | ~3-4 minutes | ~1.3s |
| gemma-3-1b-pt | Pre-trained | 4 | ~4.9 minutes | ~2.1s |
| gemma-3-1b-it | Instruction-tuned | 4 | ~4.8 minutes | ~2.1s |
| gemma-3-2b | Pre-trained | 2 | ~30-45 minutes* | ~3-4s* |
| gemma-3-7b | Pre-trained | 1 | ~1-2 hours* | ~8-10s* |

*Estimated based on model size scaling

**Key Findings:**
- 1B models (both -pt and -it) train at nearly identical speeds (~2s/step)
- Instruction-tuned models don't add training overhead
- M4 Max GPU utilization excellent across all model sizes
- Times scale linearly with dataset size

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
