# Web Deployment Guide

This guide explains how to convert your fine-tuned Gemma models to ONNX format and deploy them for browser-based inference using Transformers.js.

## Overview

The deployment process involves:
1. Converting PyTorch models to ONNX format with external data format
2. Quantizing ONNX models to reduce size (4-bit quantization)
3. Deploying to a web interface with Transformers.js and WebGPU/WASM

## Prerequisites

- Python 3.12 (ONNX Runtime doesn't support Python 3.14 yet)
- A fine-tuned model in `src/outputs/`
- Modern web browser with WebGPU support (Chrome 113+, Edge 113+)

## Quick Start

```bash
# 1. Create Python 3.12 environment
python3.12 -m venv venv_convert
source venv_convert/bin/activate

# 2. Install dependencies
pip install transformers torch optimum onnx onnxruntime onnx-ir

# 3. Download conversion script
curl -o build_gemma.py https://gist.githubusercontent.com/xenova/a219dbf3c7da7edd5dbb05f92410d7bd/raw/45f4c5a5227c1123efebe1e36d060672ee685a8e/build_gemma.py

# 4. Convert and quantize
python build_gemma.py \
  --model_name src/outputs/bluey_270m/final_model_web \
  --output web/models \
  --precision q4

# 5. Start web server
cd web
python3 -m http.server 8000

# 6. Open http://localhost:8000
```

## Detailed Steps

### Step 1: Setup Conversion Environment

Create a separate Python 3.12 environment for conversion:

```bash
# Create Python 3.12 virtual environment
python3.12 -m venv venv_convert

# Activate it
source venv_convert/bin/activate  # macOS/Linux
# or
venv_convert\Scripts\activate  # Windows

# Install required packages
pip install transformers torch optimum onnx onnxruntime onnx-ir ml_dtypes
```

**Why Python 3.12?** ONNX Runtime 1.20+ (required for proper quantization) doesn't support Python 3.14 yet.

### Step 2: Download Conversion Script

The `build_gemma.py` script from Xenova handles the complex ONNX export with external data format:

```bash
curl -o build_gemma.py https://gist.githubusercontent.com/xenova/a219dbf3c7da7edd5dbb05f92410d7bd/raw/45f4c5a5227c1123efebe1e36d060672ee685a8e/build_gemma.py
```

This script:
- Exports Gemma models to ONNX with proper graph structure
- Splits large models into `.onnx` (graph) + `.onnx_data` (weights) format
- Applies 4-bit quantization
- Generates Transformers.js-compatible configuration

### Step 3: Prepare Your Model

If you haven't already, prepare your model in standard Transformers format:

```bash
python src/export/convert_to_onnx.py src/outputs/bluey_270m/final_model
```

This creates `src/outputs/bluey_270m/final_model_web/` with all necessary files.

### Step 4: Convert to ONNX with Quantization

Run the build_gemma.py script:

```bash
python build_gemma.py \
  --model_name src/outputs/bluey_270m/final_model_web \
  --output web/models \
  --precision q4
```

**Available Precision Modes:**
- `q4` - 4-bit quantization (smallest, recommended for web, ~764MB for 270M model)
- `q4f16` - Mixed 4-bit/16-bit (balance of size and accuracy)
- `fp16` - 16-bit floating point (larger but more accurate)
- `fp32` - Full precision (largest, highest quality)

You can specify multiple modes: `--precision q4 fp16`

### Step 5: Output Files

The conversion creates these files in `web/models/`:

```
web/models/
├── onnx/
│   ├── model_q4.onnx          # ONNX graph structure (234KB)
│   └── model_q4.onnx_data     # Model weights (764MB)
├── config.json                 # Model configuration with transformers.js_config
├── generation_config.json      # Generation parameters
├── tokenizer.json             # Tokenizer vocabulary
├── tokenizer_config.json      # Tokenizer configuration
└── special_tokens_map.json    # Special tokens
```

**Key Difference from optimum-cli:** The `build_gemma.py` script:
1. Splits the model into `.onnx` + `.onnx_data` for browser compatibility
2. Adds `transformers.js_config` section to config.json
3. Properly handles Gemma 3's architecture for web deployment
4. Uses ONNX Runtime's MatMulNBitsQuantizer for optimal quantization

### Step 6: Verify the Conversion

Check that files were created correctly:

```bash
ls -lh web/models/onnx/
# Should show:
# model_q4.onnx        (~234KB)
# model_q4.onnx_data   (~764MB)

# Check config has transformers.js settings
grep -A 5 "transformers.js_config" web/models/config.json
```

### Step 7: Start Web Server

```bash
cd web
python3 -m http.server 8000
```

**Alternative servers:**
```bash
# Node.js
npx serve

# PHP
php -S localhost:8000
```

**Important:** Don't use `file://` URLs - CORS restrictions require a proper web server.

### Step 8: Test the Deployment

1. Open http://localhost:8000 in Chrome or Edge
2. Wait for "Model loaded! Ready to chat." status
3. Type a message and test the model

First load will initialize the model (~5-10 seconds). Subsequent loads are faster.

## Model Size Comparison

For a Gemma 270M model:

| Format | Size | Files | Use Case |
|--------|------|-------|----------|
| PyTorch (FP32) | 1.0 GB | .safetensors | Training, local inference |
| ONNX (FP32) | 1.0 GB | .onnx | Cross-platform inference |
| ONNX (Q4) - Single File | 701 MB | .onnx | ❌ Doesn't work in browser |
| **ONNX (Q4) - External Data** | 764 MB | .onnx + .onnx_data | ✅ **Works in browser** |

**Why external data format?** Browsers have memory limitations with single large files. Splitting into graph + data files allows proper loading and execution.

## Understanding the build_gemma.py Script

The script performs these operations:

1. **Model Loading**: Loads your PyTorch model with Transformers
2. **Graph Construction**: Builds ONNX computation graph with proper operators
3. **RoPE Cache Setup**: Configures rotary position embeddings for Gemma
4. **Quantization**: Applies 4-bit quantization to MatMul operations
5. **External Data**: Saves large tensors separately for browser compatibility
6. **Config Generation**: Adds Transformers.js-specific configuration

## Performance Notes

**Conversion Time (M4 Max):**
- ONNX Export + Quantization: ~30-60 seconds for 270M model

**Browser Inference Speed:**
- WebGPU: 5-10 tokens/second (GPU-accelerated)
- WASM: 1-3 tokens/second (CPU fallback)

**Quality Impact:**
- Q4 quantization: Minimal quality loss for chat applications
- Q4F16: Better quality for complex reasoning tasks
- Always validate on your specific use case

## Troubleshooting

### "Unsupported model type: gemma3_text"

This means you're using an older version of Transformers.js. The web interface uses `@huggingface/transformers@3.6.3` which supports Gemma 3.

**Solution:** Update the CDN URL in `web/app.js`:
```javascript
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.6.3';
```

### "WASM Error 12010720"

This error occurs when:
1. WASM files can't be loaded (CORS/MIME type issues)
2. Model format is incompatible

**Solution:** Ensure you're using `build_gemma.py` (not `optimum-cli`) for proper format.

### Model generates nonsensical output

When using wrong model format (single file ONNX), the model loads but generates garbage.

**Solution:**
1. Delete old model files
2. Re-convert using `build_gemma.py`
3. Verify both `.onnx` and `.onnx_data` files exist

### Python Version Issues

**Error:** `ModuleNotFoundError: No module named 'onnxruntime'`

**Solution:** Use Python 3.12:
```bash
# Install via Homebrew (macOS)
brew install python@3.12

# Or use pyenv
pyenv install 3.12.0
pyenv local 3.12.0
```

### Memory Issues During Conversion

If conversion fails with OOM errors:
- Close other applications
- Convert on a machine with more RAM (16GB+ recommended)
- Use smaller batch sizes in training to reduce model size

### Browser Compatibility

| Browser | Version | WebGPU | Status |
|---------|---------|--------|--------|
| Chrome  | 113+    | ✅ Yes | Recommended |
| Edge    | 113+    | ✅ Yes | Recommended |
| Firefox | 118+    | 🚧 Experimental | Limited support |
| Safari  | 17+     | 🚧 Experimental | Limited support |

## Web Interface Configuration

The web interface (`web/app.js`) automatically:
- Detects WebGPU availability and falls back to WASM
- Loads models from the `./models` directory
- Configures proper paths for external data format
- Handles tokenization and generation

**Default Generation Parameters:**
```javascript
maxNewTokens: 100,
temperature: 0.8,
topP: 0.95,
repetitionPenalty: 1.1,
```

Adjust these in `web/app.js` CONFIG section for different behavior.

## Production Deployment

For deploying to production:

### Static Hosting (Recommended)

```bash
# Netlify
npm install -g netlify-cli
cd web
netlify deploy --prod

# Vercel
npm install -g vercel
cd web
vercel --prod

# GitHub Pages
# Push web/ contents to gh-pages branch
```

### Important Considerations

1. **Model Size**: First load downloads ~764MB
   - Consider adding progress indicator
   - Enable browser caching
   - Warn users about data usage

2. **HTTPS Required**: Most browsers require HTTPS for WebGPU

3. **CORS Headers**: Ensure server allows loading .onnx files

4. **CDN**: Consider hosting model files on a CDN for faster loading

## Next Steps

- Optimize generation parameters for your use case
- Add conversation history/context management
- Implement stop generation button
- Add model switching for different personalities
- Deploy to production hosting

## Resources

- [Original Tutorial](https://developers.googleblog.com/en/own-your-ai-fine-tune-gemma-3-270m-for-on-device/)
- [Transformers.js Documentation](https://huggingface.co/docs/transformers.js)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)
- [build_gemma.py Script](https://gist.github.com/xenova/a219dbf3c7da7edd5dbb05f92410d7bd)
- [Gemma Cookbook Examples](https://github.com/google-gemini/gemma-cookbook)
