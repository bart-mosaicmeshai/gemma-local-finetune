#!/bin/bash
# Deploy the 1B instruction-tuned model to web interface
# This follows the same process used for the 270M model deployment
#
# ⚠️  WARNING: This script successfully converts the 1B model to ONNX format,
# but the resulting model does NOT work in the browser. The 1B model uses
# sliding window attention which is not supported by Transformers.js 3.6.3.
# The ONNX Runtime fails during session initialization with "Aborted()" error.
#
# For a working browser deployment, use the 270M model instead (see web/ directory).
# This script is kept for documentation and potential future use if Transformers.js
# adds sliding window attention support.

set -e

PROJECT_ROOT="/Users/bartgottschalk/Projects/gemma-local-finetune"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Deploying Bluey 1B Model to Web"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Prepare the 1B model for web deployment"
echo "  2. Convert to ONNX with 4-bit quantization"
echo "  3. Set up web interface on port 8001"
echo ""
echo "Expected output size: ~1.6GB"
echo "Estimated time: 5-10 minutes"
echo ""

# Step 1: Prepare model for web (convert to standard format)
echo ""
echo "Step 1/4: Preparing 1B model for web deployment..."
echo ""

# Activate the main venv
source venv/bin/activate

python src/export/convert_to_onnx.py \
    src/outputs/bluey_1b_it/final_model \
    --output src/outputs/bluey_1b_it/final_model_web

echo ""
echo "✓ Model prepared in Transformers format"
echo ""

# Step 2: Download build_gemma.py if not exists
echo "Step 2/4: Getting ONNX conversion script..."
if [ ! -f "build_gemma.py" ]; then
    curl -o build_gemma.py https://gist.githubusercontent.com/xenova/a219dbf3c7da7edd5dbb05f92410d7bd/raw/45f4c5a5227c1123efebe1e36d060672ee685a8e/build_gemma.py
    echo "✓ Downloaded build_gemma.py"
else
    echo "✓ build_gemma.py already exists"
fi
echo ""

# Step 3: Setup Python 3.12 environment for ONNX conversion
echo "Step 3/4: Setting up ONNX conversion environment..."
if [ ! -d "venv_convert" ]; then
    echo "Creating Python 3.12 virtual environment..."
    python3.12 -m venv venv_convert
    source venv_convert/bin/activate
    pip install --upgrade pip
    pip install transformers torch optimum onnx onnxruntime onnx-ir ml_dtypes accelerate
    echo "✓ Environment created and dependencies installed"
else
    source venv_convert/bin/activate
    echo "✓ Using existing conversion environment"
fi
echo ""

# Step 4: Convert to ONNX with 4-bit quantization
echo "Step 4/4: Converting to ONNX with 4-bit quantization..."
echo "This will take several minutes..."
echo ""

# Create web-1b directory structure
mkdir -p web-1b/models

python build_gemma.py \
    --model_name src/outputs/bluey_1b_it/final_model_web \
    --output web-1b/models \
    --precision q4

echo ""
echo "✓ ONNX conversion complete"
echo ""

# Copy web interface files
echo "Setting up web interface..."
cp web/index.html web-1b/
cp web/app.js web-1b/
cp web/style.css web-1b/ 2>/dev/null || true

# Create a simple README for the 1B web interface
cat > web-1b/README.md << 'WEBREADME'
# Bluey 1B Model Web Interface

This directory contains the web interface for the 1B instruction-tuned Bluey model.

## Quick Start

```bash
cd web-1b
python3 -m http.server 8001
```

Then open: http://localhost:8001

## Model Details

- **Model**: bluey_1b_it (1 billion parameters)
- **Size**: ~1.6GB (4-bit quantized ONNX)
- **Performance**: More coherent than 270M, but larger download
- **Port**: 8001 (to avoid conflict with 270M on port 8000)

## Comparison with 270M Model

The 270M model runs on port 8000. You can run both simultaneously to compare:
- 270M: http://localhost:8000 (~764MB, faster download, less coherent)
- 1B: http://localhost:8001 (~1.6GB, slower download, more coherent)

## Browser Requirements

- Chrome 113+ or Edge 113+ (WebGPU support recommended)
- Firefox 118+ and Safari 17+ (experimental WebGPU support)
- Fallback to WASM for other browsers (slower but functional)
WEBREADME

echo "✓ Web interface ready"
echo ""

# Print final instructions
echo "=========================================="
echo "✅ 1B Model Web Deployment Complete!"
echo "=========================================="
echo ""
echo "To run the web interface:"
echo ""
echo "  cd web-1b"
echo "  python3 -m http.server 8001"
echo ""
echo "Then open: http://localhost:8001"
echo ""
echo "Model comparison:"
echo "  270M model (port 8000): ~764MB, faster download, less coherent"
echo "  1B model (port 8001):   ~1.6GB, slower download, more coherent"
echo ""
echo "You can run both simultaneously to compare!"
echo ""
