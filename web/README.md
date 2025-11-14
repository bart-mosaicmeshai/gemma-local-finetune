# Bluey AI Web Interface

A browser-based chat interface for your fine-tuned Gemma models using Transformers.js and WebGPU/WASM.

## Features

- 🌐 **Runs entirely in the browser** - No server required for inference
- 🔒 **100% Private** - All processing happens locally on your device
- 📴 **Works offline** - After initial model load
- 💰 **Free** - No API costs or usage limits
- ⚡ **WebGPU accelerated** - Fast inference on supported devices (5-10 tokens/sec)
- 🔄 **WASM fallback** - Works on older browsers (1-3 tokens/sec)

## Prerequisites

- Modern web browser with WebGPU support (Chrome 113+, Edge 113+)
- ONNX model files from the conversion process (see [DEPLOYMENT.md](../DEPLOYMENT.md))
- Local web server for development

## Quick Start

### 1. Convert Your Model

If you haven't already, convert your fine-tuned model to ONNX format:

```bash
# From project root
python3.12 -m venv venv_convert
source venv_convert/bin/activate
pip install transformers torch optimum onnx onnxruntime onnx-ir

# Download conversion script
curl -o build_gemma.py https://gist.githubusercontent.com/xenova/a219dbf3c7da7edd5dbb05f92410d7bd/raw/45f4c5a5227c1123efebe1e36d060672ee685a8e/build_gemma.py

# Convert and quantize
python build_gemma.py \
  --model_name src/outputs/bluey_270m/final_model_web \
  --output web/models \
  --precision q4
```

This creates the required files in `web/models/`:
```
web/models/
├── onnx/
│   ├── model_q4.onnx          # Graph structure (~234KB)
│   └── model_q4.onnx_data     # Model weights (~764MB)
├── config.json
├── generation_config.json
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

### 2. Start Web Server

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

**Important:** You must use a web server - `file://` URLs won't work due to CORS restrictions.

### 3. Open in Browser

Navigate to **http://localhost:8000** in Chrome or Edge.

**First load behavior:**
- Model initialization takes ~5-10 seconds
- Wait for status: "Model loaded! Ready to chat."
- WebGPU detection happens automatically
- Falls back to WASM if WebGPU unavailable

### 4. Start Chatting!

1. Type your message in the text area
2. Press Enter or click Send
3. The AI responds with your fine-tuned personality!

**Tips:**
- Use Shift+Enter for multi-line messages
- WebGPU provides much faster inference (5-10 tokens/sec)
- WASM fallback is slower but works everywhere (1-3 tokens/sec)
- Responses are typically 50-100 tokens

## Configuration

You can adjust generation parameters in `app.js`:

```javascript
const CONFIG = {
    modelPath: './',           // Base path for models
    modelId: 'models',         // Model folder name
    maxNewTokens: 100,         // Maximum response length
    temperature: 0.8,          // Creativity (0.0-1.0, higher = more creative)
    topP: 0.95,               // Nucleus sampling threshold
    repetitionPenalty: 1.1,    // Reduce repetition (1.0 = off, higher = more penalty)
};
```

### Parameter Guidelines

**maxNewTokens** (50-200):
- 50-75: Quick, concise responses
- 100: Default, balanced
- 150-200: Longer, more detailed responses

**temperature** (0.1-1.0):
- 0.1-0.5: More focused, deterministic
- 0.6-0.8: Balanced creativity
- 0.9-1.0: Very creative, may be less coherent

**topP** (0.9-0.99):
- 0.9: More focused vocabulary
- 0.95: Default, good balance
- 0.99: Full vocabulary diversity

**repetitionPenalty** (1.0-1.5):
- 1.0: No penalty
- 1.1: Light penalty (default)
- 1.3-1.5: Strong penalty (use if model repeats too much)

## Technical Details

### Architecture

**Frontend (app.js):**
- ES6 modules with CDN import
- Transformers.js @huggingface/transformers@3.6.3
- Automatic WebGPU/WASM device selection
- Local model loading with external data format support

**Model Loading:**
```javascript
// Detects WebGPU support
const hasWebGPU = !!navigator.gpu;
const device = hasWebGPU ? 'webgpu' : 'wasm';

// Loads model with proper configuration
generator = await pipeline('text-generation', 'models', {
    dtype: 'q4',              // Use 4-bit quantized model
    device: device,           // WebGPU or WASM
    model_file_name: 'model', // Automatically finds model_q4.onnx
});
```

**Response Cleaning:**
- Removes input prompt from output
- Takes first line only (stops at newline)
- Strips JSON-like artifacts and special characters
- Handles edge cases gracefully

### File Structure

```
web/
├── index.html          # Main HTML interface
├── app.js             # Application logic (Transformers.js integration)
├── style.css          # Styling and animations
├── models/            # ONNX model files (you provide these)
│   ├── onnx/
│   │   ├── model_q4.onnx
│   │   └── model_q4.onnx_data
│   ├── config.json
│   ├── generation_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
└── README.md          # This file
```

## Troubleshooting

### Model doesn't load

**Error:** "Cannot find model files"
- ✅ Ensure model files are in `web/models/` directory
- ✅ Check that you're using a local web server (not `file://`)
- ✅ Verify all required files are present (see Quick Start section)
- ✅ Look at browser console (F12) for specific error messages

**Error:** "Unsupported model type: gemma3_text" with old Transformers.js
- ✅ Update app.js to use `@huggingface/transformers@3.6.3` (not `@xenova/transformers`)
- ✅ Clear browser cache and hard refresh (Cmd+Shift+R)

**Error:** "WASM Error 12010720"
- ✅ Model format is incorrect - reconvert using `build_gemma.py`
- ✅ Ensure both `.onnx` and `.onnx_data` files exist
- ✅ Don't use models converted with `optimum-cli` alone

### Slow performance

- ✅ WebGPU significantly improves speed - ensure it's enabled
- ✅ Close other browser tabs to free up memory
- ✅ Use the q4 quantized model (default, already optimized)
- ✅ Reduce `maxNewTokens` in app.js CONFIG
- ✅ Check if WebGPU is actually being used (status message shows device type)

### Generation quality issues

**Responses too short:**
- ⚙️ Increase `maxNewTokens` to 150-200
- ⚙️ Adjust `temperature` (try 0.9 for more variety)

**Repetitive responses:**
- ⚙️ Increase `repetitionPenalty` to 1.3-1.5
- ⚙️ Adjust `temperature` and `topP` values
- ⚙️ Ensure you're using the fine-tuned model (not base model)

**Incoherent or random responses:**
- ⚙️ Lower `temperature` to 0.6-0.7
- ⚙️ Verify model was fine-tuned successfully
- ⚙️ Check if model files were corrupted during conversion

**Responses have weird characters (brackets, quotes, etc.):**
- ✅ This is normal - app.js automatically cleans these up
- ✅ If persisting, check the response cleaning code in generateResponse()

## Browser Compatibility

| Browser | Version | WebGPU | Performance | Status |
|---------|---------|--------|-------------|--------|
| Chrome  | 113+    | ✅ Yes | ⚡ 5-10 tok/s | ✅ Recommended |
| Edge    | 113+    | ✅ Yes | ⚡ 5-10 tok/s | ✅ Recommended |
| Firefox | 118+    | 🚧 Experimental | 🐌 1-3 tok/s | ⚠️ Limited |
| Safari  | 17+     | 🚧 Experimental | 🐌 1-3 tok/s | ⚠️ Limited |

**Note:** Without WebGPU, the model runs on CPU via WASM (slower but functional).

## Deployment to Production

### Static Hosting Services

Deploy your web interface to any static hosting service:

**Netlify:**
```bash
npm install -g netlify-cli
cd web
netlify deploy --prod
```

**Vercel:**
```bash
npm install -g vercel
cd web
vercel --prod
```

**GitHub Pages:**
1. Create a `gh-pages` branch
2. Copy `web/` contents to the branch
3. Enable GitHub Pages in repository settings

### Important Production Considerations

1. **Model Size Warning**
   - Initial load downloads ~764MB
   - Consider adding:
     - Loading progress indicator
     - Data usage warning for mobile users
     - Browser caching headers

2. **HTTPS Required**
   - Most browsers require HTTPS for WebGPU
   - Use hosting services that provide automatic HTTPS

3. **CORS Headers**
   - Ensure server sends proper CORS headers for `.onnx` files
   - Static hosts usually handle this automatically

4. **CDN Hosting**
   - Consider hosting model files on a CDN for faster loading
   - Update `modelPath` in app.js to CDN URL

5. **Security**
   - No server-side processing needed
   - All data stays in user's browser
   - No privacy concerns with user inputs

## Performance Benchmarks

Typical performance on M4 Max (WebGPU enabled):

| Metric | Value |
|--------|-------|
| Model Load Time | 5-10 seconds (after download) |
| First Token | ~2-3 seconds |
| Tokens/Second | 5-10 tokens/sec (WebGPU) |
| Tokens/Second | 1-3 tokens/sec (WASM) |
| Memory Usage | ~1-2 GB |
| Model Download | ~764 MB (one-time) |

## Privacy & Security

✅ **All data stays local** - No information sent to external servers
✅ **No tracking** - No analytics or telemetry
✅ **Open source** - Inspect the code yourself
✅ **Offline capable** - Works without internet after initial load
✅ **No API keys** - No authentication required
✅ **Browser sandbox** - Protected by browser security model

## Limitations

- Large model size (~764MB download on first use)
- Requires modern browser with WebGPU for best performance
- Generation speed depends on device capabilities
- Context window limited by model configuration (typically 512-2048 tokens)
- No conversation history management (single-turn responses)

## Future Enhancements

Potential improvements for production use:

- [ ] Add conversation history and context management
- [ ] Implement stop generation button
- [ ] Add token streaming for progressive display
- [ ] Implement chat export/save functionality
- [ ] Add model switching UI
- [ ] Progressive download with loading indicator
- [ ] Service worker for offline caching
- [ ] Multi-turn conversation support
- [ ] Adjustable parameters UI

## Resources

- [DEPLOYMENT.md](../DEPLOYMENT.md) - Complete deployment guide
- [Transformers.js Documentation](https://huggingface.co/docs/transformers.js)
- [WebGPU Specification](https://gpuweb.github.io/gpuweb/)
- [Gemma Model Card](https://huggingface.co/google/gemma-3-270m)
- [Original Google Tutorial](https://developers.googleblog.com/en/own-your-ai-fine-tune-gemma-3-270m-for-on-device/)

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section above
2. Review [DEPLOYMENT.md](../DEPLOYMENT.md) for conversion issues
3. Check browser console (F12) for detailed error messages
4. Verify model files are correctly converted with `build_gemma.py`

## License

This web interface is part of the gemma-local-finetune project and follows the same license terms as the main project.
