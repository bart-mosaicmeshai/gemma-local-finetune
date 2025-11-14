#!/usr/bin/env python3
"""
Prepare a fine-tuned Gemma model for web deployment with Transformers.js.

This script prepares the model files in a format that can be converted
to ONNX using the Transformers.js Python converter (available via npx).

The workflow is:
1. This script validates and prepares the model
2. You use the Transformers.js converter to create ONNX files
3. The ONNX model can be deployed in a browser with WebGPU
"""

import argparse
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import shutil


def prepare_for_web(model_path: str, output_path: str):
    """
    Prepare a Hugging Face model for web deployment.

    Args:
        model_path: Path to the fine-tuned model
        output_path: Directory where prepared model will be saved
    """
    model_path = Path(model_path)
    output_path = Path(output_path)

    if not model_path.exists():
        print(f"Error: Model path {model_path} does not exist!")
        sys.exit(1)

    print(f"Preparing model from: {model_path}")
    print(f"Output directory: {output_path}")
    print()

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading model and tokenizer...")
    print("This may take a few minutes for larger models.")
    print()

    try:
        # Load the model and tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # Use FP32 for compatibility
            device_map="cpu",  # Load on CPU
        )

        print()
        print("Saving model in Transformers format...")

        # Save the model and tokenizer in standard format
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        print()
        print("✅ Model prepared successfully!")
        print(f"Model saved to: {output_path}")
        print()
        print("=" * 70)
        print("NEXT STEPS - Convert to ONNX using Transformers.js converter:")
        print("=" * 70)
        print()
        print("1. Install Node.js if you haven't already")
        print()
        print("2. Run the Transformers.js converter:")
        print(f"   npx @huggingface/transformers.js convert \\")
        print(f"       {output_path} \\")
        print(f"       --quantize")
        print()
        print("   This will create quantized ONNX files optimized for web deployment")
        print()
        print("3. The converter will create files in the same directory")
        print("   that can be used directly in your web application")
        print()
        print("For more info: https://huggingface.co/docs/transformers.js/custom_usage")
        print()

    except Exception as e:
        print(f"❌ Error during preparation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a fine-tuned Gemma model for web deployment with Transformers.js"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the fine-tuned model directory (e.g., outputs/bluey_270m/final_model)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for prepared model (default: <model_path>_web)"
    )

    args = parser.parse_args()

    # Set default output path if not provided
    if args.output is None:
        model_path = Path(args.model_path)
        args.output = str(model_path.parent / f"{model_path.name}_web")

    prepare_for_web(args.model_path, args.output)


if __name__ == "__main__":
    main()
