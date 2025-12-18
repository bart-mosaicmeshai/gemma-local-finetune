"""
Test demonstrating min_new_tokens fix for short response problem

This script demonstrates the solution to the early stopping issue where
fine-tuned models learned to generate responses matching the training data
length (52-76 words average) and would stop prematurely even with high max_new_tokens.

The fix: Use min_new_tokens parameter to force longer generation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.inference import GemmaInference

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_min_tokens_fix.py <model_path> [prompt]")
        print("Example: python test_min_tokens_fix.py outputs/bluey_1b_it/final_model")
        return

    model_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Tell me about your family"

    print("=" * 70)
    print(f"Testing WITH min_new_tokens=50 (the fix)")
    print(f"Model: {model_path}")
    print(f"Prompt: {prompt}")
    print("=" * 70)

    inference = GemmaInference(model_path)

    print("\nParameters:")
    print("  max_new_tokens=400")
    print("  min_new_tokens=50      # Prevents early stopping")
    print("  temperature=0.8")
    print("  top_p=0.95")
    print("  top_k=50")
    print("  repetition_penalty=1.1")
    print("-" * 70)

    response = inference.generate(
        prompt,
        max_new_tokens=400,
        min_new_tokens=50,      # Force longer responses
        temperature=0.8,         # Balanced creativity
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,  # Reduce loops
        do_sample=True
    )

    word_count = len(response.split())
    print(f"Response: {response}")
    print(f"\nWord count: {word_count} words")
    print("=" * 70)
    print("\nCompare this to test_params_it.py which shows truncated responses")
    print("without min_new_tokens (typically 12-63 words).")

if __name__ == "__main__":
    main()
