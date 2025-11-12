"""
Test generation parameters specifically for instruction-tuned Bluey model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.inference import GemmaInference

def test_parameters(model_path, prompt):
    """Test different generation parameter combinations for instruction-tuned model"""

    print("=" * 70)
    print(f"Testing instruction-tuned model: {model_path}")
    print(f"Prompt: {prompt}")
    print("=" * 70)

    # Load model once
    inference = GemmaInference(model_path)

    # Parameter combinations optimized for instruction-tuned models
    configs = [
        {
            "name": "Current Creative (temp=0.9)",
            "max_new_tokens": 400,
            "temperature": 0.9,
            "top_p": 0.98,
            "top_k": 60,
            "do_sample": True
        },
        {
            "name": "Balanced (temp=0.8)",
            "max_new_tokens": 400,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "do_sample": True
        },
        {
            "name": "More focused (temp=0.7)",
            "max_new_tokens": 400,
            "temperature": 0.7,
            "top_p": 0.92,
            "top_k": 45,
            "do_sample": True
        },
        {
            "name": "Consistent (temp=0.6)",
            "max_new_tokens": 400,
            "temperature": 0.6,
            "top_p": 0.9,
            "top_k": 40,
            "do_sample": True
        },
        {
            "name": "Very focused (temp=0.5)",
            "max_new_tokens": 400,
            "temperature": 0.5,
            "top_p": 0.85,
            "top_k": 30,
            "do_sample": True
        }
    ]

    for i, config in enumerate(configs, 1):
        print(f"\n{'=' * 70}")
        print(f"Test {i}/{len(configs)}: {config['name']}")
        print(f"Parameters: temp={config['temperature']}, top_p={config['top_p']}, "
              f"top_k={config['top_k']}, max_tokens={config['max_new_tokens']}")
        print("-" * 70)

        try:
            response = inference.generate(
                prompt,
                max_new_tokens=config['max_new_tokens'],
                temperature=config['temperature'],
                top_p=config['top_p'],
                top_k=config['top_k'],
                do_sample=config['do_sample']
            )
            print(f"Response: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "=" * 70)
    print("Testing complete!")
    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_params_it.py <model_path> [prompt]")
        print("Example: python test_params_it.py ../outputs/bluey_1b_it/final_model")
        return

    model_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Tell me about your family"

    test_parameters(model_path, prompt)


if __name__ == "__main__":
    main()
