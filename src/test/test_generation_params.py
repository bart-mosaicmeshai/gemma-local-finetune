"""
Test different generation parameters to find optimal settings for Bluey model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.inference import GemmaInference

def test_parameters(model_path, prompt):
    """Test different generation parameter combinations"""

    print("=" * 70)
    print(f"Testing model: {model_path}")
    print(f"Prompt: {prompt}")
    print("=" * 70)

    # Load model once
    inference = GemmaInference(model_path)

    # Different parameter combinations to test
    configs = [
        {
            "name": "Current (default)",
            "max_new_tokens": 200,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "do_sample": True
        },
        {
            "name": "Lower temperature",
            "max_new_tokens": 200,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 50,
            "do_sample": True
        },
        {
            "name": "Higher temperature",
            "max_new_tokens": 200,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 50,
            "do_sample": True
        },
        {
            "name": "Greedy (no sampling)",
            "max_new_tokens": 200,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 50,
            "do_sample": False
        },
        {
            "name": "More tokens + lower temp",
            "max_new_tokens": 300,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "do_sample": True
        },
        {
            "name": "Creative (high temp + high top_p)",
            "max_new_tokens": 250,
            "temperature": 0.9,
            "top_p": 0.98,
            "top_k": 60,
            "do_sample": True
        },
        {
            "name": "Focused (low temp + low top_p)",
            "max_new_tokens": 250,
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
        print("Usage: python test_generation_params.py <model_path> [prompt]")
        print("Example: python test_generation_params.py ./outputs/bluey_1b/final_model")
        return

    model_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "What's your favorite game?"

    test_parameters(model_path, prompt)


if __name__ == "__main__":
    main()
