"""
Interactive chat with the fine-tuned Bluey model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.inference import GemmaInference


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./outputs/bluey_270m/final_model"

    print("=" * 60)
    print("🐶 Chat with Bluey! 🐶")
    print("=" * 60)
    print(f"Loading model from: {model_path}\n")

    # Load model
    inference = GemmaInference(model_path)

    print("\n" + "=" * 60)
    print("Ready to chat! Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    # Show some example prompts
    print("Try asking things like:")
    print("  - What's your favorite game?")
    print("  - Tell me about your family")
    print("  - Can you help me with homework?")
    print("  - I'm feeling sad today")
    print()

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nSee ya later! For real life!")
                break

            if not user_input:
                continue

            print("Bluey: ", end="", flush=True)
            response = inference.generate(
                user_input,
                max_new_tokens=400,
                min_new_tokens=50,  # Force at least 50 tokens (~40 words)
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                do_sample=True,
                repetition_penalty=1.1  # Slightly discourage repetition
            )
            print(response + "\n")

        except KeyboardInterrupt:
            print("\n\nSee ya later! For real life!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            continue


if __name__ == "__main__":
    main()
