"""
Interactive chat with the fine-tuned Gemma model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.inference import GemmaInference


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./outputs/final_model"

    print("=" * 60)
    print("Chat with Your Fine-Tuned Martian NPC")
    print("=" * 60)
    print(f"Loading model from: {model_path}\n")

    # Load model
    inference = GemmaInference(model_path)

    print("\n" + "=" * 60)
    print("Ready to chat! Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("🧑 You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            print("🛸 Martian NPC: ", end="", flush=True)
            response = inference.generate(
                user_input,
                max_new_tokens=200,
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                do_sample=True
            )
            print(response + "\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            continue


if __name__ == "__main__":
    main()
