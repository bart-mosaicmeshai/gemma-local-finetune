"""
Test script to interact with the base (untrained/pre-trained) Gemma model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import TrainingConfig


def load_base_model(model_name: str):
    """Load the base pre-trained model"""
    print(f"Loading base model: {model_name}")
    print("This may take a few minutes on first run (downloading ~500MB)...\n")

    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
        print("✓ Using Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("✓ Using CUDA GPU")
    else:
        device = "cpu"
        print("⚠ Using CPU")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True
    ).to(device)

    model.eval()
    print("✓ Model loaded successfully\n")

    return model, tokenizer, device


def generate_response(
    model,
    tokenizer,
    device,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    top_p: float = 0.9
):
    """Generate a response from the model"""

    # Format prompt for Gemma
    formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract just the model's response
    if "<start_of_turn>model" in full_text:
        response = full_text.split("<start_of_turn>model")[-1]
        response = response.split("<end_of_turn>")[0].strip()
    else:
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


def interactive_mode(model, tokenizer, device):
    """Interactive chat mode"""
    print("=" * 60)
    print("Interactive Mode - Testing Base Gemma 3 Model")
    print("=" * 60)
    print("Type your prompts below. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            prompt = input("\n🧑 You: ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\nExiting...")
                break

            if not prompt:
                continue

            print("\n🤖 Gemma (base): ", end="", flush=True)
            response = generate_response(model, tokenizer, device, prompt)
            print(response)

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


def test_predefined_prompts(model, tokenizer, device):
    """Test with some predefined prompts"""
    test_prompts = [
        "Hello! Who are you?",
        "Tell me a short story about a brave knight.",
        "What is the capital of France?",
        "Can you help me with Python programming?",
    ]

    print("=" * 60)
    print("Testing Base Model with Predefined Prompts")
    print("=" * 60)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}/{len(test_prompts)}] Prompt: {prompt}")
        print("-" * 60)
        response = generate_response(model, tokenizer, device, prompt)
        print(f"Response: {response}\n")
        input("Press Enter to continue...")


def main():
    """Main function"""
    import sys

    config = TrainingConfig()

    print("Loading base (pre-trained) Gemma model...")
    print("This model has NOT been fine-tuned yet.\n")

    model, tokenizer, device = load_base_model(config.model_name)

    # Check if user wants test mode or interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_predefined_prompts(model, tokenizer, device)
    else:
        interactive_mode(model, tokenizer, device)


if __name__ == "__main__":
    main()
