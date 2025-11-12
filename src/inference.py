"""
Inference script for fine-tuned Gemma models
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


class GemmaInference:
    """Simple inference wrapper for Gemma models"""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_length: int = 512
    ):
        """
        Initialize inference engine

        Args:
            model_path: Path to fine-tuned model
            device: Device to use (auto-detected if None)
            max_length: Maximum generation length
        """
        self.model_path = model_path
        self.max_length = max_length

        # Detect device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device
        print(f"Loading model from: {model_path}")
        print(f"Device: {device}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to(device)

        self.model.eval()
        print("✓ Model loaded successfully")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True
    ) -> str:
        """
        Generate text from prompt

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling

        Returns:
            Generated text
        """
        # Format prompt
        formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract only the model's response
        if "<start_of_turn>model" in generated_text:
            response = generated_text.split("<start_of_turn>model")[-1]
            response = response.split("<end_of_turn>")[0].strip()
        else:
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response


def main():
    """Example usage"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inference.py <model_path> [prompt]")
        print("Example: python inference.py ./outputs/final_model \"Hello, how are you?\"")
        return

    model_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello! Tell me about yourself."

    # Initialize inference
    inference = GemmaInference(model_path)

    # Generate response
    print("\n" + "=" * 60)
    print(f"Prompt: {prompt}")
    print("=" * 60)

    response = inference.generate(prompt)

    print(f"\nResponse: {response}")
    print("=" * 60)


if __name__ == "__main__":
    main()
