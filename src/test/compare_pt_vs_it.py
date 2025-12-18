"""
Compare base (-pt) vs instruction-tuned (-it) models with same prompt

This test demonstrates the difference in personality consistency between
pre-trained and instruction-tuned models for blog post Part 5.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.inference import GemmaInference
from datetime import datetime

def main():
    # Test multiple prompts to see consistency
    prompts = [
        "I'm feeling sad",
        "What should I do today?",
        "Can you help me with my homework?",
        "Tell me a story"
    ]

    output_file = './logs/pt_vs_it_comparison.txt'

    with open(output_file, 'w') as f:
        f.write('=' * 80 + '\n')
        f.write('Part 5 Blog Post: Base (-pt) vs Instruction-Tuned (-it) Comparison\n')
        f.write(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('=' * 80 + '\n\n')
        f.write('Testing multiple prompts to compare personality consistency\n')
        f.write('Parameters: max_new_tokens=200, temperature=0.8, top_p=0.95, top_k=50\n\n')

        # Load both models once
        print("Loading models...")
        print("- Base model (-pt)...")
        pt_model = GemmaInference('outputs/bluey_1b/final_model')
        print("- Instruction-tuned model (-it)...")
        it_model = GemmaInference('outputs/bluey_1b_it/final_model')
        print()

        # Test each prompt with both models
        for i, prompt in enumerate(prompts, 1):
            print("=" * 80)
            print(f"PROMPT {i}/{len(prompts)}: \"{prompt}\"")
            print("=" * 80)
            f.write('=' * 80 + '\n')
            f.write(f'PROMPT {i}: "{prompt}"\n')
            f.write('=' * 80 + '\n\n')

            # Test -pt model
            print("\nBase Model (-pt):")
            print("-" * 80)
            pt_response = pt_model.generate(
                prompt,
                max_new_tokens=200,
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                do_sample=True
            )
            pt_words = len(pt_response.split())
            print(f"({pt_words} words): {pt_response}")

            f.write(f'Base Model (-pt) - {pt_words} words:\n')
            f.write(f'{pt_response}\n\n')

            # Test -it model
            print("\nInstruction-Tuned Model (-it):")
            print("-" * 80)
            it_response = it_model.generate(
                prompt,
                max_new_tokens=200,
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                do_sample=True
            )
            it_words = len(it_response.split())
            print(f"({it_words} words): {it_response}")
            print()

            f.write(f'Instruction-Tuned Model (-it) - {it_words} words:\n')
            f.write(f'{it_response}\n\n')
            f.write('-' * 80 + '\n\n')

        f.write('\n' + '=' * 80 + '\n')
        f.write('ANALYSIS:\n')
        f.write('=' * 80 + '\n')
        f.write('Compare responses for personality consistency, use of Bluey voice,\n')
        f.write('and maintenance of character across different prompts.\n')

    print(f'✅ Results saved to: {output_file}')

if __name__ == "__main__":
    main()
