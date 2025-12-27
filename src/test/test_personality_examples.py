"""
Test the 4 personality test prompts from Part 6 of the blog series.
These tests validate the claims about personality testing methodology.
"""

import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.inference import GemmaInference


def count_words(text):
    """Count words in text"""
    return len(text.split())


def main():
    # Test prompts from Part 6 blog post
    test_prompts = [
        "What's your favorite game?",
        "I'm feeling sad",
        "How do you handle disagreements?",
        "Explain quantum physics"
    ]

    expected_behaviors = [
        "should mention Keepy Uppy, Magic Claw",
        "should show empathy, give examples from her experience",
        "kid wisdom, references Bingo",
        "should stay in character, admit not knowing like a kid would"
    ]

    model_path = "../outputs/bluey_1b_it/final_model"

    print("=" * 80)
    print("Part 6 Blog Post: Personality Testing Examples")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    print("Testing personality evaluation methodology with 4 example prompts")
    print(f"Model: {model_path}")
    print("Parameters: max_new_tokens=400, min_new_tokens=50, temperature=0.8, top_p=0.95")
    print()

    # Load model
    print("Loading model...")
    inference = GemmaInference(model_path)
    print("Model loaded!\n")

    results = []

    for i, (prompt, expected) in enumerate(zip(test_prompts, expected_behaviors), 1):
        print("=" * 80)
        print(f"TEST {i}: \"{prompt}\"")
        print(f"Expected behavior: {expected}")
        print("=" * 80)
        print()

        response = inference.generate(
            prompt,
            max_new_tokens=400,
            min_new_tokens=50,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.1
        )

        word_count = count_words(response)

        print(f"Bluey's response ({word_count} words):")
        print(response)
        print()
        print("-" * 80)
        print()

        results.append({
            'prompt': prompt,
            'expected': expected,
            'response': response,
            'word_count': word_count
        })

    # Save results to log file
    log_path = "../logs/personality_test_results.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Part 6 Blog Post: Personality Testing Examples\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write("Testing personality evaluation methodology with 4 example prompts\n")
        f.write(f"Model: {model_path}\n")
        f.write("Parameters: max_new_tokens=400, min_new_tokens=50, temperature=0.8, top_p=0.95\n\n")

        for i, result in enumerate(results, 1):
            f.write("=" * 80 + "\n")
            f.write(f"TEST {i}: \"{result['prompt']}\"\n")
            f.write(f"Expected behavior: {result['expected']}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Bluey's response ({result['word_count']} words):\n")
            f.write(result['response'] + "\n\n")
            f.write("-" * 80 + "\n\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("ANALYSIS:\n")
        f.write("=" * 80 + "\n")
        f.write("These examples demonstrate personality testing methodology:\n")
        f.write("- Tests 1-3 evaluate personality consistency (catchphrases, family references, tone)\n")
        f.write("- Test 4 evaluates edge case handling (abstract topics)\n")
        f.write("- Human judgment required to assess if responses 'sound like Bluey'\n")
        f.write("- Responses should be evaluated for character breaking vs staying in voice\n")

    print("=" * 80)
    print(f"Results saved to: {log_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
