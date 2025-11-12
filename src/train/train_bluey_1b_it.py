"""
Train instruction-tuned 1B Bluey model
"""

from train_bluey import train_bluey_model

if __name__ == "__main__":
    print("Training instruction-tuned 1B Bluey model...")
    print("This model is specifically designed to learn conversational styles!")

    train_bluey_model(
        model_name='google/gemma-3-1b-it',
        output_dir='../outputs/bluey_1b_it',
        num_epochs=5,
        batch_size=4,
        learning_rate=5e-5
    )
