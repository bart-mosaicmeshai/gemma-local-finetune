#!/bin/bash

# Setup script for Gemma fine-tuning environment on Apple Silicon

echo "Setting up Gemma fine-tuning environment..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with MPS support
echo "Installing PyTorch for Apple Silicon..."
pip install torch torchvision torchaudio

# Install other requirements
echo "Installing project dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete! To activate the environment, run:"
echo "  source venv/bin/activate"
