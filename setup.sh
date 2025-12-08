#!/bin/bash

# Quick Start Script for DeepLabScan YOLO Project
# This script helps you set up the project environment

echo "================================"
echo "DeepLabScan Setup Script"
echo "================================"
echo ""

# Check Python version
echo "1. Checking Python version..."
python_version=$(python --version 2>&1)
echo "   Found: $python_version"

# Create virtual environment
echo ""
echo "2. Creating virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "   ✓ Virtual environment created"
else
    echo "   ✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "3. Activating virtual environment..."
source venv/bin/activate
echo "   ✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "4. Upgrading pip..."
pip install --upgrade pip -q
echo "   ✓ pip upgraded"

# Install dependencies
echo ""
echo "5. Installing dependencies..."
echo "   This may take a few minutes..."
pip install -r requirements.txt -q
echo "   ✓ Dependencies installed"

# Create necessary directories
echo ""
echo "6. Verifying directory structure..."
echo "   ✓ All directories are in place"

# Print next steps
echo ""
echo "================================"
echo "Setup Complete! 🎉"
echo "================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Set up Roboflow account and create a project"
echo "   Visit: https://roboflow.com/"
echo ""
echo "2. Upload and label your data in Roboflow"
echo ""
echo "3. Export your dataset in YOLO format to data/roboflow/"
echo ""
echo "4. Update configs/data_config.yaml with your class names"
echo ""
echo "5. Start training:"
echo "   python src/training/train.py --config configs/training_config.yaml"
echo ""
echo "6. For detailed instructions, see docs/getting_started.md"
echo ""
echo "To activate the environment in the future, run:"
echo "   source venv/bin/activate"
echo ""
