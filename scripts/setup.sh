#!/bin/bash
# Setup script for eRisk 2026 Task 1

echo "=========================================="
echo "eRisk 2026 Task 1 - Setup Script"
echo "=========================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p submissions
mkdir -p personas

# Copy .env.example to .env if .env doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "Please edit .env and add your Hugging Face token!"
fi

# Run tests
echo ""
echo "Running setup tests..."
python scripts/test_setup.py

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your HF_TOKEN"
echo "2. Run: python scripts/test_setup.py"
echo "3. Run: python quick_start.py"
echo "4. Run: streamlit run demo/app.py"
echo ""
