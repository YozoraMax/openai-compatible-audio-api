#!/bin/bash
# Installation script for OpenAI Compatible API dependencies

echo "Installing core dependencies..."
python3 -m pip install -r requirements-core.txt

echo "Attempting to install problematic packages with pre-built wheels..."
python3 -m pip install sentencepiece --only-binary=all --prefer-binary || echo "sentencepiece failed - will skip"
python3 -m pip install editdistance --only-binary=all --prefer-binary || echo "editdistance failed - will skip"
python3 -m pip install pyworld --only-binary=all --prefer-binary || echo "pyworld failed - will skip"

echo "Installing espnet (may work without problematic deps)..."
python3 -m pip install espnet --only-binary=all --prefer-binary || echo "espnet failed - will skip"

echo "Installing funasr and addict..."
python3 -m pip install funasr addict || echo "funasr/addict failed - will skip"

echo "Installation complete. Some packages may have been skipped due to compilation issues."
echo "The API server should still work with available packages."