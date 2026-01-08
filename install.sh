#!/bin/bash

# Installation script for Audio Dataset Recording Application

echo "================================================"
echo "Audio Dataset Recording Application - Setup"
echo "================================================"
echo ""

# Check if running on Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux system"
    echo ""
    echo "Installing system dependencies..."
    echo "You may be prompted for your sudo password."
    echo ""
    
    # Install system dependencies
    sudo apt-get update
    sudo apt-get install -y python3-pyqt6 python3-pyaudio portaudio19-dev python3-numpy
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ System dependencies installed successfully!"
    else
        echo ""
        echo "✗ Failed to install system dependencies."
        echo "Please run manually: sudo apt-get install python3-pyqt6 python3-pyaudio portaudio19-dev python3-numpy"
        exit 1
    fi

elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS system"
    echo ""
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Homebrew is not installed. Please install it from https://brew.sh"
        exit 1
    fi
    
    echo "Installing portaudio..."
    brew install portaudio
    
    echo "Installing Python packages..."
    pip install PyQt6 PyAudio numpy
    
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "Detected Windows system"
    echo ""
    echo "Installing Python packages..."
    pip install PyQt6 PyAudio numpy
    
else
    echo "Unknown operating system: $OSTYPE"
    echo "Please install dependencies manually."
    exit 1
fi

echo ""
echo "================================================"
echo "✓ Installation complete!"
echo "================================================"
echo ""
echo "Run the application with:"
echo "  python3 audio_recorder.py"
echo ""
