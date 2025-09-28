#!/bin/bash

# Visual Odometry Runner Script
# This script sets up the environment and runs the visual odometry system

echo "Visual Odometry System"
echo "====================="

# Check if config file is provided
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <config_file> [additional_arguments]"
    echo ""
    echo "Example:"
    echo "  $0 config.yaml"
    echo "  $0 config.yaml --max_frames 100"
    echo "  $0 config.yaml --step_mode"
    echo "  $0 config.yaml --save_trajectory trajectory.txt"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed or not in PATH"
    exit 1
fi

# Check if config file exists
CONFIG_FILE="$1"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found"
    exit 1
fi

echo "Using configuration file: $CONFIG_FILE"

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to install some dependencies"
    fi
fi

# Run the visual odometry system
echo "Starting Visual Odometry System..."
echo ""

python3 main.py "$@"

echo ""
echo "Visual Odometry System finished"