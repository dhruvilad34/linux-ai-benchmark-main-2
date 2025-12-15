#!/bin/bash

# Simple run script for the project

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Load environment variables if .env exists
if [ -f .env ]; then
    source .env
fi

# Run the main script
python main.py --config config/config.yaml






