#!/bin/bash

# Emotion Detection Systems Evaluation Script
# This script runs the complete evaluation pipeline

echo "🚀 Starting Emotion Detection Systems Evaluation"
echo "================================================"

# Check if virtual environment exists
if [ ! -d "../venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv ../venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source ../venv/bin/activate

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt

# Run evaluation
echo "🔍 Running system evaluation..."
python3 evaluate_separate.py

# Generate graphs
echo "📈 Generating comparison graphs..."
python3 generate_graphs.py

echo ""
echo "✅ Evaluation complete! Check the evaluation folder for results."
echo "📁 Results location: $(pwd)"
