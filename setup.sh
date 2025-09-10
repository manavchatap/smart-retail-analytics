#!/bin/bash

# Smart Retail Analytics System Setup Script

echo "🚀 Setting up Smart Retail Analytics System..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Run data generation and model training
echo "🤖 Training machine learning models..."
python -c "
import subprocess
import sys

# Generate data and train models
exec(open('generate_models.py').read())
"

echo "✅ Setup complete!"
echo ""
echo "To start the system:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Start the API server: uvicorn main:app --reload"
echo "3. Open the frontend dashboard in your browser"
echo ""
echo "API will be available at: http://localhost:8000"
echo "API docs will be available at: http://localhost:8000/docs"
