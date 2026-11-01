#!/bin/bash

# Advanced RAG System Startup Script
# This script sets up and runs the RAG system

set -e

echo "🚀 Starting Advanced RAG System..."

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if we're in the correct directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found. Please run this script from the project root directory."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/models
mkdir -p data/performance
mkdir -p logs
mkdir -p tmp

# Copy environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📋 Setting up environment file..."
    cp .env.template .env
    echo "⚠️  Please edit .env file with your API keys"
fi

# Download sample models if needed
echo "⬇️  Setting up models..."
python -c "
from sentence_transformers import SentenceTransformer
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print('✅ Base embedding model ready')
except Exception as e:
    print(f'❌ Error loading model: {e}')
"

# Start the application
echo "🌐 Starting Streamlit application..."
echo "📱 Open your browser and go to: http://localhost:8501"
echo "🔧 Press Ctrl+C to stop the application"
echo ""

# Run Streamlit
streamlit run web/app.py