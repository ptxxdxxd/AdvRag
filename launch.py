#!/usr/bin/env python3
"""
Launch script for Advanced RAG System
Handles setup and launches the Streamlit application
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.8+"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def install_requirements():
    """Install required packages"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        print("📦 Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        print("✅ Packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False


def check_model_availability():
    """Check if required models can be loaded"""
    try:
        print("🤖 Checking model availability...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Base embedding model loaded successfully")
        return True
    except Exception as e:
        print(f"⚠️  Model loading issue: {e}")
        print("💡 This is normal for first-time setup. Models will be downloaded automatically.")
        return True  # Continue anyway


def create_directories():
    """Create necessary directories"""
    directories = [
        "data/models",
        "data/performance", 
        "data/vector_db",
        "logs",
        "tmp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("📁 Directories created successfully")


def launch_streamlit():
    """Launch the Streamlit application"""
    app_file = Path(__file__).parent / "web" / "app.py"
    
    if not app_file.exists():
        print("❌ App file not found:", app_file)
        return False
    
    print("🌐 Starting Streamlit application...")
    print("📱 Open your browser and navigate to: http://localhost:8501")
    print("🔧 Press Ctrl+C to stop the application")
    print("=" * 50)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_file)])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        return False
    
    return True


def main():
    """Main function"""
    print("🚀 Advanced RAG System Launcher")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check model availability
    check_model_availability()
    
    # Setup environment file if it doesn't exist
    env_file = Path(__file__).parent / ".env"
    env_template = Path(__file__).parent / ".env.template"
    
    if not env_file.exists() and env_template.exists():
        print("📋 Creating environment configuration...")
        env_template_content = env_template.read_text()
        env_file.write_text(env_template_content)
        print("⚠️  Please edit .env file with your API keys if using LLM APIs")
    
    # Launch application
    print("\n🎯 All checks passed! Starting application...")
    launch_streamlit()


if __name__ == "__main__":
    main()