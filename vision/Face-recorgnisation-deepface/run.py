#!/usr/bin/env python3
"""
Simple launch script for Face Recognition System
Handles common setup issues and provides helpful feedback
"""

import sys
import os
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'deepface', 'gradio', 'opencv-python', 'numpy', 'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'opencv-python':
                import cv2
            else:
                __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("💡 Install with: pip install -r requirements.txt")
        return False
    
    return True

def install_dependencies():
    """Offer to install missing dependencies"""
    print("\n🔧 Would you like to install missing dependencies?")
    response = input("Enter 'y' to install, or any other key to exit: ").strip().lower()
    
    if response == 'y':
        print("📥 Installing dependencies...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    else:
        print("❌ Cannot run without required dependencies")
        return False

def main():
    """Main launch function"""
    print("🎯 Face Recognition System - Launcher")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        if not install_dependencies():
            sys.exit(1)
    
    print("\n🚀 Launching Face Recognition System...")
    print("💡 The system will open in your web browser")
    print("💡 Press Ctrl+C to stop the application")
    print("=" * 40)
    
    try:
        # Import and run the main application
        from main import create_gradio_interface
        
        interface = create_gradio_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try running: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        print("💡 Check the error message above for details")
        sys.exit(1)

if __name__ == "__main__":
    main()
