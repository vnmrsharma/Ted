#!/usr/bin/env python3
"""
Quick test script to verify Ollama and GPT-OSS model work
"""

import requests
import json
import subprocess
import time
import os

def test_ollama():
    """Test Ollama service and GPT-OSS model"""
    
    print("🧪 Testing Ollama + GPT-OSS Setup")
    print("=" * 40)
    
    # Start Ollama service
    print("1. Starting Ollama service...")
    try:
        # Kill existing
        subprocess.run(["pkill", "-f", "ollama"], capture_output=True)
        time.sleep(2)
        
        # Start fresh
        with open(os.devnull, 'w') as devnull:
            process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=devnull,
                stderr=devnull
            )
        
        time.sleep(5)
        print("✅ Ollama service started")
        
    except Exception as e:
        print(f"❌ Failed to start Ollama: {e}")
        return False
    
    # Test connection
    print("2. Testing connection...")
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Connection successful")
        else:
            print(f"❌ Connection failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    
    # Check for GPT-OSS model
    print("3. Checking for gpt-oss:20b model...")
    try:
        models = response.json().get("models", [])
        gpt_oss_found = any("gpt-oss" in model.get("name", "") for model in models)
        
        if gpt_oss_found:
            print("✅ GPT-OSS model found")
        else:
            print("⚠️ GPT-OSS model not found, attempting to pull...")
            
            # Try to pull the model
            pull_response = requests.post(
                "http://127.0.0.1:11434/api/pull",
                json={"name": "gpt-oss:20b"},
                timeout=300
            )
            
            if pull_response.status_code == 200:
                print("✅ GPT-OSS model downloaded")
            else:
                print(f"❌ Failed to download model: {pull_response.status_code}")
                return False
        
    except Exception as e:
        print(f"❌ Model check error: {e}")
        return False
    
    # Test text generation
    print("4. Testing text generation...")
    try:
        test_payload = {
            "model": "gpt-oss:20b",
            "prompt": "Hello! Please respond with 'AI is working correctly.'",
            "stream": False,
            "options": {
                "num_predict": 10,
                "temperature": 0.1
            }
        }
        
        gen_response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json=test_payload,
            timeout=30
        )
        
        if gen_response.status_code == 200:
            result = gen_response.json()
            generated_text = result.get("response", "").strip()
            print(f"✅ Generation successful: '{generated_text}'")
        else:
            print(f"❌ Generation failed: {gen_response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return False
    
    print("\n🎉 All tests passed! GPT-OSS is ready to use.")
    return True

if __name__ == "__main__":
    success = test_ollama()
    if success:
        print("\n🚀 You can now run: python3 smart-vision.py")
    else:
        print("\n❌ Please fix the issues above before running smart-vision.py")

