#!/usr/bin/env python3
"""
Debug script to diagnose GPT-OSS issues
"""

import requests
import json
import time
import subprocess
import os

def debug_gpt_oss():
    """Comprehensive GPT-OSS debugging"""
    
    print("🔍 GPT-OSS Debug Session")
    print("=" * 40)
    
    base_url = "http://127.0.0.1:11434"
    model_name = "gpt-oss:20b"
    
    # 1. Start Ollama
    print("1. Starting Ollama...")
    subprocess.run(["pkill", "-f", "ollama"], capture_output=True)
    time.sleep(2)
    
    with open(os.devnull, 'w') as devnull:
        process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=devnull,
            stderr=devnull
        )
    
    time.sleep(8)
    print("✅ Ollama started")
    
    # 2. Test connection
    print("2. Testing connection...")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        print(f"✅ Connection: {response.status_code}")
        
        models = response.json().get("models", [])
        print(f"📋 Available models: {len(models)}")
        for model in models:
            print(f"   - {model.get('name', 'Unknown')}")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # 3. Test different prompts
    test_prompts = [
        "Hello",
        "Say the word 'test'",
        "Describe a red car",
        "What do you see?",
        "Looking at a camera feed, I can see: 1 person. Please describe this scene in 1-2 natural sentences."
    ]
    
    print("3. Testing different prompts...")
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n🧪 Test {i+1}: '{prompt}'")
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 50,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        
        try:
            response = requests.post(
                f"{base_url}/api/generate",
                json=payload,
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                generated = result.get("response", "")
                
                print(f"✅ Status: {response.status_code}")
                print(f"📝 Response: '{generated}'")
                print(f"📏 Length: {len(generated)} chars")
                
                # Check for common issues
                if not generated:
                    print("⚠️ ISSUE: Empty response")
                elif len(generated.strip()) == 0:
                    print("⚠️ ISSUE: Only whitespace")
                elif "error" in generated.lower():
                    print("⚠️ ISSUE: Error in response")
                else:
                    print("✅ Response looks good!")
                    
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"📄 Content: {response.text[:200]}")
                
        except requests.exceptions.Timeout:
            print("⏰ Timeout - model taking too long")
        except Exception as e:
            print(f"❌ Request failed: {e}")
        
        time.sleep(2)
    
    # 4. Test with different parameters
    print("\n4. Testing parameter variations...")
    
    param_tests = [
        {"temperature": 0.1, "top_p": 0.9},
        {"temperature": 0.9, "top_p": 0.9},
        {"temperature": 0.7, "top_p": 0.5},
        {"temperature": 0.7, "top_p": 1.0, "top_k": 40},
    ]
    
    test_prompt = "Describe a simple scene with a person."
    
    for i, params in enumerate(param_tests):
        print(f"\n🔧 Param test {i+1}: {params}")
        
        payload = {
            "model": model_name,
            "prompt": test_prompt,
            "stream": False,
            "options": {
                "num_predict": 30,
                **params
            }
        }
        
        try:
            response = requests.post(
                f"{base_url}/api/generate",
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                generated = result.get("response", "")
                print(f"📝 Result: '{generated[:100]}'")
            else:
                print(f"❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # 5. Model info
    print("\n5. Getting model info...")
    try:
        response = requests.post(
            f"{base_url}/api/show",
            json={"name": model_name},
            timeout=10
        )
        
        if response.status_code == 200:
            info = response.json()
            print(f"📊 Model size: {info.get('size', 'Unknown')}")
            print(f"🏗️ Architecture: {info.get('details', {}).get('family', 'Unknown')}")
            print(f"📋 Parameters: {info.get('details', {}).get('parameter_size', 'Unknown')}")
        else:
            print(f"❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Model info error: {e}")
    
    print("\n" + "=" * 40)
    print("🏁 Debug session complete")

if __name__ == "__main__":
    debug_gpt_oss()

