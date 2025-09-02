#!/usr/bin/env python3
"""
Deep debug script for GPT-OSS 20B
Find exactly what works and what doesn't
"""

import requests
import json
import time
import subprocess
import os

def deep_debug_gpt_oss():
    """Comprehensive debugging of GPT-OSS"""
    
    print("🔬 Deep GPT-OSS 20B Debug Session")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:11434"
    model_name = "gpt-oss:20b"
    
    # Start fresh
    print("1. Starting clean Ollama instance...")
    subprocess.run(["pkill", "-f", "ollama"], capture_output=True)
    time.sleep(3)
    
    with open(os.devnull, 'w') as devnull:
        process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=devnull,
            stderr=devnull
        )
    
    time.sleep(10)
    
    # Test different prompt styles
    print("2. Testing various prompt styles...")
    
    test_cases = [
        # Simple greetings (we know these work)
        ("Hello", "Basic greeting"),
        ("Hi there", "Simple greeting"),
        ("Good morning", "Time greeting"),
        
        # Very short commands
        ("Yes", "Single word"),
        ("No", "Single word negative"),
        ("OK", "Acknowledgment"),
        
        # Simple statements
        ("The sky is blue", "Simple fact"),
        ("I see a cat", "Simple observation"),
        ("There is a dog", "Existence statement"),
        
        # Questions
        ("What is this?", "Simple question"),
        ("How are you?", "Personal question"),
        ("Where am I?", "Location question"),
        
        # Scene descriptions (what we need)
        ("Describe: person", "Object description 1"),
        ("Tell me about: person", "Object description 2"),
        ("I see: person", "Object observation"),
        ("Scene: person", "Scene format 1"),
        ("Objects: person", "Object list format"),
        
        # Different formats
        ("person sitting", "Action description"),
        ("A person is here", "Present tense"),
        ("There is a person", "Existence format"),
        
        # Chat/conversation style
        ("User: I see a person\nAssistant:", "Chat format"),
        ("Q: What do you see?\nA:", "Q&A format"),
        
        # Vision-style prompts
        ("Camera shows: person", "Camera format"),
        ("Video feed: person", "Video format"),
        ("Detection: person", "Detection format"),
    ]
    
    working_prompts = []
    failing_prompts = []
    
    for prompt, description in test_cases:
        print(f"\n🧪 Testing: {description}")
        print(f"   Prompt: '{prompt}'")
        
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 30,
                    "temperature": 0.7
                }
            }
            
            response = requests.post(
                f"{base_url}/api/generate",
                json=payload,
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                generated = result.get("response", "")
                
                print(f"   ✅ Response: '{generated}'")
                print(f"   📏 Length: {len(generated)} chars")
                
                if len(generated.strip()) > 0:
                    working_prompts.append((prompt, description, generated))
                    print("   ✅ SUCCESS!")
                else:
                    failing_prompts.append((prompt, description, "Empty"))
                    print("   ❌ EMPTY RESPONSE")
                    
            else:
                failing_prompts.append((prompt, description, f"HTTP {response.status_code}"))
                print(f"   ❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            failing_prompts.append((prompt, description, str(e)))
            print(f"   ❌ Exception: {e}")
        
        time.sleep(1)  # Don't overwhelm the model
    
    # Analysis
    print("\n" + "=" * 50)
    print("📊 ANALYSIS RESULTS")
    print("=" * 50)
    
    print(f"\n✅ WORKING PROMPTS ({len(working_prompts)}):")
    for prompt, desc, response in working_prompts:
        print(f"   '{prompt}' → '{response[:50]}...'")
    
    print(f"\n❌ FAILING PROMPTS ({len(failing_prompts)}):")
    for prompt, desc, error in failing_prompts:
        print(f"   '{prompt}' → {error}")
    
    # Pattern analysis
    print(f"\n🔍 PATTERN ANALYSIS:")
    
    # Length analysis
    working_lengths = [len(p[0]) for p in working_prompts]
    failing_lengths = [len(p[0]) for p in failing_prompts]
    
    if working_lengths:
        print(f"   Working prompt lengths: {min(working_lengths)}-{max(working_lengths)} chars")
    if failing_lengths:
        print(f"   Failing prompt lengths: {min(failing_lengths)}-{max(failing_lengths)} chars")
    
    # Word analysis
    working_words = []
    failing_words = []
    
    for prompt, _, _ in working_prompts:
        working_words.extend(prompt.lower().split())
    
    for prompt, _, _ in failing_prompts:
        failing_words.extend(prompt.lower().split())
    
    # Common words in working prompts
    from collections import Counter
    working_word_counts = Counter(working_words)
    failing_word_counts = Counter(failing_words)
    
    print(f"\n   Most common words in WORKING prompts:")
    for word, count in working_word_counts.most_common(5):
        print(f"     '{word}': {count} times")
    
    print(f"\n   Most common words in FAILING prompts:")
    for word, count in failing_word_counts.most_common(5):
        print(f"     '{word}': {count} times")
    
    # Test parameter variations on working prompt
    if working_prompts:
        print(f"\n🔧 PARAMETER TESTING with working prompt:")
        best_prompt = working_prompts[0][0]  # Use first working prompt
        print(f"   Using: '{best_prompt}'")
        
        param_tests = [
            {"temperature": 0.1, "top_p": 0.9},
            {"temperature": 0.5, "top_p": 0.9},
            {"temperature": 0.8, "top_p": 0.9},
            {"temperature": 1.0, "top_p": 0.9},
            {"temperature": 0.7, "top_p": 0.5},
            {"temperature": 0.7, "top_p": 1.0},
            {"temperature": 0.7, "top_k": 10},
            {"temperature": 0.7, "top_k": 50},
        ]
        
        for params in param_tests:
            try:
                payload = {
                    "model": model_name,
                    "prompt": best_prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 20,
                        **params
                    }
                }
                
                response = requests.post(
                    f"{base_url}/api/generate",
                    json=payload,
                    timeout=15
                )
                
                if response.status_code == 200:
                    result = response.json()
                    generated = result.get("response", "")
                    print(f"   {params} → '{generated[:40]}'")
                else:
                    print(f"   {params} → HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"   {params} → Error: {e}")
            
            time.sleep(1)
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if working_prompts:
        print("   ✅ Use these working prompt patterns:")
        unique_patterns = set()
        for prompt, desc, _ in working_prompts:
            if len(prompt.split()) <= 3:  # Short prompts
                unique_patterns.add(prompt)
        
        for pattern in list(unique_patterns)[:5]:
            print(f"     '{pattern}'")
    
    if failing_prompts:
        print("   ❌ Avoid these patterns:")
        problem_patterns = set()
        for prompt, desc, _ in failing_prompts:
            if "describe" in prompt.lower() or "scene" in prompt.lower():
                problem_patterns.add(prompt)
        
        for pattern in list(problem_patterns)[:3]:
            print(f"     '{pattern}'")
    
    print("\n" + "=" * 50)
    print("🏁 Deep debug complete!")

if __name__ == "__main__":
    deep_debug_gpt_oss()

