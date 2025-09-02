import requests
import json
import subprocess
import time
import os
import sys

class OllamaGPTOSS:
    """Simple GPT-OSS-20B client using Ollama"""
    
    def __init__(self):
        self.base_url = "http://127.0.0.1:11434"
        self.model_name = "gpt-oss:20b"
        self.server_process = None
        self.ready = False
    
    def start_server(self):
        """Start Ollama server or connect to existing one"""
        print("🚀 Connecting to GPT-OSS-20B via Ollama...")
        
        try:
            # First try to connect to existing server
            print("🔍 Checking for existing Ollama server...")
            if self._test_connection():
                self.ready = True
                print("✅ GPT-OSS-20B ready via existing Ollama server!")
                return True
            
            print("⚡ Starting new Ollama server...")
            # Kill any existing ollama processes
            subprocess.run(["pkill", "-f", "ollama"], capture_output=True)
            time.sleep(2)
            
            # Start server in background
            with open(os.devnull, 'w') as devnull:
                self.server_process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=devnull,
                    stderr=devnull,
                    preexec_fn=os.setsid
                )
            
            # Wait for server to start
            print("⏳ Waiting for new Ollama server...")
            time.sleep(8)
            
            # Test connection
            if self._test_connection():
                self.ready = True
                print("✅ GPT-OSS-20B ready via new Ollama server!")
                return True
            else:
                print("❌ Failed to connect to Ollama server")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start Ollama: {e}")
            return False
    
    def _test_connection(self):
        """Test if Ollama server is responding"""
        try:
            # Check if server is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            
            # Test with simple prompt
            test_prompt = "Hello"
            result = self._generate(test_prompt, max_tokens=10)
            return result is not None
            
        except Exception:
            return False
    
    def _generate(self, prompt, max_tokens=100):
        """Generate response using Ollama API"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "").strip()
                
                # GPT-OSS sometimes puts content in "thinking" field
                if not generated_text:
                    thinking_text = result.get("thinking", "").strip()
                    if thinking_text:
                        # Extract useful content from thinking
                        if ". " in thinking_text:
                            # Get the last sentence that seems like a response
                            sentences = thinking_text.split(". ")
                            for sentence in reversed(sentences[-3:]):  # Check last 3 sentences
                                if len(sentence) > 10 and not sentence.startswith("The user"):
                                    generated_text = sentence.strip()
                                    break
                        if not generated_text:
                            generated_text = thinking_text[:100]  # Fallback to first part
                
                return generated_text if len(generated_text) > 0 else None
            else:
                print(f"❌ API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return None
    
    def cleanup(self):
        """Cleanup Ollama server"""
        try:
            if self.server_process:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            subprocess.run(["pkill", "-f", "ollama"], capture_output=True)
        except Exception:
            pass
        self.ready = False

def generate_response(client, prompt, max_tokens=100):
    """Generate response using Ollama client"""
    if not client.ready:
        print("❌ GPT-OSS not ready. Please start the server first.")
        return None
    
    print(f"🧠 Generating response for: '{prompt[:50]}...'")
    return client._generate(prompt, max_tokens)

def interactive_mode():
    """Interactive chat mode using Ollama"""
    print("🚀 GPT-OSS-20B Interactive Chat via Ollama")
    print("=" * 50)
    print("Commands:")
    print("  /quit - Exit")
    print("  /help - Show this help")
    print("=" * 50)
    
    # Initialize Ollama client
    client = OllamaGPTOSS()
    
    # Start Ollama server
    if not client.start_server():
        print("❌ Failed to start GPT-OSS. Exiting.")
        return
    
    print("✅ GPT-OSS-20B ready! Start chatting...")
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == '/help':
                print("Commands: /quit to exit, or just type your message")
                continue
            elif not user_input:
                continue
            
            # Generate response
            response = generate_response(client, user_input, max_tokens=150)
            
            if response:
                print(f"\n🤖 GPT-OSS: {response}")
            else:
                print("❌ Failed to generate response. Please try again.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Cleanup
    client.cleanup()

def check_memory():
    """Check available system memory"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"💾 Total RAM: {memory.total / (1024**3):.1f} GB")
        print(f"💾 Available RAM: {memory.available / (1024**3):.1f} GB")
        print(f"💾 Used RAM: {memory.used / (1024**3):.1f} GB")
        return memory.available / (1024**3)
    except ImportError:
        print("💾 Memory info unavailable (install psutil)")
        return 16  # Assume reasonable amount
    except Exception:
        print("💾 Memory info unavailable")
        return 16

def test_model():
    """Test GPT-OSS-20B via Ollama"""
    print("🧪 Testing GPT-OSS-20B via Ollama...")
    
    # Check memory
    available_gb = check_memory()
    print(f"💾 Available memory: {available_gb:.1f} GB")
    
    # Initialize client
    client = OllamaGPTOSS()
    
    # Start server
    if not client.start_server():
        print("❌ Failed to start Ollama server")
        return False
    
    # Test prompt
    test_prompt = "Hello! Can you explain what you are briefly?"
    print(f"\n📝 Testing with: {test_prompt}")
    
    response = generate_response(client, test_prompt, max_tokens=80)
    
    if response:
        print(f"🤖 Response: {response}")
        print("\n✅ Test completed successfully!")
        success = True
    else:
        print("❌ Test failed")
        success = False
    
    client.cleanup()
    return success

def quick_chat():
    """Quick chat mode using Ollama"""
    print("🚀 GPT-OSS-20B Quick Chat via Ollama")
    print("=" * 50)
    print("Commands: /quit to exit, /status to check server")
    print("=" * 50)
    
    # Initialize client
    client = OllamaGPTOSS()
    
    # Start server
    if not client.start_server():
        print("❌ Failed to start Ollama. Make sure it's installed.")
        return
    
    print("✅ GPT-OSS ready! Start chatting...")
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['/quit', 'quit', 'exit']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == '/status':
                if client.ready:
                    print("✅ GPT-OSS server is running")
                else:
                    print("❌ GPT-OSS server not responding")
                continue
            elif user_input.lower() == '/memory':
                check_memory()
                continue
            elif not user_input:
                continue
            
            response = generate_response(client, user_input, max_tokens=120)
            if response:
                print(f"🤖 GPT-OSS: {response}")
            else:
                print("❌ Generation failed. Try rephrasing your question.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Cleanup
    client.cleanup()

if __name__ == "__main__":
    import sys
    
    print("🚀 Simple GPT-OSS-20B via Ollama")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--chat":
            quick_chat()
        elif sys.argv[1] == "--interactive":
            interactive_mode()
        elif sys.argv[1] == "--memory":
            check_memory()
        elif sys.argv[1] == "--test":
            test_model()
        else:
            print("Usage: python3 gpt-oos.py [--chat|--interactive|--memory|--test]")
            print("  --chat        Quick chat mode")
            print("  --interactive Full interactive mode")
            print("  --memory      Check system memory")
            print("  --test        Test GPT-OSS connection")
    else:
        # Default: run quick chat
        print("🎯 Starting quick chat mode (use --help for options)")
        quick_chat()