"""
Character Persona Generator - Simple Demo
A command-line demo that shows the core functionality without Gradio issues
"""

import json
import logging
from pathlib import Path
from script_processor import ScriptProcessor
from persona_generator import PersonaGenerator, PersonaConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main demo function."""
    print("🎭 Character Persona Generator - Simple Demo")
    print("=" * 50)
    
    # Initialize components
    processor = ScriptProcessor()
    
    # Load the sample script
    script_file = Path("script.txt")
    if not script_file.exists():
        print("❌ script.txt not found. Please ensure the script file exists.")
        return
    
    print("📖 Loading and processing script...")
    with open(script_file, 'r', encoding='utf-8') as f:
        script_text = f.read()
    
    # Process the script
    print("🔍 Analyzing characters and relationships...")
    processed_data = processor.process_script(script_text)
    
    # Display results
    print("\n📊 Script Analysis Results:")
    print("-" * 30)
    
    metadata = processed_data.get('metadata', {})
    print(f"Total Characters: {metadata.get('total_characters', 0)}")
    print(f"Total Scenes: {metadata.get('total_scenes', 0)}")
    print(f"Script Length: {metadata.get('script_length', 0):,} characters")
    
    # Show top characters
    characters = processed_data.get('characters', {})
    char_dialogue = [(name, char.get('dialogue_count', 0)) for name, char in characters.items()]
    char_dialogue.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🎭 Top Characters by Dialogue:")
    for i, (name, count) in enumerate(char_dialogue[:5], 1):
        print(f"{i}. {name}: {count} lines")
    
    # Show character with most personality traits
    char_traits = [(name, len(char.get('personality_traits', []))) for name, char in characters.items()]
    char_traits.sort(key=lambda x: x[1], reverse=True)
    
    if char_traits and char_traits[0][1] > 0:
        best_char = char_traits[0][0]
        char_data = characters[best_char]
        
        print(f"\n🌟 Best Character for Persona: {best_char}")
        print(f"Personality Traits: {', '.join(char_data.get('personality_traits', []))}")
        print(f"Dialogue Count: {char_data.get('dialogue_count', 0)}")
        
        if char_data.get('key_quotes'):
            print(f"Sample Quote: \"{char_data['key_quotes'][0][:100]}...\"")
    
    # Demo persona generation (without actual model loading)
    print(f"\n🤖 Persona Generation Demo:")
    print("-" * 30)
    
    if char_traits and char_traits[0][1] > 0:
        demo_char = char_traits[0][0]
        char_data = characters[demo_char]
        
        print(f"Character: {demo_char}")
        print(f"Personality: {', '.join(char_data.get('personality_traits', []))}")
        print(f"Dialogue Lines: {char_data.get('dialogue_count', 0)}")
        
        # Show how persona generation would work
        print(f"\n📝 Example Persona Context:")
        context = f"""
Character: {char_data['name']}
Personality Traits: {', '.join(char_data.get('personality_traits', []))}
Dialogue Count: {char_data.get('dialogue_count', 0)}
Key Quotes: {char_data.get('key_quotes', [])[:2]}
"""
        print(context)
        
        print(f"\n💬 Example Chat Interaction:")
        print(f"User: Hello {demo_char}, tell me about yourself.")
        print(f"{demo_char}: [GPT-OSS-20B would generate a response based on the character's personality and dialogue patterns]")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"\n📋 To use the full application with GPT-OSS-20B:")
    print(f"1. Ensure you have internet connection")
    print(f"2. Run: python main.py")
    print(f"3. Open browser to the provided URL")
    print(f"4. Process script and initialize the model")
    print(f"5. Select a character and start chatting!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in demo: {str(e)}")
        print(f"❌ Error: {str(e)}")
        print("Please check your setup and try again.")
