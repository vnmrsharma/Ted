"""
Character Persona Generator - Simplified Main Application
A simplified version that avoids Gradio compatibility issues
"""

import gradio as gr
import json
import os
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from script_processor import ScriptProcessor
from persona_generator import PersonaGenerator, PersonaConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CharacterPersonaApp:
    """Main application class for the Character Persona Generator."""
    
    def __init__(self):
        self.script_processor = ScriptProcessor()
        self.persona_generator = None
        self.processed_data = None
        self.current_character = None
        
        logger.info("Character Persona App initialized")
    
    def initialize_model(self, use_quantization: bool = True, use_lora: bool = True):
        """Initialize the GPT-OSS-20B model."""
        try:
            config = PersonaConfig(
                use_quantization=use_quantization,
                use_lora=use_lora
            )
            self.persona_generator = PersonaGenerator(config)
            logger.info("Model initialization started...")
            return "Model initialization started. This may take a few minutes..."
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            return f"Error initializing model: {str(e)}"
    
    def process_script(self, script_text: str) -> Tuple[str, str]:
        """Process uploaded script and extract character information."""
        try:
            if not script_text.strip():
                return "Please provide script text.", ""
            
            logger.info("Starting script processing...")
            
            # Process the script
            self.processed_data = self.script_processor.process_script(script_text)
            
            # Generate summary
            summary = self._generate_processing_summary()
            
            logger.info("Script processing completed successfully")
            return summary, "Script processed successfully! You can now select a character to chat with."
            
        except Exception as e:
            logger.error(f"Error processing script: {str(e)}")
            return f"Error processing script: {str(e)}", ""
    
    def _generate_processing_summary(self) -> str:
        """Generate a summary of the processed script."""
        if not self.processed_data:
            return "No data processed yet."
        
        metadata = self.processed_data.get('metadata', {})
        characters = self.processed_data.get('characters', {})
        
        summary = f"""
## Script Processing Summary

**Basic Statistics:**
- Total Characters: {metadata.get('total_characters', 0)}
- Total Scenes: {metadata.get('total_scenes', 0)}
- Script Length: {metadata.get('script_length', 0):,} characters

**Character Analysis:**
"""
        
        # Top characters by dialogue
        char_dialogue = [(name, char.get('dialogue_count', 0)) for name, char in characters.items()]
        char_dialogue.sort(key=lambda x: x[1], reverse=True)
        
        summary += "\n**Top Characters by Dialogue Count:**\n"
        for name, count in char_dialogue[:5]:
            summary += f"- {name}: {count} lines\n"
        
        # Characters with personality traits
        chars_with_traits = [name for name, char in characters.items() if char.get('personality_traits')]
        summary += f"\n**Characters with Identified Personality Traits:** {len(chars_with_traits)}\n"
        
        return summary
    
    def get_character_list(self) -> List[str]:
        """Get list of available characters."""
        if not self.processed_data:
            return []
        
        characters = self.processed_data.get('characters', {})
        return list(characters.keys())
    
    def select_character(self, character_name: str) -> Tuple[str, str]:
        """Select a character for persona generation."""
        if not self.processed_data:
            return "Please process a script first.", ""
        
        characters = self.processed_data.get('characters', {})
        if character_name not in characters:
            return f"Character '{character_name}' not found in processed data.", ""
        
        self.current_character = character_name
        character_data = characters[character_name]
        
        # Generate character profile
        profile = self._generate_character_profile(character_data)
        
        return f"Selected character: {character_name}", profile
    
    def _generate_character_profile(self, character_data: Dict) -> str:
        """Generate a detailed character profile."""
        profile = f"""
## Character Profile: {character_data['name']}

**Basic Information:**
- Dialogue Count: {character_data.get('dialogue_count', 0)}
- Total Words: {character_data.get('total_words', 0)}
- Scenes Appeared: {len(character_data.get('scenes_appeared', []))}

**Personality Traits:**
{', '.join(character_data.get('personality_traits', ['No traits identified']))}

**Key Quotes:**
"""
        
        quotes = character_data.get('key_quotes', [])
        if quotes:
            for i, quote in enumerate(quotes[:3], 1):
                profile += f"{i}. \"{quote[:100]}{'...' if len(quote) > 100 else ''}\"\n"
        else:
            profile += "No key quotes available.\n"
        
        profile += "\n**Relationships:**\n"
        relationships = character_data.get('relationships', {})
        if relationships:
            for other_char, strength in list(relationships.items())[:5]:
                profile += f"- {other_char} (interaction strength: {strength})\n"
        else:
            profile += "No relationships identified.\n"
        
        return profile
    
    def chat_with_character(self, user_input: str, history: List) -> Tuple[str, List]:
        """Chat with the selected character."""
        if not self.current_character or not self.processed_data:
            return "Please select a character first.", history
        
        if not self.persona_generator:
            return "Please initialize the model first.", history
        
        try:
            character_data = self.processed_data['characters'][self.current_character]
            
            # Generate response
            response = self.persona_generator.chat_with_character(
                character_data, 
                self.processed_data, 
                user_input
            )
            
            # Update chat history
            history.append([user_input, response])
            
            return "", history
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return f"Error generating response: {str(e)}", history

def create_interface():
    """Create the simplified Gradio interface."""
    app = CharacterPersonaApp()
    
    with gr.Blocks(title="Character Persona Generator", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🎭 Character Persona Generator
        
        Generate realistic character personas from movie scripts using GPT-OSS-20B model.
        Upload a script, process it, and chat with any character!
        """)
        
        with gr.Tab("Script Processing"):
            gr.Markdown("## Step 1: Upload and Process Script")
            
            script_input = gr.Textbox(
                label="Script Text",
                placeholder="Paste your movie script here...",
                lines=10,
                max_lines=20
            )
            process_btn = gr.Button("Process Script", variant="primary")
            
            processing_status = gr.Textbox(
                label="Processing Status",
                interactive=False,
                lines=3
            )
            summary_output = gr.Markdown(label="Processing Summary")
        
        with gr.Tab("Model Setup"):
            gr.Markdown("## Step 2: Initialize GPT-OSS-20B Model")
            
            use_quantization = gr.Checkbox(
                label="Use 4-bit Quantization (Recommended for memory efficiency)",
                value=True
            )
            use_lora = gr.Checkbox(
                label="Use LoRA for Fine-tuning",
                value=True
            )
            
            init_btn = gr.Button("Initialize Model", variant="primary")
            model_status = gr.Textbox(
                label="Model Status",
                interactive=False,
                lines=3
            )
        
        with gr.Tab("Character Selection"):
            gr.Markdown("## Step 3: Select Character")
            
            character_dropdown = gr.Dropdown(
                label="Select Character",
                choices=[],
                interactive=True
            )
            select_btn = gr.Button("Select Character", variant="primary")
            
            character_status = gr.Textbox(
                label="Selection Status",
                interactive=False
            )
            character_profile = gr.Markdown(label="Character Profile")
        
        with gr.Tab("Chat with Character"):
            gr.Markdown("## Step 4: Chat with Character Persona")
            
            chatbot = gr.Chatbot(
                label="Chat with Character",
                height=400,
                show_label=True
            )
            
            user_input = gr.Textbox(
                label="Your Message",
                placeholder="Type your message here...",
                lines=2
            )
            send_btn = gr.Button("Send", variant="primary")
        
        # Event handlers
        def update_character_choices():
            choices = app.get_character_list()
            return gr.Dropdown.update(choices=choices)
        
        process_btn.click(
            app.process_script,
            inputs=[script_input],
            outputs=[processing_status, summary_output]
        ).then(
            update_character_choices,
            outputs=character_dropdown
        )
        
        init_btn.click(
            app.initialize_model,
            inputs=[use_quantization, use_lora],
            outputs=model_status
        )
        
        select_btn.click(
            app.select_character,
            inputs=character_dropdown,
            outputs=[character_status, character_profile]
        )
        
        send_btn.click(
            app.chat_with_character,
            inputs=[user_input, chatbot],
            outputs=[user_input, chatbot]
        )
        
        user_input.submit(
            app.chat_with_character,
            inputs=[user_input, chatbot],
            outputs=[user_input, chatbot]
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        debug=False
    )
