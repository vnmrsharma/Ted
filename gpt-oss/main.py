"""
Character Persona Generator - Main Application
A comprehensive system for generating realistic character personas from movie scripts
using GPT-OSS-20B model with an interactive Streamlit interface.
"""

import streamlit as st
import json
import os
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import tempfile
import shutil

from script_processor import ScriptProcessor
from persona_generator import PersonaGenerator, PersonaConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Character Persona Generator",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

class CharacterPersonaApp:
    """Main application class for the Character Persona Generator."""
    
    def __init__(self):
        self.script_processor = ScriptProcessor()
        self.persona_generator = None
        self.processed_data = None
        self.current_character = None
        self.chat_history = []
        
        # Create necessary directories
        self.data_dir = Path("data")
        self.models_dir = Path("character_models")
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
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
    
    def process_script(self, script_text: str, script_file = None) -> Tuple[str, str, str]:
        """Process uploaded script and extract character information."""
        try:
            if script_file is not None:
                # Handle file upload
                with open(script_file.name, 'r', encoding='utf-8') as f:
                    script_text = f.read()
            
            if not script_text.strip():
                return "Please provide script text or upload a file.", "", ""
            
            logger.info("Starting script processing...")
            
            # Process the script
            self.processed_data = self.script_processor.process_script(script_text)
            
            # Save processed data
            processed_file = self.data_dir / "processed_script.json"
            self.script_processor.save_processed_data(self.processed_data, str(processed_file))
            
            # Generate summary
            summary = self._generate_processing_summary()
            
            # Get character suggestions
            suggestions = self._get_character_suggestions()
            
            logger.info("Script processing completed successfully")
            return summary, suggestions, "Script processed successfully! You can now select a character to chat with."
            
        except Exception as e:
            logger.error(f"Error processing script: {str(e)}")
            return f"Error processing script: {str(e)}", "", ""
    
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
    
    def _get_character_suggestions(self) -> str:
        """Get character suggestions for persona generation."""
        if not self.processed_data:
            return "No data available."
        
        characters = self.processed_data.get('characters', {})
        suggestions = []
        
        for name, char_data in characters.items():
            score = char_data.get('dialogue_count', 0) * 10
            score += len(char_data.get('personality_traits', [])) * 5
            score += len(char_data.get('key_quotes', [])) * 3
            
            suggestions.append({
                'name': name,
                'score': score,
                'dialogue_count': char_data.get('dialogue_count', 0),
                'personality_traits': char_data.get('personality_traits', [])
            })
        
        suggestions.sort(key=lambda x: x['score'], reverse=True)
        
        result = "**Recommended Characters for Persona Generation:**\n\n"
        for i, char in enumerate(suggestions[:10], 1):
            traits = ', '.join(char['personality_traits'][:3]) if char['personality_traits'] else 'No traits identified'
            result += f"{i}. **{char['name']}** (Score: {char['score']})\n"
            result += f"   - Dialogue Lines: {char['dialogue_count']}\n"
            result += f"   - Personality: {traits}\n\n"
        
        return result
    
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
        
        # Reset chat history
        self.chat_history = []
        
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
    
    def fine_tune_character(self, character_name: str) -> str:
        """Fine-tune the model for a specific character."""
        if not self.current_character or not self.processed_data:
            return "Please select a character first."
        
        if not self.persona_generator:
            return "Please initialize the model first."
        
        try:
            character_data = self.processed_data['characters'][character_name]
            
            # Fine-tune the model
            self.persona_generator.fine_tune_character(
                character_data, 
                self.processed_data,
                str(self.models_dir)
            )
            
            return f"Fine-tuning completed for {character_name}. The model has been saved."
            
        except Exception as e:
            logger.error(f"Error in fine-tuning: {str(e)}")
            return f"Error in fine-tuning: {str(e)}"
    
    def get_character_list(self) -> List[str]:
        """Get list of available characters."""
        if not self.processed_data:
            return []
        
        characters = self.processed_data.get('characters', {})
        return list(characters.keys())

def main():
    """Main Streamlit application."""
    st.title("🎭 Character Persona Generator")
    st.markdown("Generate realistic character personas from movie scripts using GPT-OSS-20B model.")
    
    # Initialize session state
    if 'app' not in st.session_state:
        st.session_state.app = CharacterPersonaApp()
    
    app = st.session_state.app
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Script Processing",
        "Model Setup", 
        "Character Selection",
        "Chat with Character"
    ])
    
    if page == "Script Processing":
        st.header("📝 Step 1: Upload and Process Script")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Script Input")
            script_text = st.text_area(
                "Script Text",
                placeholder="Paste your movie script here...",
                height=300,
                help="Enter your movie script text directly"
            )
            
            uploaded_file = st.file_uploader(
                "Or upload script file",
                type=['txt', 'md'],
                help="Upload a .txt or .md file containing your script"
            )
        
        with col2:
            st.subheader("Processing")
            if st.button("🚀 Process Script", type="primary", use_container_width=True):
                with st.spinner("Processing script..."):
                    summary, suggestions, status = app.process_script(script_text, uploaded_file)
                    
                    st.success(status)
                    st.markdown(summary)
                    st.markdown(suggestions)
        
        # Show processing status
        if app.processed_data:
            st.success("✅ Script processed successfully!")
            st.info(f"Found {len(app.get_character_list())} characters")
    
    elif page == "Model Setup":
        st.header("🤖 Step 2: Initialize GPT-OSS-20B Model")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Model Configuration")
            st.info("GPT-OSS-20B is already quantized with Mxfp4Config for optimal performance")
            use_quantization = st.checkbox(
                "Use 4-bit Quantization",
                value=True,
                disabled=True,
                help="GPT-OSS-20B is already quantized - this option is always enabled"
            )
            use_lora = st.checkbox(
                "Use LoRA for Fine-tuning",
                value=True,
                help="Enables efficient fine-tuning for character-specific models"
            )
        
        with col2:
            st.subheader("Initialization")
            if st.button("🚀 Initialize Model", type="primary", use_container_width=True):
                with st.spinner("Initializing GPT-OSS-20B model..."):
                    status = app.initialize_model(use_quantization, use_lora)
                    st.info(status)
        
        # Show model status
        if app.persona_generator:
            st.success("✅ Model initialized successfully!")
        else:
            st.warning("⚠️ Model not initialized yet")
    
    elif page == "Character Selection":
        st.header("👤 Step 3: Select Character")
        
        if not app.processed_data:
            st.warning("Please process a script first in the Script Processing page.")
            return
        
        characters = app.get_character_list()
        if not characters:
            st.error("No characters found. Please process a script first.")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Character Selection")
            selected_character = st.selectbox(
                "Choose a character",
                characters,
                help="Select a character to generate a persona for"
            )
            
            if st.button("🎭 Select Character", type="primary", use_container_width=True):
                with st.spinner("Loading character profile..."):
                    status, profile = app.select_character(selected_character)
                    st.success(status)
        
        with col2:
            st.subheader("Character Profile")
            if app.current_character:
                character_data = app.processed_data['characters'][app.current_character]
                profile = app._generate_character_profile(character_data)
                st.markdown(profile)
            else:
                st.info("Select a character to view their profile")
    
    elif page == "Chat with Character":
        st.header("💬 Step 4: Chat with Character Persona")
        
        if not app.current_character:
            st.warning("Please select a character first in the Character Selection page.")
            return
        
        if not app.persona_generator:
            st.warning("Please initialize the model first in the Model Setup page.")
            return
        
        st.subheader(f"Chatting with {app.current_character}")
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input(f"Type a message to {app.current_character}..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    try:
                        character_data = app.processed_data['characters'][app.current_character]
                        response = app.persona_generator.chat_with_character(
                            character_data, 
                            app.processed_data, 
                            prompt
                        )
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Fine-tuning section
        st.subheader("🔧 Fine-tuning")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("Fine-tune the model for better character-specific responses")
        
        with col2:
            if st.button("🎯 Fine-tune Model", type="secondary"):
                with st.spinner("Fine-tuning model..."):
                    status = app.fine_tune_character(app.current_character)
                    st.success(status)
    
    # Footer
    st.markdown("---")
    st.markdown("**Character Persona Generator** - Powered by GPT-OSS-20B")

if __name__ == "__main__":
    main()