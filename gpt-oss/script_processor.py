"""
Script Processing Pipeline for Character Persona Generation
This module handles the intensive data cleaning and preprocessing of movie scripts
to extract character information, relationships, and context.
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import spacy

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install spaCy English model: python -m spacy download en_core_web_sm")
    nlp = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Character:
    """Represents a character with their attributes and relationships."""
    name: str
    full_name: str = ""
    description: str = ""
    dialogue_count: int = 0
    total_words: int = 0
    relationships: Dict[str, List[str]] = None
    personality_traits: List[str] = None
    key_quotes: List[str] = None
    scenes_appeared: List[str] = None
    
    def __post_init__(self):
        if self.relationships is None:
            self.relationships = {}
        if self.personality_traits is None:
            self.personality_traits = []
        if self.key_quotes is None:
            self.key_quotes = []
        if self.scenes_appeared is None:
            self.scenes_appeared = []

@dataclass
class Scene:
    """Represents a scene with its context and characters."""
    scene_id: str
    location: str
    time_of_day: str = ""
    characters_present: List[str] = None
    dialogue: List[Dict[str, str]] = None
    action_lines: List[str] = None
    context: str = ""
    
    def __post_init__(self):
        if self.characters_present is None:
            self.characters_present = []
        if self.dialogue is None:
            self.dialogue = []
        if self.action_lines is None:
            self.action_lines = []

class ScriptProcessor:
    """Main class for processing movie scripts and extracting character information."""
    
    def __init__(self):
        self.characters: Dict[str, Character] = {}
        self.scenes: List[Scene] = []
        self.character_relationships: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.stop_words = set(stopwords.words('english'))
        
        # Common script patterns
        self.scene_pattern = re.compile(r'^(INT\.|EXT\.|ESTAB\.)\s+(.+?)(?:\s+-\s+(.+))?$', re.MULTILINE)
        self.character_pattern = re.compile(r'^([A-Z][A-Z\s]+?)(?:\s*\([^)]*\))?\s*$', re.MULTILINE)
        self.dialogue_pattern = re.compile(r'^([A-Z][A-Z\s]+?)(?:\s*\([^)]*\))?\s*\n(.*?)(?=\n[A-Z][A-Z\s]+|\n\n|\Z)', re.MULTILINE | re.DOTALL)
        self.action_pattern = re.compile(r'^([A-Z][A-Z\s]+?)(?:\s*\([^)]*\))?\s*\n(.*?)(?=\n[A-Z][A-Z\s]+|\n\n|\Z)', re.MULTILINE | re.DOTALL)
        
    def clean_script(self, script_text: str) -> str:
        """Clean and normalize the script text."""
        logger.info("Starting script cleaning...")
        
        # Remove page numbers and formatting artifacts
        script_text = re.sub(r'^\s*\d+\s*$', '', script_text, flags=re.MULTILINE)
        script_text = re.sub(r'^\s*CONTINUED:\s*$', '', script_text, flags=re.MULTILINE)
        script_text = re.sub(r'^\s*\(CONTINUED\)\s*$', '', script_text, flags=re.MULTILINE)
        
        # Normalize whitespace
        script_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', script_text)
        script_text = re.sub(r'[ \t]+', ' ', script_text)
        
        # Remove excessive spacing
        script_text = re.sub(r'^\s+', '', script_text, flags=re.MULTILINE)
        
        logger.info("Script cleaning completed")
        return script_text.strip()
    
    def extract_scenes(self, script_text: str) -> List[Scene]:
        """Extract scenes from the script."""
        logger.info("Extracting scenes...")
        scenes = []
        scene_id = 1
        
        lines = script_text.split('\n')
        current_scene = None
        current_location = ""
        current_time = ""
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check for scene headers
            scene_match = self.scene_pattern.match(line)
            if scene_match:
                if current_scene:
                    scenes.append(current_scene)
                
                location = scene_match.group(2).strip()
                time_info = scene_match.group(3).strip() if scene_match.group(3) else ""
                
                current_scene = Scene(
                    scene_id=f"scene_{scene_id}",
                    location=location,
                    time_of_day=time_info,
                    context=f"Scene {scene_id}: {location} - {time_info}"
                )
                scene_id += 1
                current_location = location
                current_time = time_info
                
            elif current_scene and line:
                # Check if it's character dialogue or action
                if re.match(r'^[A-Z][A-Z\s]+(?:\s*\([^)]*\))?\s*$', line):
                    # This is a character name
                    character_name = re.sub(r'\s*\([^)]*\)', '', line).strip()
                    if character_name not in ['NARRATOR', 'V.O.', 'O.S.', 'CONT\'D']:
                        if character_name not in current_scene.characters_present:
                            current_scene.characters_present.append(character_name)
                elif not re.match(r'^[A-Z][A-Z\s]+', line):
                    # This is likely action or description
                    current_scene.action_lines.append(line)
        
        if current_scene:
            scenes.append(current_scene)
        
        logger.info(f"Extracted {len(scenes)} scenes")
        return scenes
    
    def extract_characters(self, script_text: str) -> Dict[str, Character]:
        """Extract character information from the script."""
        logger.info("Extracting characters...")
        characters = {}
        
        # Find all character names
        character_matches = self.character_pattern.findall(script_text)
        character_names = set()
        
        for match in character_matches:
            name = match.strip()
            # Clean up character names
            name = re.sub(r'\s*\([^)]*\)', '', name)
            name = re.sub(r'\s+', ' ', name).strip()
            
            # Skip common non-character elements
            if name not in ['NARRATOR', 'V.O.', 'O.S.', 'CONT\'D', 'CONTINUED', 'FADE IN', 'FADE OUT', 'CUT TO', 'DISSOLVE TO']:
                character_names.add(name)
        
        # Extract dialogue and analyze each character
        dialogue_matches = self.dialogue_pattern.findall(script_text)
        
        for character_name in character_names:
            character = Character(name=character_name)
            character_dialogue = []
            
            # Extract all dialogue for this character
            for match in dialogue_matches:
                speaker = re.sub(r'\s*\([^)]*\)', '', match[0]).strip()
                if speaker == character_name:
                    dialogue_text = match[1].strip()
                    character_dialogue.append(dialogue_text)
                    character.dialogue_count += 1
                    character.total_words += len(word_tokenize(dialogue_text))
            
            # Extract key quotes (longest and most significant)
            if character_dialogue:
                # Sort by length and take top quotes
                character_dialogue.sort(key=len, reverse=True)
                character.key_quotes = character_dialogue[:5]
                
                # Analyze personality traits
                character.personality_traits = self._analyze_personality(character_dialogue)
            
            characters[character_name] = character
        
        logger.info(f"Extracted {len(characters)} characters")
        return characters
    
    def _analyze_personality(self, dialogue: List[str]) -> List[str]:
        """Analyze personality traits from character dialogue."""
        if not nlp:
            return []
        
        traits = []
        all_text = ' '.join(dialogue)
        doc = nlp(all_text)
        
        # Analyze sentiment and emotional content
        sentiment_words = []
        for token in doc:
            if token.pos_ in ['ADJ', 'ADV'] and token.text.lower() not in self.stop_words:
                sentiment_words.append(token.text.lower())
        
        # Count sentiment words
        sentiment_counts = Counter(sentiment_words)
        
        # Define personality trait keywords
        trait_keywords = {
            'aggressive': ['angry', 'furious', 'violent', 'aggressive', 'hostile'],
            'friendly': ['nice', 'kind', 'friendly', 'warm', 'gentle'],
            'intelligent': ['smart', 'clever', 'wise', 'brilliant', 'genius'],
            'funny': ['funny', 'hilarious', 'joke', 'laugh', 'humor'],
            'serious': ['serious', 'grave', 'solemn', 'formal', 'strict'],
            'emotional': ['sad', 'happy', 'excited', 'worried', 'anxious'],
            'confident': ['confident', 'bold', 'brave', 'proud', 'strong'],
            'shy': ['shy', 'quiet', 'timid', 'nervous', 'hesitant']
        }
        
        for trait, keywords in trait_keywords.items():
            score = sum(sentiment_counts.get(word, 0) for word in keywords)
            if score > 0:
                traits.append(trait)
        
        return traits[:5]  # Return top 5 traits
    
    def analyze_relationships(self, script_text: str) -> Dict[str, Dict[str, int]]:
        """Analyze relationships between characters."""
        logger.info("Analyzing character relationships...")
        relationships = defaultdict(lambda: defaultdict(int))
        
        # Extract scenes and analyze character interactions
        scenes = self.extract_scenes(script_text)
        
        for scene in scenes:
            if len(scene.characters_present) > 1:
                # Count co-occurrences
                for i, char1 in enumerate(scene.characters_present):
                    for char2 in scene.characters_present[i+1:]:
                        relationships[char1][char2] += 1
                        relationships[char2][char1] += 1
        
        # Also analyze dialogue patterns
        dialogue_matches = self.dialogue_pattern.findall(script_text)
        current_speaker = None
        
        for match in dialogue_matches:
            speaker = re.sub(r'\s*\([^)]*\)', '', match[0]).strip()
            dialogue_text = match[1].strip()
            
            # Look for character mentions in dialogue
            for character_name in self.characters.keys():
                if character_name != speaker and character_name.lower() in dialogue_text.lower():
                    relationships[speaker][character_name] += 1
        
        logger.info("Character relationship analysis completed")
        return dict(relationships)
    
    def process_script(self, script_text: str) -> Dict:
        """Main processing function that orchestrates the entire pipeline."""
        logger.info("Starting script processing pipeline...")
        
        # Clean the script
        cleaned_script = self.clean_script(script_text)
        
        # Extract characters
        self.characters = self.extract_characters(cleaned_script)
        
        # Extract scenes
        self.scenes = self.extract_scenes(cleaned_script)
        
        # Analyze relationships
        self.character_relationships = self.analyze_relationships(cleaned_script)
        
        # Update character relationships
        for char_name, char in self.characters.items():
            if char_name in self.character_relationships:
                char.relationships = dict(self.character_relationships[char_name])
        
        # Create processed data structure
        processed_data = {
            'characters': {name: asdict(char) for name, char in self.characters.items()},
            'scenes': [asdict(scene) for scene in self.scenes],
            'relationships': dict(self.character_relationships),
            'metadata': {
                'total_characters': len(self.characters),
                'total_scenes': len(self.scenes),
                'script_length': len(cleaned_script),
                'processing_timestamp': str(datetime.now())
            }
        }
        
        logger.info("Script processing pipeline completed successfully")
        return processed_data
    
    def save_processed_data(self, processed_data: Dict, filename: str):
        """Save processed data to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Processed data saved to {filename}")
    
    def load_processed_data(self, filename: str) -> Dict:
        """Load processed data from JSON file."""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Processed data loaded from {filename}")
        return data

