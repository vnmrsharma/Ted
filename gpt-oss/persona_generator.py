"""
Persona Generator using GPT-OSS-20B Model
This module handles the integration with GPT-OSS-20B for generating realistic character personas
based on processed script data.
"""

import torch
import json
import logging
from typing import Dict, List, Optional, Tuple
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
import numpy as np
from dataclasses import dataclass
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PersonaConfig:
    """Configuration for persona generation."""
    model_name: str = "openai/gpt-oss-20b"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    use_quantization: bool = True
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

class PersonaGenerator:
    """Main class for generating character personas using GPT-OSS-20B."""
    
    def __init__(self, config: PersonaConfig = None):
        self.config = config or PersonaConfig()
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        
        logger.info(f"Initializing PersonaGenerator on device: {self.device}")
    
    def setup_quantization(self):
        """Setup quantization configuration for memory efficiency."""
        if not self.config.use_quantization:
            return None
        
        # For GPT-OSS-20B, the model is already quantized with Mxfp4Config
        # We don't need to apply additional quantization
        logger.info("GPT-OSS-20B model is already quantized, skipping additional quantization")
        return None
    
    def setup_lora(self) -> LoraConfig:
        """Setup LoRA configuration for efficient fine-tuning."""
        if not self.config.use_lora:
            return None
            
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        return lora_config
    
    def load_model(self):
        """Load the GPT-OSS-20B model and tokenizer."""
        if self.is_loaded:
            logger.info("Model already loaded")
            return
            
        logger.info(f"Loading GPT-OSS-20B model: {self.config.model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Setup quantization
            quantization_config = self.setup_quantization()
            
            # Load model - GPT-OSS-20B is already quantized
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            
            # Don't apply additional quantization as the model is already quantized
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            
            # Setup LoRA if enabled
            if self.config.use_lora:
                lora_config = self.setup_lora()
                self.model = get_peft_model(self.model, lora_config)
                logger.info("LoRA configuration applied")
            
            self.is_loaded = True
            logger.info("GPT-OSS-20B model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def create_character_context(self, character_data: Dict, script_data: Dict) -> str:
        """Create comprehensive context for a character based on script analysis."""
        context_parts = []
        
        # Character basic info
        context_parts.append(f"Character: {character_data['name']}")
        if character_data.get('full_name'):
            context_parts.append(f"Full Name: {character_data['full_name']}")
        if character_data.get('description'):
            context_parts.append(f"Description: {character_data['description']}")
        
        # Personality traits
        if character_data.get('personality_traits'):
            traits = ', '.join(character_data['personality_traits'])
            context_parts.append(f"Personality Traits: {traits}")
        
        # Key quotes and dialogue patterns
        if character_data.get('key_quotes'):
            context_parts.append("Key Quotes:")
            for quote in character_data['key_quotes'][:3]:  # Top 3 quotes
                context_parts.append(f'  "{quote}"')
        
        # Relationships
        if character_data.get('relationships'):
            context_parts.append("Relationships:")
            for other_char, strength in character_data['relationships'].items():
                if strength > 0:
                    context_parts.append(f"  - {other_char} (interaction strength: {strength})")
        
        # Scene appearances
        if character_data.get('scenes_appeared'):
            context_parts.append(f"Appears in {len(character_data['scenes_appeared'])} scenes")
        
        # Dialogue statistics
        context_parts.append(f"Dialogue Count: {character_data.get('dialogue_count', 0)}")
        context_parts.append(f"Total Words: {character_data.get('total_words', 0)}")
        
        return "\n".join(context_parts)
    
    def create_persona_prompt(self, character_data: Dict, script_data: Dict, user_input: str = "") -> str:
        """Create a comprehensive prompt for persona generation."""
        character_context = self.create_character_context(character_data, script_data)
        
        prompt = f"""You are {character_data['name']}, a character from a movie script. Based on the following character analysis, respond in character to any questions or conversations.

CHARACTER ANALYSIS:
{character_context}

SCRIPT CONTEXT:
This character appears in a movie script with {script_data['metadata']['total_characters']} characters across {script_data['metadata']['total_scenes']} scenes.

INSTRUCTIONS:
- Stay completely in character as {character_data['name']}
- Use the personality traits, dialogue patterns, and relationships shown above
- Respond naturally as this character would
- Maintain consistency with the character's established personality
- If asked about other characters, respond based on the relationships shown
- Keep responses engaging and true to the character

User: {user_input if user_input else "Hello, tell me about yourself."}

{character_data['name']}:"""
        
        return prompt
    
    def generate_response(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate a response using the GPT-OSS-20B model."""
        if not self.is_loaded:
            self.load_model()
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length - max_new_tokens,
            padding=True
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new generated text
        response = full_response[len(prompt):].strip()
        
        return response
    
    def fine_tune_character(self, character_data: Dict, script_data: Dict, output_dir: str = "character_models"):
        """Fine-tune the model for a specific character."""
        if not self.is_loaded:
            self.load_model()
        
        logger.info(f"Starting fine-tuning for character: {character_data['name']}")
        
        # Prepare model for training (for quantized models)
        try:
            from peft import prepare_model_for_kbit_training
            self.model = prepare_model_for_kbit_training(self.model)
            logger.info("Model prepared for k-bit training")
        except ImportError:
            logger.warning("prepare_model_for_kbit_training not available, using standard preparation")
        
        # Create training data
        training_data = self._create_training_data(character_data, script_data)
        
        if not training_data:
            logger.warning("No training data created, skipping fine-tuning")
            return
        
        # Create dataset
        dataset = Dataset.from_list(training_data)
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Training arguments optimized for quantized models
        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, character_data['name'].lower().replace(' ', '_')),
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=5e-5,
            logging_steps=10,
            save_steps=100,
            evaluation_strategy="no",
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=False,  # Disable fp16 for quantized models
            bf16=True if self.device == "cuda" else False,  # Use bf16 if available
            optim="adamw_torch",  # Use torch optimizer
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        logger.info(f"Fine-tuning completed for {character_data['name']}")
    
    def _create_training_data(self, character_data: Dict, script_data: Dict) -> List[Dict[str, str]]:
        """Create training data for fine-tuning."""
        training_data = []
        
        # Create persona examples
        persona_context = self.create_character_context(character_data, script_data)
        
        # Example conversations
        examples = [
            {
                "text": f"{persona_context}\n\nUser: Tell me about yourself.\n{character_data['name']}: I'm {character_data['name']}. {self._generate_character_intro(character_data)}"
            },
            {
                "text": f"{persona_context}\n\nUser: What are you like?\n{character_data['name']}: {self._generate_personality_description(character_data)}"
            },
            {
                "text": f"{persona_context}\n\nUser: How do you feel about the other characters?\n{character_data['name']}: {self._generate_relationship_opinions(character_data)}"
            }
        ]
        
        # Add dialogue examples from script
        if character_data.get('key_quotes'):
            for quote in character_data['key_quotes'][:2]:  # Use top 2 quotes
                examples.append({
                    "text": f"{persona_context}\n\nUser: Say something in character.\n{character_data['name']}: {quote}"
                })
        
        return examples
    
    def _generate_character_intro(self, character_data: Dict) -> str:
        """Generate a character introduction based on analysis."""
        traits = character_data.get('personality_traits', [])
        if traits:
            trait_desc = ', '.join(traits[:3])
            return f"I'm known for being {trait_desc}. "
        return "I'm a character in this story. "
    
    def _generate_personality_description(self, character_data: Dict) -> str:
        """Generate personality description."""
        traits = character_data.get('personality_traits', [])
        if traits:
            return f"I would describe myself as {', '.join(traits)}. "
        return "I have my own unique personality. "
    
    def _generate_relationship_opinions(self, character_data: Dict) -> str:
        """Generate opinions about other characters."""
        relationships = character_data.get('relationships', {})
        if relationships:
            top_relationship = max(relationships.items(), key=lambda x: x[1])
            return f"I have a strong relationship with {top_relationship[0]}. "
        return "I interact with various characters in the story. "
    
    def load_character_model(self, character_name: str, model_path: str):
        """Load a fine-tuned model for a specific character."""
        if not self.is_loaded:
            self.load_model()
        
        logger.info(f"Loading fine-tuned model for {character_name}")
        
        try:
            # Load the fine-tuned model
            self.model = PeftModel.from_pretrained(self.model, model_path)
            logger.info(f"Fine-tuned model loaded for {character_name}")
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {str(e)}")
            raise
    
    def chat_with_character(self, character_data: Dict, script_data: Dict, user_input: str) -> str:
        """Main function for chatting with a character persona."""
        prompt = self.create_persona_prompt(character_data, script_data, user_input)
        response = self.generate_response(prompt)
        return response
    
    def get_character_suggestions(self, script_data: Dict) -> List[Dict[str, str]]:
        """Get suggestions for characters that would make good personas."""
        characters = script_data.get('characters', {})
        suggestions = []
        
        for name, char_data in characters.items():
            # Score based on dialogue count and personality traits
            score = char_data.get('dialogue_count', 0) * 10
            score += len(char_data.get('personality_traits', [])) * 5
            score += len(char_data.get('key_quotes', [])) * 3
            
            suggestions.append({
                'name': name,
                'score': score,
                'dialogue_count': char_data.get('dialogue_count', 0),
                'personality_traits': char_data.get('personality_traits', []),
                'description': f"Character with {char_data.get('dialogue_count', 0)} lines of dialogue"
            })
        
        # Sort by score
        suggestions.sort(key=lambda x: x['score'], reverse=True)
        return suggestions[:10]  # Return top 10
