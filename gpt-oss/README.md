# 🎭 Character Persona Generator

A comprehensive system for generating realistic character personas from movie scripts using the GPT-OSS-20B model. This tool allows filmmakers and actors to gain deeper understanding of characters through AI-powered persona generation and interactive chat interfaces.

## 🌟 Features

- **Intensive Script Processing**: Advanced NLP pipeline for character extraction and relationship mapping
- **GPT-OSS-20B Integration**: State-of-the-art language model for realistic persona generation
- **Character Analysis**: Automatic personality trait detection and relationship analysis
- **Interactive Chat Interface**: Chat with any character persona using Streamlit
- **Fine-tuning Capabilities**: Customize models for specific characters
- **Memory Efficient**: 4-bit quantization and LoRA for optimal performance

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for best performance)
- At least 16GB RAM (32GB recommended for full model)

### Installation

1. **Clone and navigate to the project:**
```bash
cd gpt-oss
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install spaCy English model:**
```bash
python -m spacy download en_core_web_sm
```

5. **Download NLTK data:**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('punkt_tab')"
```

### Running the Application

#### Option 1: Simple Demo (Recommended for testing)
```bash
python simple_demo.py
```
This will process your script and show character analysis without requiring the GPT-OSS-20B model.

#### Option 2: Full Web Interface
1. **Start the application:**
```bash
streamlit run main.py
```

2. **Open your browser and navigate to:**
```
http://localhost:8501
```
**Note:** The web interface requires internet connection to download the GPT-OSS-20B model.

## 📖 Usage Guide

### Step 1: Script Processing
1. Upload a movie script file (.txt) or paste the script text
2. Click "Process Script" to analyze characters and relationships
3. Review the processing summary and character suggestions

### Step 2: Model Initialization
1. Configure model settings (quantization, LoRA)
2. Click "Initialize Model" to load GPT-OSS-20B
3. Wait for model loading (may take several minutes)

### Step 3: Character Selection
1. Choose a character from the dropdown menu
2. Review the character profile and analysis
3. Click "Select Character" to activate the persona

### Step 4: Chat with Character
1. Type messages in the chat interface
2. Receive responses in character
3. Optionally fine-tune the model for better responses

## 📁 Project Structure

```
gpt-oss/
├── main.py                 # Main application with Streamlit interface
├── simple_demo.py          # Simple command-line demo (recommended for testing)
├── script_processor.py     # Script analysis and character extraction
├── persona_generator.py    # GPT-OSS-20B integration and persona generation
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── script.txt             # Sample movie script (TED)
└── venv/                  # Virtual environment
```

## 🎯 Character Analysis Features

### Automatic Character Detection
- Extracts character names from script formatting
- Identifies dialogue patterns and speaking frequency
- Maps character relationships and interactions

### Personality Trait Analysis
- Uses NLP to identify personality traits from dialogue
- Analyzes emotional content and speech patterns
- Generates personality profiles automatically

### Relationship Mapping
- Tracks character co-occurrences in scenes
- Analyzes dialogue references between characters
- Creates relationship strength matrices

## 🔄 Fine-tuning Process

### When to Fine-tune
- Character has unique speech patterns
- Standard responses don't capture character essence
- Need more accurate personality representation

### Fine-tuning Process
1. Select character and click "Fine-tune for Better Responses"
2. System creates training data from character's dialogue
3. LoRA fine-tuning runs for 3 epochs
4. Model saves automatically for future use

## 🛠️ Advanced Usage

### Custom Script Formats

To support different script formats, modify the regex patterns in `script_processor.py`:

```python
# Example: Support for different character name formats
self.character_pattern = re.compile(r'^([A-Z][A-Z\s]+?)(?:\s*\([^)]*\))?\s*$', re.MULTILINE)
```

### Batch Processing

Process multiple scripts:

```python
from script_processor import ScriptProcessor

processor = ScriptProcessor()
scripts = ["script1.txt", "script2.txt", "script3.txt"]

for script_file in scripts:
    with open(script_file, 'r') as f:
        script_text = f.read()
    
    processed_data = processor.process_script(script_text)
    processor.save_processed_data(processed_data, f"processed_{script_file}")
```

### API Integration

Use the persona generator programmatically:

```python
from persona_generator import PersonaGenerator, PersonaConfig
from script_processor import ScriptProcessor

# Initialize components
processor = ScriptProcessor()
generator = PersonaGenerator()

# Process script
with open("script.txt", 'r') as f:
    script_text = f.read()
processed_data = processor.process_script(script_text)

# Generate persona response
character_data = processed_data['characters']['CHARACTER_NAME']
response = generator.chat_with_character(character_data, processed_data, "Hello!")
print(response)
```

## 🔧 Configuration

### Model Configuration

Edit `persona_generator.py` to modify model settings:

```python
@dataclass
class PersonaConfig:
    model_name: str = "openai/gpt-oss-20b"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    use_quantization: bool = True
    use_lora: bool = True
```

## 🐛 Troubleshooting

### Common Issues

**Model Loading Errors:**
- Ensure sufficient GPU memory (16GB+ recommended)
- Try enabling quantization for memory efficiency
- Check CUDA installation and compatibility

**Script Processing Issues:**
- Verify script format matches expected patterns
- Check for proper character name formatting
- Ensure script has sufficient dialogue content

**Character Selection Problems:**
- Process script before selecting characters
- Verify character names exist in processed data
- Check character has sufficient dialogue for analysis

### Performance Optimization

**Memory Usage:**
- Use 4-bit quantization for large models
- Enable LoRA for efficient fine-tuning
- Process scripts in smaller chunks if needed

**Speed Improvements:**
- Use GPU acceleration when available
- Enable model caching
- Batch process multiple characters

## 📊 Model Performance

### GPT-OSS-20B Capabilities
- 20 billion parameters for high-quality responses
- Advanced reasoning and context understanding
- Excellent character consistency and personality capture

### Quantization Benefits
- 4-bit quantization reduces memory usage by ~75%
- Minimal quality loss for most use cases
- Enables running on consumer hardware

### LoRA Fine-tuning
- Efficient parameter updates (0.1% of model parameters)
- Fast fine-tuning (minutes vs hours)
- Preserves base model capabilities

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- OpenAI for the GPT-OSS-20B model
- Hugging Face for the Transformers library
- Gradio for the user interface framework
- The open-source community for various dependencies

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review existing GitHub issues
3. Create a new issue with detailed information
4. Include system specifications and error logs

---

**Happy Character Creation! 🎭✨**