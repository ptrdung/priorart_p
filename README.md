# Patent Seed Keyword Extraction System

A simplified system implementing a 3-step methodology with human-in-the-loop to extract seed keywords from technical/patent documents using LangChain, LangGraph and Ollama.

## 🏗️ Architecture

- **LangChain**: LLM integration and prompt processing
- **LangGraph**: Simplified workflow with human-in-the-loop
- **Ollama**: Local LLM (Qwen2.5:32b-instruct)
- **Pydantic**: Data validation and structure

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install Ollama (if not already installed)
# https://ollama.ai/

# Pull Qwen2.5:32b model
ollama pull qwen2.5:32b-instruct
```

## 🚀 Usage

### 1. Interactive mode with simplified workflow

```bash
python demo_new_workflow.py
```

### 2. Run demo with sample data

```bash
python demo.py
```

### 3. Use programmatically

```python
from core_concept_extractor import CoreConceptExtractor

extractor = CoreConceptExtractor(model_name="qwen2.5:32b-instruct")
results = extractor.extract_keywords("Your patent description...")
```

## 📋 Simplified 3-Step Process

### Step 1: Document Summary by Fields

- Analyze input document  
- Create Concept Matrix with 6 components:
  - Problem/Purpose
  - Object/System
  - Action/Method
  - Key Technical Feature
  - Environment/Field
  - Advantage/Result

### Step 2: Keyword Generation

- From Concept Matrix → generate main keywords for each field
- Focus on technical specificity and search effectiveness
- Use optimized prompts for better keyword quality

### Step 3: Human in the Loop Evaluation

- Review generated keywords directly
- Three simple options:
  1. **✅ Approve**: Export to JSON file
  2. **❌ Reject**: Restart from beginning  
  3. **✏️ Edit**: Manually modify keywords and finish

## 🆕 Simplified Workflow Benefits

### Streamlined Process

- **Direct Path**: 3 steps without complex intermediate evaluations
- **Faster Execution**: No AI reflection delays
- **Clear Decisions**: Simple human choices at the end
- **Better Performance**: Uses powerful Qwen2.5:32b model for higher quality

### Benefits

- ⚡ **Speed**: Direct 3-step process
- 🎯 **Simplicity**: Clear workflow with single human decision point
- 🧠 **Quality**: Powerful 32b model produces better initial results
- 🛠️ **Control**: Direct editing capability for keyword refinement
- 🔄 **Efficiency**: Quick restart option with feedback

## 📁 File Structure

```text
priorart_project/
├── requirements.txt          # Dependencies
├── core_concept_extractor.py # Core system logic
├── prompts.py                # Prompt templates and messages
├── demo.py                   # Demo and interactive mode  
├── utils.py                  # Utilities and analysis tools
└── README.md                 # Documentation
```

## 🔧 Features

- ✅ Automated 3-phase workflow
- ✅ **LangChain Structured Output Parsers** for reliable JSON parsing
- ✅ Automatic keyword refinement and quality enhancement
- ✅ Final human evaluation with multiple options
- ✅ Direct manual editing capability
- ✅ Re-run option with feedback
- ✅ Output fixing parsers with fallback mechanisms
- ✅ Keyword quality analysis
- ✅ Boolean/Natural Language search query generation
- ✅ Detailed process reporting
- ✅ JSON results export

### Structured Output Parsing

- **Pydantic Models**: Well-defined output schemas for each phase
- **Auto-correction**: Output fixing parsers handle malformed LLM responses
- **Fallback Mechanisms**: Manual parsing as backup for critical failures
- **Type Safety**: Guaranteed data structure consistency

## 🎯 Output

The system generates:

1. **Concept Matrix**: 6 core components
2. **Seed Keywords**: 1-3 keywords/component  
3. **Search Queries**: Boolean and Natural Language
4. **Quality Report**: Analysis and recommendations

## 📊 Example Output

```json
{
  "final_keywords": {
    "problem_purpose": ["water conservation", "automatic irrigation"],
    "object_system": ["irrigation system"],
    "action_method": ["control", "schedule"],
    "key_technical_feature": ["soil moisture sensor", "weather data"],
    "environment_field": ["agriculture", "gardening"],
    "advantage_result": ["optimize water usage", "reduce cost"]
  }
}
```

## 🔍 Search Queries

**Boolean Query:**

```text
("water conservation" OR "automatic irrigation" OR "irrigation system") AND ("control" OR "schedule") AND ("agriculture" OR "gardening")
```

**Natural Language Query:**

```text
water conservation automatic irrigation irrigation system control schedule soil moisture sensor weather data
```
