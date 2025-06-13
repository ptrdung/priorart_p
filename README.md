# Patent Seed Keyword Extraction System

A system implementing a 3-phase methodology to extract seed keywords from technical/patent documents using LangChain, LangGraph and Ollama.

## 🏗️ Architecture

- **LangChain**: LLM integration and prompt processing
- **LangGraph**: Workflow with human-in-the-loop
- **Ollama**: Local LLM (Llama3)
- **Pydantic**: Data validation and structure

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install Ollama (if not already installed)
# https://ollama.ai/

# Pull Llama3 model
ollama pull llama3
```

## 🚀 Usage

### 1. Interactive mode

```bash
python demo.py
```

### 2. Run demo with sample data

```bash
python demo.py demo
```

### 3. Use programmatically

```python
from core_concept_extractor import CoreConceptExtractor

extractor = CoreConceptExtractor(model_name="llama3")
results = extractor.extract_keywords("Your patent description...")
```

## 📋 3-Phase Process

### Phase 1: Abstraction & Concept Definition

- Analyze input document
- Create Concept Matrix with 6 components

### Phase 2: Initial Seed Keyword Extraction

- From Concept Matrix → 1-3 keywords/component
- Prioritize technical nouns and main verbs

### Phase 3: Automatic Refinement & Quality Enhancement

- Automatically improve keyword quality and specificity
- Optimize for patent search effectiveness
- Ensure technical precision and coverage

### Final Human Evaluation

- Review complete results only at the end
- Three options available:
  1. **Approve**: Accept results as final
  2. **Manual Edit**: Directly modify keywords
  3. **Re-run**: Restart process with feedback

## 🆕 Workflow Improvements

### Enhanced User Experience
- **Streamlined Process**: Phases 1-3 run automatically without interruption
- **Final Review**: Human evaluation only at the end with complete results
- **Flexible Actions**: Three clear options for final results handling
- **Direct Editing**: Modify keywords immediately without going through refinement cycles
- **Smart Re-runs**: Restart with specific feedback for targeted improvements

### Benefits
- ⚡ **Faster Processing**: No intermediate stops for validation
- 🎯 **Better Focus**: Evaluate complete results rather than partial outputs  
- ✏️ **Direct Control**: Manual editing capability for precise adjustments
- 🔄 **Efficient Iterations**: Targeted re-runs with specific feedback
- 📈 **Higher Quality**: Automatic refinement ensures consistent baseline quality

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
