# Patent Seed Keyword Extraction System

A system implementing a 4-step methodology with reflection and human-in-the-loop to extract seed keywords from technical/patent documents using LangChain, LangGraph and Ollama.

## 🏗️ Architecture

- **LangChain**: LLM integration and prompt processing
- **LangGraph**: Workflow with reflection and human-in-the-loop
- **Ollama**: Local LLM (Qwen2.5:0.5b-instruct)
- **Pydantic**: Data validation and structure

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install Ollama (if not already installed)
# https://ollama.ai/

# Pull Qwen2.5 model
ollama pull qwen2.5:0.5b-instruct
```

## 🚀 Usage

### 1. Interactive mode with new workflow

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

extractor = CoreConceptExtractor(model_name="qwen2.5:0.5b-instruct")
results = extractor.extract_keywords("Your patent description...")
```

## 📋 4-Step Process with Reflection

### Step 1: Document Summary by Fields

- Analyze input document  
- Create Concept Matrix with 6 components:
  - Problem/Purpose
  - Object/System
  - Action/Method
  - Key Technical Feature
  - Environment/Field
  - Advantage/Result

### Step 2: Main Keyword Generation

- From Concept Matrix → generate main keywords for each field
- Focus on technical specificity and search effectiveness
- Prioritize domain-specific terms and technical nouns

### Step 3: Reflection Evaluation ⭐ NEW

- **AI Self-Assessment**: Automatically evaluate keyword quality
- **Quality Metrics**: Technical specificity, distinctiveness, completeness
- **Smart Regeneration**: Auto-regenerate if quality is poor
- **Iteration Limit**: Maximum 3 reflection cycles to avoid infinite loops
- **Assessment Criteria**:
  - Technical specificity level
  - Search discriminative power
  - Coverage completeness
  - Redundancy detection
  - Generic term filtering

### Step 4: Human in the Loop Evaluation

- Review AI-approved keywords with reflection assessment
- Three options available:
  1. **✅ Approve**: Export to JSON file
  2. **❌ Reject**: Restart from beginning  
  3. **✏️ Edit**: Manually modify keywords and finish

## 🆕 Workflow Improvements

### Enhanced Quality Control

- **AI Reflection**: Automatic quality assessment before human review
- **Intelligent Regeneration**: Keywords regenerated automatically if quality is poor
- **Reflection Feedback**: Uses specific issues and recommendations for improvement
- **Iteration Control**: Prevents infinite reflection loops with maximum iteration limit

### Streamlined Process

- **Step 1-3**: Run automatically without interruption
- **Final Review**: Human evaluation only after AI approval
- **Better Context**: Users see both final keywords AND AI assessment
- **Targeted Feedback**: Rejection provides feedback for full restart

### Benefits

- 🤖 **AI Quality Gate**: Automatic filtering of poor-quality keywords
- 🎯 **Higher Baseline**: Human review starts with AI-approved keywords  
- 📊 **Transparency**: See AI reasoning and assessment scores
- 🔄 **Smart Iterations**: Reflection uses specific feedback for improvements
- ⚡ **Efficiency**: Less human intervention for obviously poor results

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
