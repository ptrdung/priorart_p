"""
Input Normalization Prompt Templates
Contains prompt templates and parsers for input normalization before patent extraction
"""

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional

class NormalizedInputOutput(BaseModel):
    """Output model for normalized input"""
    normalized_text: str = Field(
        description="The normalized and cleaned version of the input text maintaining all original technical information"
    )
    input_type: str = Field(
        description="Type of input detected: 'patent_description', 'technical_document', 'research_paper', 'brief_idea', 'specification', or 'other'"
    )
    language: str = Field(
        description="Detected language of the input text"
    )
    completeness_score: float = Field(
        description="Score from 0.0 to 1.0 indicating how complete the technical description is"
    )
    technical_complexity: str = Field(
        description="Level of technical complexity: 'basic', 'intermediate', 'advanced'"
    )
    quality_notes: Optional[str] = Field(
        description="Optional notes about input quality or normalization performed"
    )

class NormalizationPrompts:
    """Collection of prompt templates for input normalization"""
    
    @staticmethod
    def get_input_normalization_prompt_and_parser():
        """Input normalization prompt and parser"""
        parser = PydanticOutputParser(pydantic_object=NormalizedInputOutput)
        
        prompt = PromptTemplate(
            template="""<OBJECTIVE_AND_PERSONA>
You are an expert technical document processor specializing in patent and technical documentation normalization. Your task is to analyze, clean, and standardize the input text while preserving all original technical information and maintaining absolute fidelity to the source content.
</OBJECTIVE_AND_PERSONA>

<INSTRUCTIONS>
To complete this task, you need to follow these steps:
1. Carefully read and analyze the provided input text
2. Identify the type and characteristics of the input document
3. Clean and normalize the text structure while preserving all technical content
4. Assess the completeness and technical complexity of the description
5. Ensure no technical information is lost, modified, or fabricated
6. Provide metadata about the input for downstream processing
7. Output the normalized text with preserved technical accuracy
</INSTRUCTIONS>

<CONSTRAINTS>
Dos:
- Preserve ALL original technical information exactly as provided
- Maintain the exact meaning and technical specifications from the source
- Clean up formatting inconsistencies, spelling errors, and structural issues
- Standardize terminology only when clearly equivalent (e.g., "temp" -> "temperature")
- Organize information logically while maintaining original content
- Detect and note the input type and characteristics accurately
- Ensure the normalized text is coherent and well-structured

Don'ts:
- Do NOT add any technical information not present in the original
- Do NOT modify technical specifications, measurements, or parameters
- Do NOT interpret or infer technical details beyond what is explicitly stated
- Do NOT paraphrase technical terms that may have specific meanings
- Do NOT remove any technical content, even if it seems redundant
- Do NOT fabricate or hallucinate any details
- Do NOT change the fundamental meaning of any statements
</CONSTRAINTS>

<CONTEXT>
The input text may be in various formats including:
- Patent descriptions or abstracts
- Technical specifications
- Research papers or reports
- Brief invention ideas
- Technical documentation
- Product descriptions

Input text to normalize:
{input_text}

Your task is to clean and standardize this input while maintaining absolute fidelity to all technical information.
</CONTEXT>

<OUTPUT_FORMAT>
{format_instructions}
</OUTPUT_FORMAT>

<RECAP>
Normalize the input text by cleaning structure and formatting while preserving every technical detail exactly as provided. Assess input characteristics and provide metadata. Maintain absolute fidelity to source content - no additions, modifications, or interpretations of technical information. IMPORTANT: Only generate the JSON output as defined - do not provide explanations, commentary, or any additional text beyond the required JSON format.
</RECAP>""",
            input_variables=["input_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        return prompt, parser
    
    @staticmethod
    def get_validation_messages():
        """Get validation messages for normalization process"""
        return {
            "normalization_started": "🔄 Starting input normalization...",
            "normalization_completed": "✅ Input normalization completed",
            "low_quality_warning": "⚠️  Input quality is low - results may be limited",
            "incomplete_input_warning": "⚠️  Input appears incomplete - consider providing more details",
            "processing_complete": "✅ Input processed and ready for extraction"
        }
