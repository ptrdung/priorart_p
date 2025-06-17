"""
Simplified prompt templates for patent seed keyword extraction system
"""

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict


# Simplified Data Models
class ConceptMatrixOutput(BaseModel):
    """Output model for Phase 1 concept extraction"""
    problem_purpose: str = Field(description="Problem/Purpose")
    object_system: str = Field(description="Object/System")
    action_method: str = Field(description="Action/Method")
    key_technical_feature: str = Field(description="Key Technical Feature")
    environment_field: str = Field(description="Environment/Field")
    advantage_result: str = Field(description="Advantage/Result")


class SeedKeywordsOutput(BaseModel):
    """Output model for Phase 2 and 3 keyword extraction"""
    problem_purpose_keywords: List[str] = Field(description="Problem/Purpose keywords")
    object_system_keywords: List[str] = Field(description="Object/System keywords")
    action_method_keywords: List[str] = Field(description="Action/Method keywords")
    key_technical_feature_keywords: List[str] = Field(description="Key Technical Feature keywords")
    environment_field_keywords: List[str] = Field(description="Environment/Field keywords")
    advantage_result_keywords: List[str] = Field(description="Advantage/Result keywords")


class ExtractionPrompts:
    """Simplified collection of prompt templates"""
    
    @staticmethod
    def get_phase1_prompt_and_parser():
        """Phase 1: Concept Matrix extraction"""
        parser = PydanticOutputParser(pydantic_object=ConceptMatrixOutput)
        
        prompt = PromptTemplate(
            template="""You are a patent analysis specialist with expertise in extracting structured, factual insights from scientific and patent-related documents for patent search and prior art mapping.

**Task:**  
Carefully analyze the following technical document and extract concise, factual information for each component in the Concept Matrix. Only use information explicitly stated in the document — do not infer, assume, extrapolate, or include any unstated details.

**Document:**  
{input_text}

**Instructions:**  
For each component below, provide a concise, factual summary. If a component is not mentioned in the document, state: `Not mentioned.`

**Concept Matrix:**  
1. **Problem/Purpose** — Identify the specific technical problem addressed or the primary objective of the document.  
2. **Object/System** — Specify the main object, device, system, or process being described.  
3. **Action/Method** — Summarize the actions, operations, or methods applied or proposed.  
4. **Key Technical Feature/Structure** — Highlight essential technical features, structures, or configurations enabling the system or method.  
5. **Environment/Field** — Indicate the application domain, industry, or operational context.  
6. **Advantage/Result** — State the specific benefits, improvements, or outcomes achieved according to the document.

{format_instructions}
""",
            input_variables=["input_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        return prompt, parser
    
    @staticmethod
    def get_phase2_prompt_and_parser():
        """Phase 2: Seed keyword extraction"""
        parser = PydanticOutputParser(pydantic_object=SeedKeywordsOutput)
        
        prompt = PromptTemplate(
            template="""You are an expert patent search analyst specializing in extracting high-value, domain-specific technical keywords for prior art search and patent landscaping.

**Task:**  
From the following Concept Matrix, extract distinctive, high-impact technical keywords for each component.

**Instructions:**  
- Focus exclusively on:
  - Specific **technical nouns** (e.g., "optical sensor", "convolutional layer")
  - Precise **technical action verbs or processes** (e.g., "segmentation", "data fusion")
- **Exclude generic terms** such as "system", "method", "device", unless paired with a qualified technical descriptor.
- **Do not generate keywords** if no relevant technical terms are available — leave the field empty.
- **Avoid duplicate or synonymous terms across components.** If a term is conceptually similar, a synonym, variant, or plural form of a previously selected keyword, omit it.
- **Prioritize terms with high discriminative power for patent search** — including sensor types, algorithm names, processing techniques, or domain-specific technologies.
- **Avoid generic keyphrases** unless inherently technical and discriminative.
- For each component, extract a concise list of unique, domain-relevant keywords.

**Concept Matrix:**  
- Problem/Purpose: {problem_purpose}  
- Object/System: {object_system}  
- Action/Method: {action_method}  
- Key Technical Feature: {key_technical_feature}  
- Environment/Field: {environment_field}  
- Advantage/Result: {advantage_result}  

{format_instructions}
""",
            input_variables=["problem_purpose", "object_system", "action_method", 
                           "key_technical_feature", "environment_field", "advantage_result"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        return prompt, parser
    
    @staticmethod
    def get_phase3_prompt_and_parser():
        """Phase 3: Keyword refinement with feedback"""
        parser = PydanticOutputParser(pydantic_object=SeedKeywordsOutput)
        
        prompt = PromptTemplate(
            template="""You are a technical keyword optimization expert specializing in patent search and prior art analysis.

**Task:**  
Refine the following seed keywords based on user feedback to improve their distinctiveness, technical specificity, and patent search value.

**Instructions:**  
- Review the **current keywords** and the **user feedback**.  
- Based on the feedback:
  1. **Identify and add missing important technical concepts** explicitly mentioned in the feedback.
  2. **Remove overly generic or non-technical terms** from the current keywords.
  3. Ensure each keyword is **highly distinctive and technically specific** — prioritize algorithm names, sensor types, process names, or domain-specific technical terms.
  4. **Avoid duplicate keywords or synonyms/variants of existing keywords.** If a suggested term has the same or similar meaning as an existing one, omit it.
  5. Optimize the final list for **patent search discriminative power** — select terms most likely to enhance prior art search precision.

**Current keywords:**  
{current_keywords}

**User feedback:**  
{feedback}

**Output:**  
Return the final list of improved keywords in the following format:

{format_instructions}
""",
            input_variables=["current_keywords", "feedback"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        return prompt, parser
    
    # Phương thức reflection đã được loại bỏ trong workflow mới

    # Legacy methods for backward compatibility
    @staticmethod
    def get_phase1_prompt():
        prompt, _ = ExtractionPrompts.get_phase1_prompt_and_parser()
        return prompt
    
    @staticmethod
    def get_phase2_prompt():
        prompt, _ = ExtractionPrompts.get_phase2_prompt_and_parser()
        return prompt
    
    @staticmethod
    def get_phase3_prompt():
        prompt, _ = ExtractionPrompts.get_phase3_prompt_and_parser()
        return prompt
    
    # Simplified message collections
    @staticmethod
    def get_validation_messages():
        return {
            "title": "🔍 KEYWORD EXTRACTION RESULTS",
            "separator": "="*60,
            "final_evaluation_title": "🎯 FINAL EVALUATION - HUMAN DECISION",
            "concept_matrix_header": "\n📋 Concept Matrix:",
            "seed_keywords_header": "\n🔑 Generated Keywords:",
            "divider": "\n" + "-"*60,
            "action_options": "\n📝 Choose your action:\n  1. ✅ Approve - Export to JSON file\n  2. ❌ Reject - Restart workflow\n  3. ✏️  Edit - Manually modify keywords",
            "action_prompt": "\nEnter your choice [1/2/3 or approve/reject/edit]: ",
            "reject_feedback_prompt": "\nOptional: Provide feedback for improvement: ",
            "invalid_action": "❌ Invalid choice. Please enter 1, 2, 3, approve, reject, or edit.",
            "approved": "✅ Keywords approved - exporting to JSON",
            "edited": "✏️ Keywords manually edited", 
            "rejected": "❌ Keywords rejected - restarting workflow"
        }
    
    @staticmethod
    def get_phase_completion_messages():
        return {
            "phase1_completed": "Phase 1 completed: Concept Matrix extracted",
            "phase2_completed": "Phase 2 completed: Seed keywords extracted", 
            "phase3_completed": "Phase 3 completed: Keywords refined",
            "extraction_completed": "✅ Patent seed keyword extraction completed"
        }
    
    # Simplified parser methods
    @staticmethod
    def get_concept_matrix_parser():
        return PydanticOutputParser(pydantic_object=ConceptMatrixOutput)
    
    @staticmethod
    def get_seed_keywords_parser():
        return PydanticOutputParser(pydantic_object=SeedKeywordsOutput)
    
    # Phương thức reflection đã được loại bỏ trong workflow mới


if __name__ == "__main__":
    print("🧪 Testing simplified prompt templates...")
    
    # Test prompts
    phase1_prompt, phase1_parser = ExtractionPrompts.get_phase1_prompt_and_parser()
    phase2_prompt, phase2_parser = ExtractionPrompts.get_phase2_prompt_and_parser()
    phase3_prompt, phase3_parser = ExtractionPrompts.get_phase3_prompt_and_parser()
    
    print("✅ Phase 1 prompt and parser created")
    print("✅ Phase 2 prompt and parser created")
    print("✅ Phase 3 prompt and parser created")
    
    # Test messages
    validation_msgs = ExtractionPrompts.get_validation_messages()
    completion_msgs = ExtractionPrompts.get_phase_completion_messages()
    
    print(f"✅ Validation messages: {len(validation_msgs)} items")
    print(f"✅ Completion messages: {len(completion_msgs)} items")
    
    print("\n🎉 All simplified prompt templates work correctly!")
