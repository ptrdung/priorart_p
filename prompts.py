"""
Prompt templates for patent seed keyword extraction system
"""

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict


class ExtractionPrompts:
    """Collection of prompt templates for the 3-phase extraction process"""
    
    @staticmethod
    def get_phase1_prompt_and_parser():
        """Phase 1: Concept Matrix extraction prompt with parser"""
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
    def get_phase1_prompt() -> PromptTemplate:
        """Legacy method for backward compatibility"""
        prompt, _ = ExtractionPrompts.get_phase1_prompt_and_parser()
        return prompt
    
    @staticmethod
    def get_phase2_prompt_and_parser():
        """Phase 2: Seed keyword extraction prompt with parser"""
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
    def get_phase2_prompt() -> PromptTemplate:
        """Legacy method for backward compatibility"""
        prompt, _ = ExtractionPrompts.get_phase2_prompt_and_parser()
        return prompt
    
    @staticmethod
    def get_phase3_auto_prompt_and_parser():
        """Phase 3: Automatic refinement prompt with parser"""
        parser = PydanticOutputParser(pydantic_object=SeedKeywordsOutput)
        
        prompt = PromptTemplate(
            template="""You are a technical keyword optimization specialist for patent search and prior art analysis.

**Task:**  
Automatically refine and enhance the seed keywords based on the provided concept matrix and initial extraction.

**Instructions:**  
- Review the **Original Concept Matrix** and the **Current Keywords**.
- Perform the following steps in order:
  1. **Identify and add important missing technical terms** from the Concept Matrix that are not present in the Current Keywords.
  2. **Remove overly general, non-technical, or redundant terms.**
  3. Ensure each keyword is **technically specific, distinctive, and valuable for patent searches** — prioritize component names, algorithm/process names, sensor types, and industry-standard terminology.
  4. **Avoid duplicate or synonymous terms across categories.** If a term is conceptually similar to an existing keyword, omit it.
  5. Ensure **comprehensive coverage across all Concept Matrix components** by assigning **1-3 optimized keywords per category.**
  6. Optimize the final list for **patent search discriminative power** — select terms most likely to improve prior art search precision.

**Focus on:**  
- Highly technical terminology typically used in patent claims and prior art documents  
- Specific component names, processing methods, and unique technical features  
- Standardized industry terminology relevant to the domain  
- Terms that clearly distinguish this invention from related technologies

**Original Concept Matrix:**  
{concept_matrix}

**Current Keywords:**  
{current_keywords}

**Output:**  
Provide the final improved keyword list in the following format:

{format_instructions}
""",
            input_variables=["concept_matrix", "current_keywords"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        return prompt, parser
    
    @staticmethod
    def get_phase3_auto_prompt() -> PromptTemplate:
        """Legacy method for backward compatibility"""
        prompt, _ = ExtractionPrompts.get_phase3_auto_prompt_and_parser()
        return prompt
    
    @staticmethod
    def get_phase3_prompt_and_parser():
        """Phase 3: Manual refinement prompt with parser (for re-runs)"""
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
    
    @staticmethod
    def get_reflection_prompt_and_parser():
        """Reflection step: Evaluate generated keywords quality"""
        parser = PydanticOutputParser(pydantic_object=ReflectionEvaluation)
        
        prompt = PromptTemplate(
            template="""You are an expert patent search keyword quality evaluator.

**Task:**  
Evaluate the quality of the generated seed keywords for patent search effectiveness.

**Instructions:**  
Analyze the keywords extracted from each component of the concept matrix and provide:
1. An overall quality assessment: 'good' or 'poor'
2. Individual scores (0-1) for each keyword category
3. Specific issues found (if any)
4. Recommendations for improvement
5. Whether keywords should be regenerated

**Evaluation Criteria:**
- **Technical Specificity**: Are keywords technically precise and domain-specific?
- **Search Discriminative Power**: Will these keywords effectively narrow down patent search results?
- **Coverage**: Do keywords adequately represent the technical concept in each category?
- **Distinctiveness**: Are keywords unique and not overly generic?
- **Completeness**: Are any important technical terms missing?

**Concept Matrix:**
- Problem/Purpose: {problem_purpose}
- Object/System: {object_system}  
- Action/Method: {action_method}
- Key Technical Feature: {key_technical_feature}
- Environment/Field: {environment_field}
- Advantage/Result: {advantage_result}

**Generated Keywords:**
- Problem/Purpose Keywords: {problem_purpose_keywords}
- Object/System Keywords: {object_system_keywords}
- Action/Method Keywords: {action_method_keywords}
- Key Technical Feature Keywords: {key_technical_feature_keywords}
- Environment/Field Keywords: {environment_field_keywords}
- Advantage/Result Keywords: {advantage_result_keywords}

**Iteration:** {iteration}

{format_instructions}
""",
            input_variables=["problem_purpose", "object_system", "action_method", 
                           "key_technical_feature", "environment_field", "advantage_result",
                           "problem_purpose_keywords", "object_system_keywords", 
                           "action_method_keywords", "key_technical_feature_keywords",
                           "environment_field_keywords", "advantage_result_keywords", "iteration"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        return prompt, parser

    @staticmethod
    def get_validation_messages() -> dict:
        """User interface messages for validation phase"""
        return {
            "title": "🔍 KEYWORD EXTRACTION RESULTS EVALUATION",
            "final_evaluation_title": "🎯 FINAL RESULTS EVALUATION",
            "separator": "="*60,
            "concept_matrix_header": "\n📋 Concept Matrix:",
            "seed_keywords_header": "\n🔑 Final Seed Keywords:",
            "divider": "\n" + "-"*60,
            "action_options": """
Choose your action:
1. [A]pprove - Accept the results as final
2. [E]dit - Manually edit the keywords
3. [R]erun - Run the extraction process again with feedback
            """,
            "action_prompt": "Your choice (1/2/3 or a/e/r): ",
            "rerun_feedback_prompt": "Feedback for re-run (what should be improved): ",
            "invalid_action": "Please enter 1, 2, 3 or a, e, r",
            "approved": "Approved",
            "edited": "Manually edited",
            "rerun_requested": "Re-run requested"
        }
    
    @staticmethod
    def get_phase_completion_messages() -> dict:
        """Messages for phase completion"""
        return {
            "phase1_completed": "Phase 1 completed: Concept Matrix extracted",
            "phase2_completed": "Phase 2 completed: Initial seed keywords extracted", 
            "phase3_completed": "Phase 3 completed: Keywords automatically refined",
            "extraction_completed": "✅ Patent seed keyword extraction completed",
            "user_evaluation": "User evaluation: {status}",
            "manual_edit_completed": "Keywords manually edited by user",
            "rerun_initiated": "Re-running extraction process with user feedback"
        }
    
    @staticmethod
    def get_analysis_prompts() -> dict:
        """Additional prompts for keyword analysis"""
        return {
            "quality_analysis": """
            Analyze the quality of these patent search keywords:
            {keywords}
            
            Evaluate based on:
            1. Technical specificity
            2. Search effectiveness
            3. Coverage of key concepts
            4. Distinctiveness
            
            Provide recommendations for improvement.
            """,
            
            "search_strategy": """
            Create a patent search strategy using these keywords:
            {keywords}
            
            Provide:
            1. Primary search terms (most important)
            2. Secondary search terms (supporting)
            3. Boolean query structure
            4. Alternative terms to consider
            """
        }
    
    @staticmethod
    def get_concept_matrix_parser():
        """Get parser for concept matrix output"""
        from langchain.output_parsers import PydanticOutputParser
        return PydanticOutputParser(pydantic_object=ConceptMatrixOutput)
    
    @staticmethod
    def get_seed_keywords_parser():
        """Get parser for seed keywords output"""
        from langchain.output_parsers import PydanticOutputParser
        return PydanticOutputParser(pydantic_object=SeedKeywordsOutput)
    
    @staticmethod
    def create_output_fixing_parser(base_parser, llm):
        """Create an output fixing parser that can handle malformed responses"""
        from langchain.output_parsers import OutputFixingParser
        return OutputFixingParser.from_llm(parser=base_parser, llm=llm)

# Output Models for Structured Parsing
from pydantic import BaseModel, Field

class ConceptMatrixOutput(BaseModel):
    """Output model for Phase 1 concept extraction from technical documents"""
    
    problem_purpose: str = Field(
        description="What specific problem is addressed or what is the main objective, as explicitly stated in the document?"
    )
    object_system: str = Field(
        description="What is the main object, device, system, or process being described?"
    )
    action_method: str = Field(
        description="What actions, processes, or methods are applied or proposed in the document?"
    )
    key_technical_feature: str = Field(
        description="What are the essential technical features, structures, or configurations that enable the system or method?"
    )
    environment_field: str = Field(
        description="What is the application domain or operating environment?"
    )
    advantage_result: str = Field(
        description="What specific benefits, improvements, or outcomes are achieved, according to the document?"
    )


class ReflectionEvaluation(BaseModel):
    """Output model for reflection evaluation of keywords"""
    overall_quality: str = Field(
        description="Overall quality assessment: 'good' or 'poor'"
    )
    keyword_scores: Dict[str, float] = Field(
        description="Score for each category (0-1)"
    )
    issues_found: List[str] = Field(
        description="List of specific issues identified"
    )
    recommendations: List[str] = Field(
        description="Recommendations for improvement"
    )
    should_regenerate: bool = Field(
        description="Whether keywords should be regenerated"
    )


class SeedKeywordsOutput(BaseModel):
    """Output model for Phase 2 and 3 keyword extraction"""
    problem_purpose_keywords: List[str] = Field(
        description="Distinctive technical keywords extracted from the problem or purpose component."
    )
    object_system_keywords: List[str] = Field(
        description="Distinctive technical keywords extracted from the object or system component."
    )
    action_method_keywords: List[str] = Field(
        description="Distinctive technical keywords extracted from the action or method component."
    )
    key_technical_feature_keywords: List[str] = Field(
        description="Distinctive technical keywords extracted from the key technical feature or structure component."
    )
    environment_field_keywords: List[str] = Field(
        description="Distinctive technical keywords extracted from the environment or application field component."
    )
    advantage_result_keywords: List[str] = Field(
        description="Distinctive technical keywords extracted from the advantage or result component."
    )

if __name__ == "__main__":
    # Test prompts functionality
    print("🧪 Testing prompt templates...")
    
    # Test Phase 1 prompt
    phase1_prompt = ExtractionPrompts.get_phase1_prompt()
    print("\n📋 Phase 1 Prompt Template:")
    print("✅ Created successfully")
    
    # Test Phase 2 prompt
    phase2_prompt = ExtractionPrompts.get_phase2_prompt()
    print("\n🔑 Phase 2 Prompt Template:")
    print("✅ Created successfully")
    
    # Test Phase 3 prompt
    phase3_prompt = ExtractionPrompts.get_phase3_prompt()
    print("\n🔧 Phase 3 Prompt Template:")
    print("✅ Created successfully")
    
    # Test messages
    validation_msgs = ExtractionPrompts.get_validation_messages()
    completion_msgs = ExtractionPrompts.get_phase_completion_messages()
    analysis_prompts = ExtractionPrompts.get_analysis_prompts()
    
    print(f"\n📢 Validation messages: {len(validation_msgs)} items")
    print(f"✅ Completion messages: {len(completion_msgs)} items")
    print(f"🔍 Analysis prompts: {len(analysis_prompts)} items")
    
    print("\n🎉 All prompt templates loaded successfully!")
