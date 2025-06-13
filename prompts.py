"""
Prompt templates for patent seed keyword extraction system
"""

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List


class ExtractionPrompts:
    """Collection of prompt templates for the 3-phase extraction process"""
    
    @staticmethod
    def get_phase1_prompt_and_parser():
        """Phase 1: Concept Matrix extraction prompt with parser"""
        parser = PydanticOutputParser(pydantic_object=ConceptMatrixOutput)
        
        prompt = PromptTemplate(
            template="""
            Analyze the following technical document and extract information for the Concept Matrix:

            Document: {input_text}

            Please fill in concise information for each component (1-2 short sentences):

            1. Problem/Purpose: What problem does this solve or what is the main objective?
            2. Object/System: What is the main object, device, or system being described?
            3. Action/Method: What actions, processes, or methods are performed?
            4. Key Technical Feature/Structure: What are the core technical features or structural elements?
            5. Environment/Field: What is the application domain or operating environment?
            6. Advantage/Result: What benefits or results are achieved?

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
            template="""
            From the following Concept Matrix, extract 1-3 distinctive technical keywords/phrases for each component.
            Focus on:
            - Technical nouns and specific terminology
            - Action verbs and processes
            - Avoid overly general terms like "system", "method", "device" without qualifiers
            - Prioritize terms that would be useful for patent search

            Concept Matrix:
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
            template="""
            Automatically refine and enhance the seed keywords based on the concept matrix and initial extraction.
            
            Original Concept Matrix:
            {concept_matrix}
            
            Current Keywords:
            {current_keywords}

            Please improve the keywords by:
            1. Ensuring technical specificity and distinctiveness
            2. Removing overly general terms
            3. Adding important technical terms that may have been missed
            4. Optimizing for patent search effectiveness
            5. Ensuring good coverage of all concept areas
            6. Maintaining 1-3 keywords per category

            Focus on:
            - Technical terminology that would appear in patent documents
            - Specific component names, processes, and methods
            - Industry-standard terminology
            - Terms that distinguish this invention from others

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
            template="""
            Improve the seed keywords based on user feedback:

            Current keywords:
            {current_keywords}

            User feedback: {feedback}

            Please refine the keywords to:
            1. Ensure sufficient distinctiveness and technical specificity
            2. Avoid overly general terms
            3. Add important missing technical concepts mentioned in feedback
            4. Optimize for patent search effectiveness
            5. Maintain 1-3 keywords per category

            {format_instructions}
            """,
            input_variables=["current_keywords", "feedback"],
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
class ConceptMatrixOutput(BaseModel):
    """Output model for Phase 1 concept extraction"""
    problem_purpose: str = Field(description="What problem does this solve or what is the main objective?")
    object_system: str = Field(description="What is the main object, device, or system being described?")
    action_method: str = Field(description="What actions, processes, or methods are performed?")
    key_technical_feature: str = Field(description="What are the core technical features or structural elements?")
    environment_field: str = Field(description="What is the application domain or operating environment?")
    advantage_result: str = Field(description="What benefits or results are achieved?")


class SeedKeywordsOutput(BaseModel):
    """Output model for Phase 2 and 3 keyword extraction"""
    problem_purpose: List[str] = Field(description="Keywords for problem/purpose")
    object_system: List[str] = Field(description="Keywords for object/system")
    action_method: List[str] = Field(description="Keywords for action/method")
    key_technical_feature: List[str] = Field(description="Keywords for key technical features")
    environment_field: List[str] = Field(description="Keywords for environment/field")
    advantage_result: List[str] = Field(description="Keywords for advantage/result")

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
