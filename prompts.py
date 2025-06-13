"""
Prompt templates for patent seed keyword extraction system
"""

from langchain.prompts import PromptTemplate


class ExtractionPrompts:
    """Collection of prompt templates for the 3-phase extraction process"""
    
    @staticmethod
    def get_phase1_prompt() -> PromptTemplate:
        """Phase 1: Concept Matrix extraction prompt"""
        return PromptTemplate.from_template("""
        Analyze the following technical document and extract information for the Concept Matrix:

        Document: {input_text}

        Please fill in concise information for each component (1-2 short sentences):

        1. Problem/Purpose: What problem does this solve or what is the main objective?
        2. Object/System: What is the main object, device, or system being described?
        3. Action/Method: What actions, processes, or methods are performed?
        4. Key Technical Feature/Structure: What are the core technical features or structural elements?
        5. Environment/Field: What is the application domain or operating environment?
        6. Advantage/Result: What benefits or results are achieved?

        Return in JSON format with keys: problem_purpose, object_system, action_method, key_technical_feature, environment_field, advantage_result
        
        Example:
        {{
            "problem_purpose": "Reduce water waste in irrigation systems",
            "object_system": "Smart irrigation control system",
            "action_method": "Automated scheduling and moisture monitoring",
            "key_technical_feature": "Soil moisture sensors and weather data integration",
            "environment_field": "Agriculture and gardening applications",
            "advantage_result": "30% water savings and optimized plant care"
        }}
        """)
    
    @staticmethod
    def get_phase2_prompt() -> PromptTemplate:
        """Phase 2: Seed keyword extraction prompt"""
        return PromptTemplate.from_template("""
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

        Return in JSON format with each component as an array of keywords:
        {{
            "problem_purpose": ["keyword1", "keyword2"],
            "object_system": ["keyword1"],
            "action_method": ["keyword1", "keyword2"],
            "key_technical_feature": ["keyword1", "keyword2", "keyword3"],
            "environment_field": ["keyword1"],
            "advantage_result": ["keyword1", "keyword2"]
        }}
        
        Example:
        {{
            "problem_purpose": ["water conservation", "irrigation optimization"],
            "object_system": ["smart irrigation system", "automated controller"],
            "action_method": ["automatic scheduling", "moisture monitoring"],
            "key_technical_feature": ["soil moisture sensor", "weather data", "IoT integration"],
            "environment_field": ["agriculture", "precision farming"],
            "advantage_result": ["water savings", "crop optimization"]
        }}
        """)
    
    @staticmethod
    def get_phase3_auto_prompt() -> PromptTemplate:
        """Phase 3: Automatic refinement prompt"""
        return PromptTemplate.from_template("""
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

        Return the refined keywords in the same JSON format:
        {{
            "problem_purpose": ["refined_keyword1", "refined_keyword2"],
            "object_system": ["refined_keyword1"],
            "action_method": ["refined_keyword1", "refined_keyword2"],
            "key_technical_feature": ["refined_keyword1", "refined_keyword2", "refined_keyword3"],
            "environment_field": ["refined_keyword1"],
            "advantage_result": ["refined_keyword1", "refined_keyword2"]
        }}
        """)
    
    @staticmethod
    def get_phase3_prompt() -> PromptTemplate:
        """Phase 3: Manual refinement prompt (deprecated - kept for compatibility)"""
        return PromptTemplate.from_template("""
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

        Return in the same JSON format as before with improved keywords.
        """)
    
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
