"""
Patent Keyword Extraction Prompt Templates
Contains all prompt templates and parsers for the patent keyword extraction system
"""

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict

# Data Models for Prompt Outputs
class ConceptMatrixOutput(BaseModel):
    """Output model for extracting core patent search concepts from technical documents"""
    problem_purpose: str = Field(
        description="The specific technical problem the invention aims to solve or the primary objective described in the document."
    )
    object_system: str = Field(
        description="The main object, device, system, material, or process that is the subject of the invention as stated in the document."
    )
    environment_field: str = Field(
        description="The application domain, industry sector, or operational context where the invention is intended to be used."
    )

class SeedKeywordsOutput(BaseModel):
    """Output model for Phase 2 and 3 keyword extraction (patent-specific fields only)"""
    problem_purpose: List[str] = Field(
        description="Distinctive technical keywords describing the technical problem addressed or primary objective."
    )
    object_system: List[str] = Field(
        description="Technical keywords specifying the main object, device, system, material, or process described."
    )
    environment_field: List[str] = Field(
        description="Keywords identifying the application domain, industry sector, or operational context."
    )

class SummaryResponse(BaseModel):
    summary: str = Field(
        description="If the patent idea description has been summarized, provide that summary."
    )

class QueriesResponse(BaseModel):
    queries: List[str] = Field(
        description="List of queries. Leave empty if none."
    )

class ExtractionPrompts:
    """Collection of prompt templates for patent keyword extraction"""
    
    @staticmethod
    def get_phase1_prompt_and_parser():
        """Phase 1: Concept Matrix extraction prompt and parser"""
        parser = PydanticOutputParser(pydantic_object=ConceptMatrixOutput)
        
        prompt = PromptTemplate(
            template="""<OBJECTIVE_AND_PERSONA>
You are a patent concept extraction specialist with extensive experience in identifying factual, structured insights from technical and patent-related documents for prior art search. Your task is to extract detailed, factual information for each component in the Concept Matrix from the provided document.
</OBJECTIVE_AND_PERSONA>

<INSTRUCTIONS>
To complete this task, you need to follow these steps:
1. Carefully read and analyze the provided technical document
2. Extract explicit information for each Concept Matrix component without making assumptions
3. Use only information explicitly stated in the document
4. If a component is missing or unspecified, respond exactly with: "Not mentioned."
5. Ensure each component contains unique, non-redundant information
6. Focus on domain-specific terminology as expressed in the document
</INSTRUCTIONS>

<CONSTRAINTS>
Dos:
- Focus on clear, domain-specific terminology as expressed in the document
- Each Concept Matrix component must contain unique, non-redundant information
- Maintain factual precision appropriate for patent analysis
- Use exact terminology found in the document
- Exclude overly generic or unrelated content

Don'ts:
- Do not make assumptions, paraphrasing, or inference beyond what is explicitly stated
- Do not repeat content between components
- Do not infer technical details not explicitly described
- Do not include generic terms without technical qualifiers
</CONSTRAINTS>

<CONTEXT>
The document provided contains technical information about an invention or patent idea. You need to extract three key components that form the Concept Matrix for patent search:

1. **Problem/Purpose** — The specific technical problem the invention addresses, or its primary objective
2. **Object/System** — The main object, device, system, material, or process being described
3. **Environment/Field** — The intended application domain, industry sector, or operational context

Document:
{input_text}
</CONTEXT>

<OUTPUT_FORMAT>
{format_instructions}
</OUTPUT_FORMAT>

<RECAP>
Extract factual information for each Concept Matrix component using only explicit information from the document. Maintain technical precision, avoid assumptions, and ensure each component contains unique information. IMPORTANT: Only generate the JSON output as defined - do not provide explanations, commentary, or any additional text beyond the required JSON format.
</RECAP>""",
            input_variables=["input_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        return prompt, parser
    
    @staticmethod
    def get_phase2_prompt_and_parser():
        """Phase 2: Seed keyword extraction prompt and parser"""
        parser = PydanticOutputParser(pydantic_object=SeedKeywordsOutput)
        
        prompt = PromptTemplate(
            template="""<OBJECTIVE_AND_PERSONA>
You are an expert patent search analyst specializing in extracting high-value, domain-specific technical keywords for prior art search and patent landscaping. Your task is to extract precise, distinctive, and high-discriminative technical keywords from the provided Concept Matrix components.
</OBJECTIVE_AND_PERSONA>

<INSTRUCTIONS>
To complete this task, you need to follow these steps:
1. Analyze each component of the provided Concept Matrix
2. Extract specific technical nouns and domain-specific industry terms
3. Prioritize discriminative terms highly relevant for patent search
4. Select the most technical and domain-preferred terms when multiple variants exist
5. Ensure each keyword is technically precise and highly distinctive
6. Create concise, unique lists for each component category
</INSTRUCTIONS>

<CONSTRAINTS>
Dos:
- Focus exclusively on specific technical nouns (e.g., "optical sensor", "convolutional")
- Include precise domain-specific industry terms
- Prioritize discriminative terms that narrow down or uniquely identify technical solutions
- Extract algorithm names, sensor types, material names, component names
- Use formal technical nomenclature from the relevant industry

Don'ts:
- Do not include general terms like "system", "method", or "device" unless explicitly modified with technical qualifiers
- Do not infer or generalize terms beyond those explicitly mentioned
- Do not include duplicate or synonymous terms across components
- Do not use verbose keyphrases unless they represent formal technical nomenclature
- Do not leave components populated if no relevant discriminative technical terms are available
</CONSTRAINTS>

<CONTEXT>
The following Concept Matrix has been extracted from a patent document:

- Problem/Purpose: {problem_purpose}
- Object/System: {object_system}
- Environment/Field: {environment_field}

A discriminative term is one that narrows down or uniquely identifies a technical solution or context, making it highly valuable for patent search precision.
</CONTEXT>

<OUTPUT_FORMAT>
{format_instructions}
</OUTPUT_FORMAT>

<RECAP>
Extract precise, distinctive technical keywords from each Concept Matrix component. Focus on discriminative terms, avoid generic language, ensure uniqueness across categories, and prioritize technical precision for effective patent search. IMPORTANT: Only generate the JSON output as defined - do not provide explanations, commentary, or any additional text beyond the required JSON format.
</RECAP>""",
            input_variables=["problem_purpose", "object_system", "environment_field"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        return prompt, parser
    
    @staticmethod
    def get_phase3_prompt_and_parser():
        """Phase 3: Keyword refinement with feedback prompt and parser"""
        parser = PydanticOutputParser(pydantic_object=SeedKeywordsOutput)
        
        prompt = PromptTemplate(
            template="""<OBJECTIVE_AND_PERSONA>
You are a technical keyword optimization expert specializing in patent search and prior art analysis. Your task is to refine existing seed keywords based on user feedback to improve their technical specificity, discriminative power, and relevance for patent prior art search.
</OBJECTIVE_AND_PERSONA>

<INSTRUCTIONS>
To complete this task, you need to follow these steps:
1. Carefully review the current keywords and user feedback
2. Identify and add any missing important technical terms explicitly mentioned in feedback
3. Remove overly generic, vague, or non-technical terms from current keywords
4. Ensure each keyword is technically precise and highly distinctive
5. Eliminate duplicate or synonymous terms within and across categories
6. Optimize the final keyword list for maximum discriminative value in patent search
</INSTRUCTIONS>

<CONSTRAINTS>
Dos:
- Prioritize technical terms such as algorithm names, sensor types, material names, component names
- Focus on domain-specific terminology relevant to patent search
- Prefer highly specific, uncommon, and industry-recognized terms
- Ensure technical precision and high distinctiveness for each keyword
- Focus only on Problem/Purpose, Object/System, and Environment/Field categories

Don'ts:
- Do not include terms that closely resemble or duplicate existing ones in meaning
- Do not retain overly generic, vague, or non-technical terms
- Do not ignore explicit technical terms mentioned in user feedback
- Do not include synonymous terms across different categories
- Do not compromise discriminative value for quantity
</CONSTRAINTS>

<CONTEXT>
Current keywords that need refinement:
{current_keywords}

User feedback providing guidance for improvement:
{feedback}

The goal is to optimize these keywords for maximum effectiveness in patent prior art search, ensuring each term contributes unique discriminative value.
</CONTEXT>

<OUTPUT_FORMAT>
{format_instructions}
</OUTPUT_FORMAT>

<RECAP>
Refine the seed keywords based on user feedback by adding missing technical terms, removing generic terms, ensuring technical precision, eliminating duplicates, and optimizing for maximum discriminative value in patent search. IMPORTANT: Only generate the JSON output as defined - do not provide explanations, commentary, or any additional text beyond the required JSON format.
</RECAP>""",
            input_variables=["current_keywords", "feedback"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        return prompt, parser

    @staticmethod
    def get_summary_prompt_and_parser():
        """Summary generation prompt and parser"""
        parser = PydanticOutputParser(pydantic_object=SummaryResponse)
        
        prompt = PromptTemplate(
            template="""<OBJECTIVE_AND_PERSONA>
You are an expert patent abstracter with extensive experience in technical documentation. Your task is to read and understand a detailed patent idea description and generate a concise, objective summary not exceeding 400 words, focusing exclusively on technical aspects of the invention.
</OBJECTIVE_AND_PERSONA>

<INSTRUCTIONS>
To complete this task, you need to follow these steps:
1. Read and analyze the provided patent idea description thoroughly
2. Identify the core technical features of the invention
3. Extract the core technical function or purpose
4. Describe the basic structure or components
5. Explain the principle of operation or method of implementation
6. Specify technical applications within the relevant technical field
7. Write a coherent, scientific summary within the 400-word limit
</INSTRUCTIONS>

<CONSTRAINTS>
Dos:
- Focus exclusively on technical aspects of the invention
- Use complete, coherent sentences with clear structure
- Maintain scientific, objective, and professional language
- Include enough detail for technical understanding without full description
- Specify technical applications (e.g., "water filtration device utilizing nano-membranes")
- Ensure total word count does not exceed 400 words

Don'ts:
- Do not include business objectives, user benefits, or non-technical elements
- Do not use keyword listing or fragmented phrases
- Do not exceed the 400-word limit
- Do not provide explanations beyond the technical summary
- Do not use generic terms without technical context (avoid "water solution", use specific technical terms)
</CONSTRAINTS>

<CONTEXT>
Patent idea description to be summarized:
{idea}

The summary should serve as a technical abstract that captures the essence of the invention for patent analysis and prior art search purposes.
</CONTEXT>

<OUTPUT_FORMAT>
{format_instructions}
</OUTPUT_FORMAT>

<RECAP>
Generate a concise technical summary (max 400 words) focusing on core technical features, structure, operation principles, and specific applications. Use scientific language, maintain objectivity, and exclude non-technical elements. IMPORTANT: Only generate the JSON output as defined - do not provide explanations, commentary, or any additional text beyond the required JSON format.
</RECAP>""",
            input_variables=["idea"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        return prompt, parser
    
    @staticmethod
    def get_queries_prompt_and_parser():
        """Query generation prompt and parser"""
        parser = PydanticOutputParser(pydantic_object=QueriesResponse)
        
        prompt = PromptTemplate(
            template="""<OBJECTIVE_AND_PERSONA>
You are an expert patent searcher with many years of experience, proficient in using Boolean operators (AND, OR, NOT), proximity operators (NEAR/x, ADJ/x), and complex search syntax on databases like Espacenet, Google Patents, and USPTO. Your task is to build comprehensive patent search queries to assess novelty and inventive step for an invention.
</OBJECTIVE_AND_PERSONA>

<INSTRUCTIONS>
To complete this task, you need to follow these steps:
1. Analyze the provided invention context and concept groups
2. Review the CPC codes for classification context
3. Generate 5 search query strings following strategies from broad to narrow
4. Use common syntax applicable across multiple patent database platforms
5. Apply the three specified search strategies with appropriate logic
6. Provide query strings and logic explanations for each strategy
</INSTRUCTIONS>

<CONSTRAINTS>
Dos:
- Use Boolean operators (AND, OR, NOT) and proximity operators (NEAR/x, ADJ/x) effectively
- Create queries compatible with multiple patent database platforms
- Follow the three strategy approaches: Broad, Focused, and Advanced Proximity
- Provide raw query strings in code blocks for easy copying
- Include clear logic explanations for each query structure
- Ensure queries progress from broad to narrow search scope

Don'ts:
- Do not create queries incompatible with major patent databases
- Do not omit logic explanations for the query structures
- Do not deviate from the three specified search strategies
- Do not create queries that are too narrow to find relevant prior art
- Do not ignore the CPC classification codes in query construction
</CONSTRAINTS>

<CONTEXT>
Invention relates to: {summary}

Key Concept Groups (Core concepts):
- Concept A (problem purpose): {problem_purpose_keys}
- Concept B (object system): {object_system_keys}  
- Concept C (environment field): {environment_field_keys}

CPC (Cooperative Patent Classification) Codes:
- Primary CPCs: {CPC_CODES}

Strategy 1 (Broad Query): Wide search capturing all potentially relevant documents, accepting some noise. Combine core concepts with CPC using OR logic.

Strategy 2 (Focused Query): High-precision search requiring keyword elements AND CPC classification. Combine all concept groups with AND, then AND with CPC codes.

Strategy 3 (Advanced Proximity Query): Increased precision using proximity operators for closely related terms instead of simple AND operators.
</CONTEXT>

<OUTPUT_FORMAT>
{format_instructions}
</OUTPUT_FORMAT>

<RECAP>
Generate 5 comprehensive patent search queries using the three specified strategies (Broad, Focused, Advanced Proximity) with Boolean and proximity operators. Provide raw query strings and logic explanations for each strategy to assess invention novelty and inventive step. IMPORTANT: Only generate the JSON output as defined - do not provide explanations, commentary, or any additional text beyond the required JSON format.
</RECAP>""",
            input_variables=["summary", "problem_purpose_keys", "object_system_keys", "environment_field_keys", "CPC_CODES"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        return prompt, parser

    @staticmethod
    def get_validation_messages():
        """Get validation messages for user interaction"""
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
        """Get phase completion messages"""
        return {
            "phase1_completed": "Phase 1 completed: Concept Matrix extracted",
            "phase2_completed": "Phase 2 completed: Seed keywords extracted", 
            "phase3_completed": "Phase 3 completed: Keywords refined",
            "extraction_completed": "✅ Patent seed keyword extraction completed"
        }
