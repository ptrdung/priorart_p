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
    """Output model for patent idea summary"""
    summary: str = Field(
        description="If the patent idea description has been summarized, provide that summary."
    )

class QueriesResponse(BaseModel):
    """Output model for patent search queries"""
    queries: List[str] = Field(
        description="List of queries. Leave empty if none."
    )

class NormalizationOutput(BaseModel):
    """Output model for normalization: extract problem and technical points from input"""
    problem: str = Field(
        description="The main problem or challenge described in the input idea."
    )
    technical: str = Field(
        description="The core technical solution, method, or approach described in the input idea."
    )

class ExtractionPrompts:
    """Collection of prompt templates for patent keyword extraction"""

    @staticmethod
    def get_normalization_prompt_and_parser():
        """Prompt and parser for normalizing input and extracting problem/technical points"""
        parser = PydanticOutputParser(pydantic_object=NormalizationOutput)
        prompt = PromptTemplate(
            template="""<OBJECTIVE_AND_PERSONA>
You are a patent analyst specializing in technical idea normalization. Your task is to read the provided input and extract two main points:
1. The main problem or challenge described.
2. The core technical solution, method, or approach proposed.
Your extraction must be as detailed and specific as possible, capturing all explicit technical details, constraints, and context present in the input.
</OBJECTIVE_AND_PERSONA>

<INSTRUCTIONS>
1. Carefully read the input idea in full.
2. Identify and clearly state the main problem or challenge, including any technical constraints, requirements, or context explicitly mentioned.
3. Identify and clearly state the core technical solution, method, or approach, including all relevant technical details, mechanisms, steps, and context provided.
4. Use only explicit information from the input; do not infer, generalize, or paraphrase.
5. If a point is missing, respond exactly with: "Not mentioned."
6. If multiple explicit details are present, include all of them in a concise, structured manner.
7. Prefer direct quotations or closely paraphrased phrases from the input for maximum fidelity.
</INSTRUCTIONS>

<CONTEXT>
Input idea:
{input}
</CONTEXT>

<OUTPUT_FORMAT>
{format_instructions}
</OUTPUT_FORMAT>

<RECAP>
Extract and return only the JSON output with two fields: "problem" and "technical". Each field should be as detailed as possible, capturing all explicit technical details and context from the input. Do not add explanations or extra text.
</RECAP>
""",
            input_variables=["input"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        return prompt, parser

    @staticmethod
    def get_phase1_prompt_and_parser():
        """Phase 1: Concept Matrix extraction prompt and parser"""
        parser = PydanticOutputParser(pydantic_object=ConceptMatrixOutput)
        
        prompt = PromptTemplate(
            template="""<OBJECTIVE_AND_PERSONA>
You are a patent concept extraction specialist skilled in identifying factual, structured insights from technical and patent documents for prior art search.
</OBJECTIVE_AND_PERSONA>

<INSTRUCTIONS>
1. Read and analyze the "Document" sections in full.
2. For each Concept Matrix component, extract explicit information verbatim—copy exact phrases as they appear.
3. Do not introduce new interpretations, synonyms, or background knowledge not directly present in the Document.
4. Ensure each component is unique and non-overlapping across the matrix; do not duplicate the same phrase in multiple components.
</INSTRUCTIONS>

<CONSTRAINTS>
- Use only domain-specific terminology and descriptive context exactly as stated in the provided text.
- Explanations are allowed only if they are explicitly stated in the source (Document).
- Each Concept Matrix component must be unique and non-redundant relative to all other components.
- Do not repeat content between components.
- Do not infer, generalize, or hallucinate beyond explicit statements.
</CONSTRAINTS>

<CONTEXT>
Document:
{problem}
</CONTEXT>

<OUTPUT_FORMAT>
{format_instructions}
</OUTPUT_FORMAT>

<RECAP>
Extract explicit terms and, where available, their directly stated descriptive context. Output strictly in the defined JSON format—no explanations, no extra text, and no code fences.
</RECAP>ss
""",
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
You are an expert patent search analyst. Your task is to extract domain-specific, high-value technical keywords from Concept Matrix components. The extracted keywords must maximize discriminative power, technical precision, and search relevance for prior art analysis.
</OBJECTIVE_AND_PERSONA>

<INSTRUCTIONS>
1. Read each Concept Matrix component carefully.
2. Extract only explicit technical nouns, industry terms, or formal nomenclature.
3. Select discriminative terms that uniquely identify the technical solution or context.
4. Each keyword must be concise: strictly 1–2 words only.
5. Provide distinct, non-redundant keywords for each component.
</INSTRUCTIONS>

<CONSTRAINTS>
Dos:
- Use domain-specific terminology directly from the text.
- Keep every keyword short: maximum 1–2 words.
- Include algorithm names, sensor types, material names, or component names.
- Ensure keywords are unique across components.

Don’ts:
- Do not include generic words such as “system”, “method”, or “device” unless explicitly modified by technical qualifiers.
- Do not create keywords longer than 3 words.
- Do not infer or generalize terms beyond explicit mentions.
- Do not duplicate or reuse keywords across components.
</CONSTRAINTS>

<CONTEXT>
Concept Matrix extracted from patent document:

- Problem/Purpose: {problem_purpose}
- Object/System: {object_system}
- Environment/Field: {environment_field}

Feedback:
{feedback}
</CONTEXT>

<OUTPUT_FORMAT>
{format_instructions}
</OUTPUT_FORMAT>

<RECAP>
Extract explicit, precise, discriminative keywords (1–2 words only) for each Concept Matrix component. Ensure uniqueness across components. Output strictly in the defined JSON format without explanations or extra text.
</RECAP>
""",
            input_variables=["problem_purpose", "object_system", "environment_field", "feedback"],
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
You are an expert prior art patent searcher with extensive experience in novelty and inventive-step assessment. Your role is to construct search queries that balance recall and precision, moving from broad coverage to highly discriminative targeting.
</OBJECTIVE_AND_PERSONA>

<INSTRUCTIONS>
1. Carefully analyze the provided invention context and the three key concept groups.  
2. Review CPC codes to understand the classification scope and leverage them to refine queries.  
3. Extract the most technically discriminative and essential 2–3 keywords per concept group (use OR operator).  
4. Construct exactly 6 concise. Two queries per strategy: Broad, Focused, Narrow.   
5. Apply strict Boolean logic without paraphrasing or redundancy.  
6. Use parentheses for clarity in OR combinations and to control operator precedence.  

</INSTRUCTIONS>

<CONSTRAINTS>
Dos:
- Use ONLY Boolean operators (AND, OR, NOT, parentheses).  
- Limit to 8–10 unique keywords per query.  
- Maintain strict technical specificity; prioritize discriminative terminology over breadth.  
- Incorporate CPC codes directly into queries.   

Don'ts:
- Do not deviate from the three strategy categories (Broad, Focused, Narrow).  
- Do not exceed keyword limits with verbose phrasing.  
- Do not omit CPC codes when they are provided.  
- Do not introduce generic or non-technical terms.  
- Do not add commentary, explanations, or rephrasing of the invention context.  

</CONSTRAINTS>

<CONTEXT>
Invention relates to: {problem}

Key Concept Groups:
- Problem purpose: {problem_purpose_keys}  
- Object system: {object_system_keys}  
- Environment field: {environment_field_keys}  

CPC (Cooperative Patent Classification) Codes:
- Primary CPCs: {CPC_CODES}  

QUERY CONSTRUCTION STRATEGIES:
- Strategy 1 (Broad Search): Use a wider combination of representative keywords and CPC codes to maximize coverage.  
- Strategy 2 (Focused Search): Use more specific and discriminative terms with primary CPC codes to balance recall and precision.  
- Strategy 3 (Narrow / Precision Search): Use only the most specific terms with strict AND logic and primary CPC codes for high precision.  
</CONTEXT>

<OUTPUT_FORMAT>
{format_instructions}
</OUTPUT_FORMAT>

<RECAP>
Output exactly 6 patent search queries in JSON format. Use only Boolean operators (AND, OR, NOT). No explanations, notes, or additional text.  
</RECAP>
""",
            input_variables=["problem", "problem_purpose_keys", "object_system_keys", "environment_field_keys", "CPC_CODES"],
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
