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
1. The main problem or challenge explicitly described.
2. The core technical solution, method, or approach explicitly proposed.
Your extraction must preserve maximum fidelity to the input, capturing all explicit technical details, constraints, and context without omission.
</OBJECTIVE_AND_PERSONA>

<INSTRUCTIONS>
1. Carefully read the input idea in full.
2. For the "problem":
   - Identify only what is explicitly described as the problem, challenge, limitation, or requirement.
   - Include all explicit technical constraints, requirements, and context.
   - If no problem is mentioned, output exactly: "Not mentioned."
3. For the "technical":
   - Identify only the explicitly proposed technical solution, method, or approach.
   - Include all explicit technical details, mechanisms, components, steps, and context.
   - If no technical solution is mentioned, output exactly: "Not mentioned."
4. Use direct quotations from the input whenever possible. If necessary for readability, you may minimally reassemble fragmented text, but do not infer or add unstated information.
5. If multiple distinct explicit details are present, include all of them clearly.
6. Do not generalize, summarize, or explain beyond what is explicitly written in the input.
</INSTRUCTIONS>

<CONTEXT>
Input idea:
{input}
</CONTEXT>

<OUTPUT_FORMAT>
{format_instructions}
</OUTPUT_FORMAT>

<RECAP>
Extract and return only the JSON output with exactly two fields: "problem" and "technical". Each field must include every explicit detail from the input. If a field is not mentioned, use "Not mentioned." Do not add any extra explanation or text outside the JSON.
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
You are a patent concept extraction specialist with expertise in identifying precise, factual, and structured insights from technical and patent documents for prior art search. Your role is to extract only what is explicitly written, ensuring maximum fidelity to the source text.
</OBJECTIVE_AND_PERSONA>

<INSTRUCTIONS>
1. Read and analyze the "Document" section in full without skipping any part.
2. For each Concept Matrix field, extract only explicit information stated in the text, preserving exact terminology and context.
3. Do not infer, generalize, summarize, or rephrase beyond the explicit wording of the document.
4. Ensure each field contains unique, non-overlapping content; do not duplicate or recycle text across fields.
5. If a Concept Matrix field is not explicitly mentioned in the document, output exactly: "Not mentioned."
</INSTRUCTIONS>

<CONCEPT MATRIX FIELDS>
1. Problem/Purpose — The specific technical problem, limitation, or primary objective the invention addresses, as explicitly stated in the document.
2. Object/System — The main object, device, system, material, or process described in the document, using the exact terminology provided.
3. Environment/Field — The intended application domain, industry sector, or operational context in which the invention is designed to be used, as explicitly stated in the document.
</CONCEPT MATRIX FIELDS>

<CONSTRAINTS>
- Use only terminology, context, and descriptions exactly as written in the provided text.
- Do not infer, add unstated details, or use synonyms.
- Each Concept Matrix field must be strictly unique and non-redundant relative to the others.
- If no explicit information is available for a field, return "Not mentioned."
</CONSTRAINTS>

<CONTEXT>
Document:
{problem}

**Concept Matrix:**  
1. **Problem/Purpose** — The specific technical problem, limitation, or objective addressed.  
2. **Object/System** — The main object, device, system, material, or process described.  
3. **Environment/Field** — The intended application domain, industry sector, or operational context.
</CONTEXT>

<OUTPUT_FORMAT>
{format_instructions}
</OUTPUT_FORMAT>

<RECAP>
Return only the JSON output. Each must contain only explicit details from the input document. If a field is missing, return "Not mentioned." Do not include explanations, comments, or text outside of the JSON.
</RECAP>
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
1. Read each Concept Matrix component in full.
2. Extract only explicit technical nouns, formal nomenclature, or industry-specific terminology exactly as written.
3. Select discriminative terms that clearly identify the technical solution, material, or context.
4. Each keyword must be concise: strictly 1–2 words.
5. Provide unique, non-overlapping keywords for each component.
</INSTRUCTIONS>

<CONSTRAINTS>
Do:
- Use terminology exactly as stated in the text, without modification.
- Keep every keyword short: maximum 1–2 words.
- Include algorithm names, sensor types, material names, standards, or component names.
- Ensure keywords are unique across components and do not repeat.

Don’t:
- Do not include generic terms
- Do not create keywords longer than 3 words.
- Do not infer, summarize, or invent keywords not explicitly mentioned.
- Do not duplicate or reuse keywords across multiple components.
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
Extract explicit, precise, discriminative keywords (strictly 1–2 words) for each Concept Matrix component. Ensure uniqueness across components. If a field has no explicit keyword, return "Not mentioned." Output strictly in the defined JSON format with no explanations or extra text.
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
You are an expert prior art patent searcher with extensive experience in novelty and inventive-step assessment. Your role is to construct Boolean patent search queries that optimize recall and precision, progressively moving from broad coverage to highly discriminative targeting.
</OBJECTIVE_AND_PERSONA>

<INSTRUCTIONS>
1. Read the invention context and all three key concept groups carefully.  
2. Review CPC codes to understand the classification scope and incorporate them into every query.  
3. From each concept group, extract the 2–3 most explicit, technical, and discriminative keywords (combine with OR).  
4. Construct exactly 6 queries:  
   - 2 Broad queries  
   - 2 Focused queries  
   - 2 Narrow queries  
5. Apply strict Boolean logic using only AND, OR, NOT, and parentheses.  
6. Keep each query concise, with 8–10 unique keywords maximum.  
7. Do not paraphrase or infer; only use explicit keywords and CPC codes provided.  
8. Use parentheses for all OR combinations and to enforce correct operator precedence.  
9. If fewer than 2–3 explicit keywords are available in a concept group, use only those given (do not invent).  
</INSTRUCTIONS>

<CONSTRAINTS>
Do:  
- Use Boolean operators only: AND, OR, NOT, parentheses.  
- Incorporate CPC codes in every query.  
- Maintain strict technical specificity and discriminative terminology.  
- Ensure each query is unique and consistent with its assigned strategy (Broad, Focused, Narrow).  

Don't:  
- Do not exceed 10 unique keywords per query.  
- Do not omit CPC codes when provided.  
- Do not use generic or non-technical words.  
- Do not add explanations, commentary, or reformulations of the invention context.  
- Do not duplicate identical queries across strategies.  
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
- Strategy 1 (Broad Search): Wider coverage using representative keywords from all concept groups + CPC codes.  
- Strategy 2 (Focused Search): Balanced recall and precision using more specific, discriminative terms + CPC codes.  
- Strategy 3 (Narrow / Precision Search): Highest precision using only the most specific keywords with strict AND logic + CPC codes.  
</CONTEXT>

<OUTPUT_FORMAT>
{format_instructions}
</OUTPUT_FORMAT>

<RECAP>
Return exactly 6 patent search queries in JSON format. Each query must follow Boolean syntax strictly, include CPC codes, and respect keyword limits. Do not add any explanations, notes, or text outside the JSON.
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
