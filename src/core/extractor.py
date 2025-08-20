"""
Core Patent Concept Extractor
Main AI agent class for patent seed keyword extraction system
"""

import json
import datetime
import os
import logging
from typing import Any, Dict, List, Literal, Optional, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from typing import Dict, List, TypedDict, Annotated, Optional
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from langchain_tavily import TavilySearch
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
import requests
import time

# Configure logging
log_filename = f"patent_extractor_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import settings from config
from config.settings import settings

# Local imports with updated paths
from ..api.ipc_classifier import get_ipc_predictions
from ..prompts.extraction_prompts import ExtractionPrompts
from ..crawling.patent_crawler import lay_thong_tin_patent
from ..evaluation.similarity_evaluator import (
    eval_url, prompt, parse_idea_text, parse_idea_input, extract_user_info
)

# Set up Tavily API key from settings
os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY

# Data Models
class NormalizationOutput(BaseModel):
    """Output model for input normalization"""
    problem: str = Field(
        description="Normalized technical problem or objective described in the document."
    )
    technical: str = Field(
        description="Normalized technical content or context of the document."
    )

class ConceptMatrix(BaseModel):
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

class SeedKeywords(BaseModel):
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

class ValidationFeedback(BaseModel):
    """User validation feedback"""
    action: str  # "approve", "edit", "reject"
    edited_keywords: Optional[SeedKeywords] = None
    feedback: Optional[str] = None

class ReflectionEvaluation(BaseModel):
    """Reflection evaluation of keywords"""
    overall_quality: str = Field(description="Overall quality assessment: 'good' or 'poor'")
    keyword_scores: Dict[str, float] = Field(description="Score for each category (0-1)")
    issues_found: List[str] = Field(description="List of specific issues identified")
    recommendations: List[str] = Field(description="Recommendations for improvement")
    should_regenerate: bool = Field(description="Whether keywords should be regenerated")

class ExtractionState(TypedDict):
    """Simplified state for LangGraph workflow"""
    input_text: str
    problem: Optional[str]
    technical: Optional[str]
    summary_text: str
    ipcs: Any 
    concept_matrix: Optional[ConceptMatrix]
    seed_keywords: Optional[SeedKeywords]
    validation_feedback: Optional[ValidationFeedback]
    final_keywords: dict
    queries: list
    final_url: list

class CoreConceptExtractor:
    """Patent seed keyword extraction system"""
    
    def __init__(self, model_name: str = None, use_checkpointer: bool = None):
        """
        Initialize the CoreConceptExtractor.
        
        Args:
            model_name: Name of the LLM model to use
            use_checkpointer: Whether to use checkpointer for graph state
        """
        # Use settings from config file with fallback to parameters
        self.model_name = model_name if model_name is not None else settings.DEFAULT_MODEL_NAME
        self.use_checkpointer = use_checkpointer if use_checkpointer is not None else settings.USE_CHECKPOINTER
        
        self.llm = Ollama(model=self.model_name, temperature=settings.MODEL_TEMPERATURE)
        self.tavily_search = TavilySearch(
            max_results=settings.MAX_SEARCH_RESULTS,
            topic="general",
            include_answer=True,
            include_raw_content=False,
            include_images=False,
            include_image_descriptions=False,
        )
        self.prompts = ExtractionPrompts()
        self.messages = ExtractionPrompts.get_phase_completion_messages()
        self.validation_messages = ExtractionPrompts.get_validation_messages()
        self.use_checkpointer = use_checkpointer
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build simplified LangGraph workflow"""
        workflow = StateGraph(ExtractionState)
        
        # Add nodes for simplified 3-step process
        workflow.add_node("input_normalization", self.input_normalization)
        workflow.add_node("step0", self.step0)
        workflow.add_node("step1_concept_extraction", self.step1_concept_extraction)
        workflow.add_node("step2_keyword_generation", self.step2_keyword_generation)
        workflow.add_node("step3_human_evaluation", self.step3_human_evaluation)
        workflow.add_node("manual_editing", self.manual_editing)
        workflow.add_node("gen_key", self.gen_key)
        workflow.add_node("summary_prompt_and_parser", self.summary_prompt_and_parser)
        workflow.add_node("call_ipcs_api", self.call_ipcs_api)
        workflow.add_node("genQuery", self.genQuery)
        workflow.add_node("genUrl", self.genUrl)
        workflow.add_node("evalUrl", self.evalUrl)

        # Define simplified flow
        workflow.set_entry_point("input_normalization")
        workflow.add_edge("input_normalization", "step0")
        workflow.add_edge("step0", "step1_concept_extraction")
        workflow.add_edge("step0", "summary_prompt_and_parser")
        workflow.add_edge("step1_concept_extraction", "step2_keyword_generation")
        workflow.add_edge("step2_keyword_generation", "step3_human_evaluation")

        workflow.add_edge("summary_prompt_and_parser", "call_ipcs_api")
        
        # Conditional edge from human evaluation
        workflow.add_conditional_edges(
            "step3_human_evaluation",
            self._get_human_action,
            {
                "approve": "gen_key",
                "reject": "step1_concept_extraction", 
                "edit": "manual_editing"
            }
        )
        
        workflow.add_edge("manual_editing", "gen_key")
        workflow.add_edge("gen_key", "genQuery")
        workflow.add_edge("genQuery", "genUrl")
        workflow.add_edge("genUrl", "evalUrl")

        return workflow.compile()
    
    def extract_keywords(self, input_text: str) -> Dict:
        """Run the simplified 3-step keyword extraction workflow"""
        initial_state = ExtractionState(
            input_text=input_text,
            problem=None,
            technical=None,
            concept_matrix=None,
            seed_keywords=None,
            validation_feedback=None,
            final_keywords=None,
            ipcs=None,
            summary_text=None,
            queries=None,
            final_url=None
        )
        
        if self.use_checkpointer:
            config = {"configurable": {"thread_id": settings.THREAD_ID}}
            result = self.graph.invoke(initial_state, config)
        else:
            result = self.graph.invoke(initial_state)
        
        # Return all ExtractionState fields
        return dict(result)
        
    def input_normalization(self, state: ExtractionState) -> ExtractionState:
        """Normalize and clean input text before processing"""    
        # Get normalization prompt and parser from ExtractionPrompts
        prompt, parser = self.prompts.get_normalization_prompt_and_parser()
        response = self.llm.invoke(prompt.format(input=state["input_text"]))

        try:
            normalized_data = parser.parse(response)
            normalized_input = NormalizationOutput(**normalized_data.dict())
            logger.info("Normalization completed.")

            updated_state = {
                "problem": normalized_input.problem,
                "technical": normalized_input.technical,
                "input_text": state["input_text"]
            }
            logger.info(f"📝 Normalized problem: {normalized_input.problem}")
            logger.info(f"📝 Normalized technical: {normalized_input.technical}")
            return updated_state

        except Exception as e:
            logger.error(f"⚠️ Normalization parsing failed: {e}, using original input")
            fallback_normalized = NormalizationOutput(
                problem="Not mentioned.",
                technical="Not mentioned."
            )
            return {
                "problem": "Not mentioned.",
                "technical": "Not mentioned.",
                "input_text": state["input_text"]
            }

    def step0(self, state: ExtractionState) -> ExtractionState:
        """Initial step - pass through state"""
        return state

    def step1_concept_extraction(self, state: ExtractionState) -> ExtractionState:
        """Step 1: Extract concept summary from document according to fields"""
        # Use normalized problem for concept extraction if available

        prompt, parser = self.prompts.get_phase1_prompt_and_parser()
        response = self.llm.invoke(prompt.format(problem=state["problem"]))
        
        try:
            concept_data = parser.parse(response)
            concept_matrix = ConceptMatrix(**concept_data.dict())
        except Exception as e:
            logger.warning(f"Parser failed: {e}, falling back to manual parsing")
            concept_matrix = self._parse_concept_response(response)
        
        return {"concept_matrix": concept_matrix}

    def step2_keyword_generation(self, state: ExtractionState) -> ExtractionState:
        """Step 2: Generate main keywords for each field from summary"""
        concept_matrix = state["concept_matrix"]
        feedback = ""
        if state.get("validation_feedback") and getattr(state["validation_feedback"], "feedback", None):
            feedback = state["validation_feedback"].feedback

        prompt, parser = self.prompts.get_phase2_prompt_and_parser()
        response = self.llm.invoke(prompt.format(
            problem_purpose=concept_matrix.problem_purpose,
            object_system=concept_matrix.object_system,
            environment_field=concept_matrix.environment_field,
            feedback=feedback
        ))
        
        try:
            keyword_data = parser.parse(response)
            seed_keywords = SeedKeywords(**keyword_data.dict())
        except Exception as e:
            logger.warning(f"Parser failed: {e}, falling back to manual parsing")
            seed_keywords = self._parse_keyword_response(response)
        
        return {"seed_keywords": seed_keywords}
    
    def step3_human_evaluation(self, state: ExtractionState, action: str = None, feedback_text: str = None, edited_keywords: SeedKeywords = None) -> ExtractionState:
        """
        Step 3: Human in the loop evaluation with three options.
        Now supports web input via parameters:
        - action: "approve", "reject", "edit"
        - feedback_text: optional feedback for "reject"
        - edited_keywords: optional SeedKeywords for "edit"
        """
        msgs = self.validation_messages

        concept_matrix = state["concept_matrix"]
        seed_keywords = state["seed_keywords"]

        # If no action provided, default to approve
        if action is None:
            action = "approve"

        if action == "approve":
            feedback = ValidationFeedback(action="approve")
        elif action == "reject":
            feedback = ValidationFeedback(action="reject", feedback=feedback_text)
            state["concept_matrix"] = None
            state["seed_keywords"] = None
        elif action == "edit":
            feedback = ValidationFeedback(action="edit", edited_keywords=edited_keywords)
        else:
            feedback = ValidationFeedback(action="approve")

        state["validation_feedback"] = feedback

        return {"validation_feedback": feedback}

    def manual_editing(self, state: ExtractionState) -> ExtractionState:
        """Allow user to manually edit keywords"""
        feedback = state["validation_feedback"]
        
        if feedback.edited_keywords:
            state["seed_keywords"] = feedback.edited_keywords
        
        return {"seed_keywords": feedback.edited_keywords}
    
    def _get_manual_edits(self, current_keywords: SeedKeywords, edited_data: dict = None) -> ValidationFeedback:
        """
        Get manual edits from user.
        Now supports web input via edited_data dict:
        - edited_data: dict with keys matching SeedKeywords fields, values are lists of strings.
        """
        logger.info("📝 Manual Editing Mode")
        if edited_data is None:
            # If no edits provided, keep current keywords
            edited_keywords = current_keywords
        else:
            edited_keywords = SeedKeywords(**edited_data)
        return ValidationFeedback(action="edit", edited_keywords=edited_keywords)
      
    def _parse_concept_response(self, response: str) -> ConceptMatrix:
        """Parse response when JSON parsing fails"""
        lines = response.strip().split('\n')
        data = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_').replace('/', '_')
                if 'problem' in key or 'purpose' in key:
                    data['problem_purpose'] = value.strip()
                elif 'object' in key or 'system' in key:
                    data['object_system'] = value.strip()
                elif 'environment' in key or 'field' in key:
                    data['environment_field'] = value.strip()
        
        return ConceptMatrix(**data)
    
    def _parse_keyword_response(self, response: str) -> SeedKeywords:
        """Parse keyword response when JSON parsing fails"""
        # Fallback parsing logic
        return SeedKeywords(
            problem_purpose=["extracted_keyword"],
            object_system=["extracted_keyword"],
            environment_field=["extracted_keyword"],
        )
    
    def _get_human_action(self, state: ExtractionState) -> str:
        """Get the human action from validation feedback"""
        feedback = state["validation_feedback"]
        return feedback.action if feedback else "approve"
    
    def gen_key(self, state: ExtractionState) -> ExtractionState:
        """Generate synonyms and related terms for keywords"""
        def search_snippets(keyword: str, max_snippets: int = 3) -> List[str]:
            results = self.tavily_search.invoke({"query": keyword})
            snippets = [r['content'] for r in results.get("results", [])[:max_snippets]]
            return snippets

        prompt_template = """
<OBJECTIVE_AND_PERSONA>
You are a patent linguist specializing in technical terminology analysis. Your task is to analyze provided technical snippets and extract high-precision synonyms and related terms for patent search optimization.
</OBJECTIVE_AND_PERSONA>

<INSTRUCTIONS>
To complete the task, you need to follow these steps:
1. Analyze the provided technical snippets for the given keyword
2. Extract core synonyms that appear in the snippets and retain the same technical function
3. Identify related terms that are broader, adjacent, or complementary to the keyword
4. Provide justifications for each term based on snippet evidence
5. Format the output as two distinct JSON lists
</INSTRUCTIONS>

<CONSTRAINTS>
Do:
- Include only terms that appear (exactly or inflected) in at least one snippet for core synonyms
- Ensure core synonyms retain the same technical function as the original keyword
- Limit core synonyms to 5-8 terms maximum
- Limit related terms to 5 terms maximum
- Provide clear justifications (≤10 words) for each core synonym
- Provide rationale for each related term
- Reference source snippet numbers for all terms

Don't:
- Don't include terms not found in the provided snippets for core synonyms
- Don't list full synonyms in the related terms section
- Don't exceed the specified term limits
- Don't provide justifications longer than 10 words for core synonyms
- Don't include terms without proper source attribution
</CONSTRAINTS>

<CONTEXT>
Keyword: {keyword}
Field Description: {context}

Technical Snippets:
{snippets}
</CONTEXT>

<OUTPUT_FORMAT>
The output format must be JSON format:
{{
    "core_synonyms": [
        {{
            "term": "example_term",
            "justification": "appears frequently with same technical meaning",
            "source": "src 1"
        }}
    ],
    "related_terms": [
        {{
            "term": "broader_term",
            "rationale": "encompasses the keyword within larger technical context",
            "source": "src 2"
        }}
    ]
}}
</OUTPUT_FORMAT>

<RECAP>
Extract 5-8 core synonyms and up to 5 related terms from the provided snippets. Core synonyms must appear in snippets and retain identical technical function. Related terms should be broader/adjacent concepts. All terms require source attribution and justification. IMPORTANT: Only generate the JSON output as defined - do not provide explanations, commentary, or any additional text beyond the required JSON format.
</RECAP>
        """

        prompt = PromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        sys_keys = {}

        def generate_synonyms(keyword: str, context: str):
            logger.info(f"🔍 Searching snippets for keyword: {keyword}")
            snippets = search_snippets(keyword)
            if not snippets:
                logger.warning(f"❌ No snippets found for keyword: {keyword}")
                return

            formatted_snippets = "\n".join([f"[{i+1}] {s}" for i, s in enumerate(snippets)])
            result = chain.run({
                "keyword": keyword,
                "snippets": formatted_snippets,
                "context": context,
            })
            
            try:
                # Try to parse JSON response
                result_clean = result.strip()
                if result_clean.startswith("```json"):
                    result_clean = result_clean.replace("```json", "").replace("```", "")
                
                parsed_result = json.loads(result_clean)
                
                # Extract terms from both core synonyms and related terms
                res = []
                
                # Add core synonyms
                if "core_synonyms" in parsed_result:
                    for item in parsed_result["core_synonyms"]:
                        if isinstance(item, dict) and "term" in item:
                            res.append(item["term"])
                        elif isinstance(item, str):
                            res.append(item)
                
                # Add related terms
                if "related_terms" in parsed_result:
                    for item in parsed_result["related_terms"]:
                        if isinstance(item, dict) and "term" in item:
                            res.append(item["term"])
                        elif isinstance(item, str):
                            res.append(item)
                
                sys_keys[keyword] = res
                logger.info(f"✅ Extracted {len(res)} terms for '{keyword}': {res}")
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"⚠️ JSON parsing failed for '{keyword}': {e}")
                logger.debug(f"Raw result: {result}")
                
                # Fallback to original parsing method
                syn_raw = result
                raws = syn_raw.split("\n")

                st = 0
                et = 0
                for i in range(len(raws)):
                    if "A. Core Synonyms" in raws[i]:
                        st = i+1
                        for j in range(st+1, len(raws)):
                            if "B. Related Terms" in raws[j]:
                                et = j
                                break
                        break
                raws = raws[st:et]
                res = []
                for item in raws:
                    try:
                        res.append(item.split("—")[0].split(".")[1].strip())
                    except:
                        pass

                sys_keys[keyword] = res

        concept_matrix = state["concept_matrix"].dict()
        seed_keywords = state["seed_keywords"].dict()

        for context in concept_matrix:
            for key in seed_keywords[context]:
                generate_synonyms(key, concept_matrix[context])

        return {"final_keywords": sys_keys}

    def summary_prompt_and_parser(self, state: ExtractionState) -> ExtractionState:
        """Generate summary using prompt and parser"""
        prompt, parser = self.prompts.get_summary_prompt_and_parser()
        
        # if state.get("problem") or state.get("technical"):
        #     input_text = f"Problem: {state.get('problem', '')}\nTechnical: {state.get('technical', '')}"
        #     response = self.llm.invoke(prompt.format(input_text=input_text))
        # else:
        #     response = self.llm.invoke(prompt.format(idea=state["input_text"]))
        response = self.llm.invoke(prompt.format(idea=state["input_text"]))

        concept_data = parser.parse(response)
        
        return {"summary_text": concept_data}

    def call_ipcs_api(self, state: ExtractionState) -> ExtractionState:
        """Call IPC classification API"""
        ipcs = get_ipc_predictions(state["summary_text"])
        logger.info(f"📋 IPC classification results: {ipcs}")
        return {"ipcs": ipcs}

    def genQuery(self, state: ExtractionState) -> ExtractionState:
        """Generate search queries"""
        keys = state["seed_keywords"]
        problem_purpose_keys = str([i for key in keys.problem_purpose for i in state["final_keywords"][key]])
        object_system_keys = str([i for key in keys.object_system for i in state["final_keywords"][key]])
        environment_field_keys = str([i for key in keys.environment_field for i in state["final_keywords"][key]])
        fipc = str([i["category"] for i in state["ipcs"]])
        problem = state.get("problem", "")

        prompt, parser = self.prompts.get_queries_prompt_and_parser()
        response = self.llm.invoke(prompt.format(
            problem=problem,
            problem_purpose_keys=problem_purpose_keys,
            object_system_keys=object_system_keys,
            environment_field_keys=environment_field_keys,
            CPC_CODES=fipc
        ))

        concept_data = parser.parse(response)
        logger.info(f"🔍 Generated {len(concept_data.queries)} search queries")
        return {"queries": concept_data}

    def genUrl(self, state: ExtractionState) -> ExtractionState:
        """Generate URLs from queries using Brave search"""
        final_url = list()

        queries = state["queries"].queries
        logger.info(f"🌐 Searching for URLs using {len(queries)} queries")
        
        for query in queries:
            url = "https://api.search.brave.com/res/v1/web/search"
            params = {
                "q": query + " site:patents.google.com/"
            }
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": "BSAQlxb-jIHFbW1mK0_S4zlTqfkuA3Z"
            }
            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()
                try:
                    for i in data["web"]["results"]:
                        url = i.get("url", None)
                        if url:
                            final_url.append(url)
                except:
                    logger.warning(f"❌ No results found for query: {query}")
            else:
                logger.error(f"❌ Search API request failed for query: {query} (status: {response.status_code})")
            time.sleep(1)  # Rate limit to avoid hitting API limits
        logger.info(f"🔗 Found {len(final_url)} URLs from search results")
        return {"final_url": final_url}

    def evalUrl(self, state: ExtractionState) -> ExtractionState:
        """Evaluate URLs for relevance"""
        final_url = list()
        logger.info(f"📊 Evaluating {len(state['final_url'])} URLs for relevance")
        
        for url in state["final_url"]:
            temp_score = dict()
            temp_score['url'] = url 
            temp_score['user_scenario'] = 0
            temp_score['user_problem'] = 0
            
            try:
                result = parse_idea_input(state["input_text"])
                temp = lay_thong_tin_patent(url)
                ex_text = prompt(temp['abstract'], temp['description'], temp['claims'])
                res = self.llm.invoke(ex_text)
                logger.debug(f"📄 LLM evaluation response for {url}: {res}")
                
                res = res.replace("```json", '')
                res = res.replace("```", '')
                data_res = json.loads(res)
                res_data = extract_user_info(data_res)
                
                score_scenario = eval_url(result["user_scenario"], res_data['user_scenario'])
                score_problem = eval_url(result["user_problem"], res_data['user_problem'])
                
                temp_score['user_scenario'] = score_scenario['llm_score']
                temp_score['user_problem'] = score_problem['llm_score']
                final_url.append(temp_score)
                
                logger.info(f"✅ Evaluated URL: {url} (scenario: {temp_score['user_scenario']}, problem: {temp_score['user_problem']})")
                
            except Exception as e:
                logger.error(f"❌ Error evaluating URL {url}: {str(e)}")
                # Add URL with zero scores if evaluation fails
                final_url.append(temp_score)
        
        logger.info(f"📊 Completed evaluation of {len(final_url)} URLs")
        return {"final_url": final_url}
