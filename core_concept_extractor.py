"""
Core Concept Seed Keyword Extraction System
A 3-phase patent seed keyword extraction system
"""

from typing import Dict, List, TypedDict, Annotated, Optional
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
import json
from prompts import ExtractionPrompts


# Data Models
class ConceptMatrix(BaseModel):
    """Concept matrix for Phase 1"""
    problem_purpose: str = Field(description="Problem / Purpose")
    object_system: str = Field(description="Object / System")
    action_method: str = Field(description="Action / Method")
    key_technical_feature: str = Field(description="Key Technical Feature")
    environment_field: str = Field(description="Environment / Application Field")
    advantage_result: str = Field(description="Advantage / Result")


class SeedKeywords(BaseModel):
    """Seed keywords for Phase 2"""
    problem_purpose: List[str] = Field(description="Keywords for problem/purpose")
    object_system: List[str] = Field(description="Keywords for object/system")
    action_method: List[str] = Field(description="Keywords for action/method")
    key_technical_feature: List[str] = Field(description="Keywords for key technical features")
    environment_field: List[str] = Field(description="Keywords for environment/field")
    advantage_result: List[str] = Field(description="Keywords for advantage/result")


class ValidationFeedback(BaseModel):
    """User validation feedback"""
    action: str  # "approve", "edit", "rerun"
    edited_keywords: Optional[SeedKeywords] = None
    feedback: Optional[str] = None


class ExtractionState(TypedDict):
    """State for LangGraph workflow"""
    input_text: str
    concept_matrix: Optional[ConceptMatrix]
    seed_keywords: Optional[SeedKeywords]
    validation_feedback: Optional[ValidationFeedback]
    final_keywords: Optional[SeedKeywords]
    current_phase: str
    messages: List[str]


class CoreConceptExtractor:
    """Patent seed keyword extraction system"""
    
    def __init__(self, model_name: str = "llama3"):
        self.llm = Ollama(model=model_name, temperature=0.3)
        self.prompts = ExtractionPrompts()
        self.messages = ExtractionPrompts.get_phase_completion_messages()
        self.validation_messages = ExtractionPrompts.get_validation_messages()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow"""
        workflow = StateGraph(ExtractionState)
        
        # Add nodes
        workflow.add_node("phase1_concept_extraction", self.phase1_concept_extraction)
        workflow.add_node("phase2_keyword_extraction", self.phase2_keyword_extraction)
        workflow.add_node("phase3_auto_refinement", self.phase3_auto_refinement)
        workflow.add_node("final_human_evaluation", self.final_human_evaluation)
        workflow.add_node("manual_editing", self.manual_editing)
        workflow.add_node("finalize", self.finalize_results)
        
        # Define flow
        workflow.set_entry_point("phase1_concept_extraction")
        workflow.add_edge("phase1_concept_extraction", "phase2_keyword_extraction")
        workflow.add_edge("phase2_keyword_extraction", "phase3_auto_refinement")
        workflow.add_edge("phase3_auto_refinement", "final_human_evaluation")
        workflow.add_conditional_edges(
            "final_human_evaluation",
            self._should_edit_or_rerun,
            {
                "edit": "manual_editing",
                "rerun": "phase1_concept_extraction",
                "approve": "finalize"
            }
        )
        workflow.add_edge("manual_editing", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile(checkpointer=MemorySaver())
    
    def phase1_concept_extraction(self, state: ExtractionState) -> ExtractionState:
        """Phase 1: Abstraction & Concept Definition"""
        prompt = self.prompts.get_phase1_prompt()
        
        response = self.llm.invoke(prompt.format(input_text=state["input_text"]))
        
        try:
            # Parse JSON response
            concept_data = json.loads(response.strip())
            concept_matrix = ConceptMatrix(**concept_data)
        except:
            # Fallback parsing if JSON is invalid
            concept_matrix = self._parse_concept_response(response)
        
        state["concept_matrix"] = concept_matrix
        state["current_phase"] = "phase1_completed"
        state["messages"].append(self.messages["phase1_completed"])
        
        return state
    
    def phase2_keyword_extraction(self, state: ExtractionState) -> ExtractionState:
        """Phase 2: Seed Keyword Extraction"""
        concept_matrix = state["concept_matrix"]
        
        prompt = self.prompts.get_phase2_prompt()
        
        response = self.llm.invoke(prompt.format(**concept_matrix.dict()))
        
        try:
            keyword_data = json.loads(response.strip())
            seed_keywords = SeedKeywords(**keyword_data)
        except:
            seed_keywords = self._parse_keyword_response(response)
        
        state["seed_keywords"] = seed_keywords
        state["current_phase"] = "phase2_completed"
        state["messages"].append(self.messages["phase2_completed"])
        
        return state
    
    def phase3_auto_refinement(self, state: ExtractionState) -> ExtractionState:
        """Phase 3: Automatic Refinement & Quality Enhancement"""
        concept_matrix = state["concept_matrix"]
        current_keywords = state["seed_keywords"]
        
        prompt = self.prompts.get_phase3_auto_prompt()
        
        response = self.llm.invoke(prompt.format(
            concept_matrix=concept_matrix.dict(),
            current_keywords=current_keywords.dict()
        ))
        
        try:
            refined_data = json.loads(response.strip())
            refined_keywords = SeedKeywords(**refined_data)
        except:
            refined_keywords = self._parse_keyword_response(response)
        
        state["seed_keywords"] = refined_keywords
        state["current_phase"] = "phase3_completed"
        state["messages"].append(self.messages["phase3_completed"])
        
        return state
    
    def final_human_evaluation(self, state: ExtractionState) -> ExtractionState:
        """Final human evaluation of complete results"""
        msgs = self.validation_messages
        
        print("\n" + msgs["separator"])
        print(msgs["final_evaluation_title"])
        print(msgs["separator"])
        
        # Display final results
        concept_matrix = state["concept_matrix"]
        seed_keywords = state["seed_keywords"]
        
        print(msgs["concept_matrix_header"])
        for field, value in concept_matrix.dict().items():
            print(f"  • {field.replace('_', ' ').title()}: {value}")
        
        print(msgs["seed_keywords_header"])
        for field, keywords in seed_keywords.dict().items():
            print(f"  • {field.replace('_', ' ').title()}: {keywords}")
        
        print(msgs["divider"])
        print(msgs["action_options"])
        
        # Get user action
        while True:
            action = input(msgs["action_prompt"]).lower().strip()
            if action in ['1', 'approve', 'a']:
                feedback = ValidationFeedback(action="approve")
                break
            elif action in ['2', 'edit', 'e']:
                feedback = self._get_manual_edits(seed_keywords)
                break
            elif action in ['3', 'rerun', 'r']:
                feedback_text = input(msgs["rerun_feedback_prompt"])
                feedback = ValidationFeedback(action="rerun", feedback=feedback_text)
                break
            else:
                print(msgs["invalid_action"])
        
        state["validation_feedback"] = feedback
        state["messages"].append(f"User action: {feedback.action}")
        
        return state
    
    def manual_editing(self, state: ExtractionState) -> ExtractionState:
        """Allow user to manually edit keywords"""
        feedback = state["validation_feedback"]
        
        if feedback.edited_keywords:
            state["seed_keywords"] = feedback.edited_keywords
            state["messages"].append("Keywords manually edited by user")
        
        return state
    
    def _get_manual_edits(self, current_keywords: SeedKeywords) -> ValidationFeedback:
        """Get manual edits from user"""
        print("\n📝 Manual Editing Mode")
        print("Current keywords will be displayed. Press Enter to keep current value, or type new keywords separated by commas.")
        
        edited_data = {}
        
        for field, keywords in current_keywords.dict().items():
            field_name = field.replace('_', ' ').title()
            current_str = ", ".join(keywords)
            print(f"\n{field_name}: [{current_str}]")
            
            new_input = input(f"New {field_name} (or Enter to keep): ").strip()
            if new_input:
                edited_data[field] = [kw.strip() for kw in new_input.split(',') if kw.strip()]
            else:
                edited_data[field] = keywords
        
        edited_keywords = SeedKeywords(**edited_data)
        return ValidationFeedback(action="edit", edited_keywords=edited_keywords)
    
    def finalize_results(self, state: ExtractionState) -> ExtractionState:
        """Finalize the results"""
        state["final_keywords"] = state["seed_keywords"]
        state["current_phase"] = "completed"
        state["messages"].append(self.messages["extraction_completed"])
        
        return state
    
    def _should_edit_or_rerun(self, state: ExtractionState) -> str:
        """Condition to decide user action"""
        feedback = state["validation_feedback"]
        return feedback.action
    
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
                elif 'action' in key or 'method' in key:
                    data['action_method'] = value.strip()
                elif 'technical' in key or 'feature' in key:
                    data['key_technical_feature'] = value.strip()
                elif 'environment' in key or 'field' in key:
                    data['environment_field'] = value.strip()
                elif 'advantage' in key or 'result' in key:
                    data['advantage_result'] = value.strip()
        
        return ConceptMatrix(**data)
    
    def _parse_keyword_response(self, response: str) -> SeedKeywords:
        """Parse keyword response when JSON parsing fails"""
        # Fallback parsing logic
        return SeedKeywords(
            problem_purpose=["extracted_keyword"],
            object_system=["extracted_keyword"],
            action_method=["extracted_keyword"],
            key_technical_feature=["extracted_keyword"],
            environment_field=["extracted_keyword"],
            advantage_result=["extracted_keyword"]
        )
    
    def extract_keywords(self, input_text: str) -> Dict:
        """Run the complete keyword extraction workflow"""
        initial_state = ExtractionState(
            input_text=input_text,
            concept_matrix=None,
            seed_keywords=None,
            validation_feedback=None,
            final_keywords=None,
            current_phase="initialized",
            messages=[]
        )
        
        # Run workflow
        result = self.graph.invoke(initial_state)
        
        return {
            "final_keywords": result["final_keywords"].dict() if result["final_keywords"] else None,
            "concept_matrix": result["concept_matrix"].dict() if result["concept_matrix"] else None,
            "messages": result["messages"]
        }
    
    def extract_keywords_with_feedback(self, input_text: str, feedback: str = None) -> Dict:
        """Run the complete keyword extraction workflow with optional feedback"""
        initial_state = ExtractionState(
            input_text=input_text,
            concept_matrix=None,
            seed_keywords=None,
            validation_feedback=None,
            final_keywords=None,
            current_phase="initialized",
            messages=[]
        )
        
        # Add feedback to initial state if provided (for re-runs)
        if feedback:
            initial_state["messages"].append(f"Re-run with feedback: {feedback}")
        
        # Run workflow
        result = self.graph.invoke(initial_state)
        
        return {
            "final_keywords": result["final_keywords"].dict() if result["final_keywords"] else None,
            "concept_matrix": result["concept_matrix"].dict() if result["concept_matrix"] else None,
            "messages": result["messages"],
            "user_action": result.get("validation_feedback", {}).action if result.get("validation_feedback") else None
        }


if __name__ == "__main__":
    # Example usage
    extractor = CoreConceptExtractor(model_name="llama3")
    
    sample_text = """
    A smart irrigation system that uses soil moisture sensors and weather data 
    to automatically control irrigation schedules. The system helps save water and 
    optimize plant care in agriculture and gardening.
    """
    
    print("🚀 Starting patent seed keyword extraction...")
    results = extractor.extract_keywords(sample_text)
    
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2, ensure_ascii=False))
