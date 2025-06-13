"""
Core Concept Seed Keyword Extraction System
Hệ thống trích xuất từ khóa gốc sáng chế với 3 pha
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


# Data Models
class ConceptMatrix(BaseModel):
    """Ma trận khái niệm cho Pha 1"""
    problem_purpose: str = Field(description="Vấn đề / mục tiêu")
    object_system: str = Field(description="Đối tượng / hệ thống")
    action_method: str = Field(description="Hành động / phương pháp")
    key_technical_feature: str = Field(description="Đặc điểm kỹ thuật cốt lõi")
    environment_field: str = Field(description="Môi trường / lĩnh vực ứng dụng")
    advantage_result: str = Field(description="Lợi ích / kết quả đạt được")


class SeedKeywords(BaseModel):
    """Từ khóa gốc cho Pha 2"""
    problem_purpose: List[str] = Field(description="Từ khóa cho vấn đề/mục tiêu")
    object_system: List[str] = Field(description="Từ khóa cho đối tượng/hệ thống")
    action_method: List[str] = Field(description="Từ khóa cho hành động/phương pháp")
    key_technical_feature: List[str] = Field(description="Từ khóa cho đặc điểm kỹ thuật")
    environment_field: List[str] = Field(description="Từ khóa cho môi trường/lĩnh vực")
    advantage_result: List[str] = Field(description="Từ khóa cho lợi ích/kết quả")


class ValidationFeedback(BaseModel):
    """Phản hồi đánh giá từ người dùng"""
    is_approved: bool
    feedback: Optional[str] = None
    suggestions: Optional[List[str]] = None


class ExtractionState(TypedDict):
    """State cho LangGraph workflow"""
    input_text: str
    concept_matrix: Optional[ConceptMatrix]
    seed_keywords: Optional[SeedKeywords]
    validation_feedback: Optional[ValidationFeedback]
    final_keywords: Optional[SeedKeywords]
    current_phase: str
    messages: List[str]


class CoreConceptExtractor:
    """Hệ thống trích xuất từ khóa gốc sáng chế"""
    
    def __init__(self, model_name: str = "llama3"):
        self.llm = Ollama(model=model_name, temperature=0.3)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Xây dựng LangGraph workflow"""
        workflow = StateGraph(ExtractionState)
        
        # Thêm các nodes
        workflow.add_node("phase1_concept_extraction", self.phase1_concept_extraction)
        workflow.add_node("phase2_keyword_extraction", self.phase2_keyword_extraction)
        workflow.add_node("human_validation", self.human_validation)
        workflow.add_node("phase3_refinement", self.phase3_refinement)
        workflow.add_node("finalize", self.finalize_results)
        
        # Định nghĩa luồng
        workflow.set_entry_point("phase1_concept_extraction")
        workflow.add_edge("phase1_concept_extraction", "phase2_keyword_extraction")
        workflow.add_edge("phase2_keyword_extraction", "human_validation")
        workflow.add_conditional_edges(
            "human_validation",
            self._should_refine,
            {
                "refine": "phase3_refinement",
                "approve": "finalize"
            }
        )
        workflow.add_edge("phase3_refinement", "human_validation")
        workflow.add_edge("finalize", END)
        
        return workflow.compile(checkpointer=MemorySaver())
    
    def phase1_concept_extraction(self, state: ExtractionState) -> ExtractionState:
        """Pha 1: Trừu tượng hóa & Định nghĩa Khái niệm"""
        prompt = PromptTemplate.from_template("""
        Phân tích tài liệu kỹ thuật sau và trích xuất thông tin cho Ma trận Khái niệm:

        Tài liệu: {input_text}

        Hãy điền thông tin súc tích cho từng thành phần (1-2 câu ngắn):

        1. Problem/Purpose (Vấn đề/mục tiêu):
        2. Object/System (Đối tượng/hệ thống):
        3. Action/Method (Hành động/phương pháp):
        4. Key Technical Feature/Structure (Đặc điểm kỹ thuật cốt lõi):
        5. Environment/Field (Môi trường/lĩnh vực ứng dụng):
        6. Advantage/Result (Lợi ích/kết quả đạt được):

        Trả về định dạng JSON với các key: problem_purpose, object_system, action_method, key_technical_feature, environment_field, advantage_result
        """)
        
        response = self.llm.invoke(prompt.format(input_text=state["input_text"]))
        
        try:
            # Parse JSON response
            concept_data = json.loads(response.strip())
            concept_matrix = ConceptMatrix(**concept_data)
        except:
            # Fallback parsing nếu JSON không hợp lệ
            concept_matrix = self._parse_concept_response(response)
        
        state["concept_matrix"] = concept_matrix
        state["current_phase"] = "phase1_completed"
        state["messages"].append(f"Pha 1 hoàn thành: Đã trích xuất Ma trận Khái niệm")
        
        return state
    
    def phase2_keyword_extraction(self, state: ExtractionState) -> ExtractionState:
        """Pha 2: Trích xuất Từ khóa Gốc"""
        concept_matrix = state["concept_matrix"]
        
        prompt = PromptTemplate.from_template("""
        Từ Ma trận Khái niệm sau, trích xuất 1-3 từ khóa/cụm từ kỹ thuật đặc trưng cho mỗi thành phần.
        Ưu tiên danh từ kỹ thuật và động từ chính, tránh từ quá chung.

        Ma trận Khái niệm:
        - Problem/Purpose: {problem_purpose}
        - Object/System: {object_system}
        - Action/Method: {action_method}
        - Key Technical Feature: {key_technical_feature}
        - Environment/Field: {environment_field}
        - Advantage/Result: {advantage_result}

        Trả về định dạng JSON với mỗi thành phần là một mảng từ khóa:
        {{
            "problem_purpose": ["keyword1", "keyword2"],
            "object_system": ["keyword1"],
            ...
        }}
        """)
        
        response = self.llm.invoke(prompt.format(**concept_matrix.dict()))
        
        try:
            keyword_data = json.loads(response.strip())
            seed_keywords = SeedKeywords(**keyword_data)
        except:
            seed_keywords = self._parse_keyword_response(response)
        
        state["seed_keywords"] = seed_keywords
        state["current_phase"] = "phase2_completed"
        state["messages"].append(f"Pha 2 hoàn thành: Đã trích xuất từ khóa gốc")
        
        return state
    
    def human_validation(self, state: ExtractionState) -> ExtractionState:
        """Human-in-the-loop validation"""
        print("\n" + "="*60)
        print("🔍 ĐÁNH GIÁ KẾT QUẢ TRÍCH XUẤT TỪ KHÓA")
        print("="*60)
        
        # Hiển thị kết quả
        concept_matrix = state["concept_matrix"]
        seed_keywords = state["seed_keywords"]
        
        print("\n📋 Ma trận Khái niệm:")
        for field, value in concept_matrix.dict().items():
            print(f"  • {field.replace('_', ' ').title()}: {value}")
        
        print("\n🔑 Từ khóa gốc:")
        for field, keywords in seed_keywords.dict().items():
            print(f"  • {field.replace('_', ' ').title()}: {keywords}")
        
        print("\n" + "-"*60)
        
        # Lấy phản hồi từ người dùng
        while True:
            approval = input("Bạn có hài lòng với kết quả? (y/n): ").lower().strip()
            if approval in ['y', 'yes', 'có']:
                feedback = ValidationFeedback(is_approved=True)
                break
            elif approval in ['n', 'no', 'không']:
                feedback_text = input("Nhận xét của bạn: ")
                suggestions = input("Đề xuất cải thiện (cách nhau bởi dấu ;): ")
                
                feedback = ValidationFeedback(
                    is_approved=False,
                    feedback=feedback_text,
                    suggestions=suggestions.split(';') if suggestions else None
                )
                break
            else:
                print("Vui lòng nhập 'y' hoặc 'n'")
        
        state["validation_feedback"] = feedback
        state["messages"].append(f"Đánh giá người dùng: {'Chấp thuận' if feedback.is_approved else 'Yêu cầu cải thiện'}")
        
        return state
    
    def phase3_refinement(self, state: ExtractionState) -> ExtractionState:
        """Pha 3: Kiểm tra & Tinh chỉnh"""
        feedback = state["validation_feedback"]
        current_keywords = state["seed_keywords"]
        
        prompt = PromptTemplate.from_template("""
        Cải thiện từ khóa gốc dựa trên phản hồi của người dùng:

        Từ khóa hiện tại:
        {current_keywords}

        Phản hồi người dùng: {feedback}
        Đề xuất: {suggestions}

        Hãy tinh chỉnh từ khóa để:
        1. Đảm bảo đủ đặc trưng, tránh quá chung
        2. Bổ sung khái niệm kỹ thuật quan trọng bị thiếu
        3. Tối ưu hóa cho tìm kiếm sáng chế

        Trả về định dạng JSON tương tự như trước.
        """)
        
        response = self.llm.invoke(prompt.format(
            current_keywords=current_keywords.dict(),
            feedback=feedback.feedback or "",
            suggestions="; ".join(feedback.suggestions) if feedback.suggestions else ""
        ))
        
        try:
            refined_data = json.loads(response.strip())
            refined_keywords = SeedKeywords(**refined_data)
        except:
            refined_keywords = self._parse_keyword_response(response)
        
        state["seed_keywords"] = refined_keywords
        state["current_phase"] = "phase3_completed"
        state["messages"].append(f"Pha 3 hoàn thành: Đã tinh chỉnh từ khóa")
        
        return state
    
    def finalize_results(self, state: ExtractionState) -> ExtractionState:
        """Hoàn thiện kết quả cuối cùng"""
        state["final_keywords"] = state["seed_keywords"]
        state["current_phase"] = "completed"
        state["messages"].append("✅ Hoàn thành trích xuất từ khóa gốc sáng chế")
        
        return state
    
    def _should_refine(self, state: ExtractionState) -> str:
        """Điều kiện để quyết định có cần tinh chỉnh không"""
        feedback = state["validation_feedback"]
        return "approve" if feedback.is_approved else "refine"
    
    def _parse_concept_response(self, response: str) -> ConceptMatrix:
        """Parse response khi JSON parsing thất bại"""
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
        """Parse keyword response khi JSON parsing thất bại"""
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
        """Chạy quy trình trích xuất từ khóa hoàn chỉnh"""
        initial_state = ExtractionState(
            input_text=input_text,
            concept_matrix=None,
            seed_keywords=None,
            validation_feedback=None,
            final_keywords=None,
            current_phase="initialized",
            messages=[]
        )
        
        # Chạy workflow
        result = self.graph.invoke(initial_state)
        
        return {
            "final_keywords": result["final_keywords"].dict() if result["final_keywords"] else None,
            "concept_matrix": result["concept_matrix"].dict() if result["concept_matrix"] else None,
            "messages": result["messages"]
        }


if __name__ == "__main__":
    # Example usage
    extractor = CoreConceptExtractor(model_name="llama3")
    
    sample_text = """
    Hệ thống tưới tiêu thông minh sử dụng cảm biến độ ẩm đất và dữ liệu thời tiết 
    để tự động điều khiển lịch tưới nước. Hệ thống giúp tiết kiệm nước và tối ưu 
    hóa việc chăm sóc cây trồng trong nông nghiệp và làm vườn.
    """
    
    print("🚀 Bắt đầu trích xuất từ khóa gốc sáng chế...")
    results = extractor.extract_keywords(sample_text)
    
    print("\n" + "="*60)
    print("📊 KẾT QUẢ CUỐI CÙNG")
    print("="*60)
    print(json.dumps(results, indent=2, ensure_ascii=False))
