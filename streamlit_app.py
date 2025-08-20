"""
Streamlit Web Interface for Patent AI Agent
Provides an intuitive web interface for patent keyword extraction
"""

import streamlit as st
import json
import datetime
from typing import Dict, Any
import traceback

from src.core.extractor import CoreConceptExtractor, ValidationFeedback, SeedKeywords

# Configure Streamlit page
st.set_page_config(
    page_title="Patent AI Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin: 1rem 0;
        padding: 0.5rem;
        background-color: #f0f8f0;
        border-radius: 5px;
        border-left: 4px solid #2e8b57;
    }
    .result-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    .info-box {
        background-color: #d1ecf1;
        border-color: #bee5eb;
        color: #0c5460;
    }
    .warning-box {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'extractor' not in st.session_state:
        st.session_state.extractor = CoreConceptExtractor()
    
    if 'extraction_results' not in st.session_state:
        st.session_state.extraction_results = None
    
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    
    if 'step_results' not in st.session_state:
        st.session_state.step_results = {}

def display_step_header(step_num: int, title: str, description: str):
    """Display a step header with styling"""
    st.markdown(f'<div class="step-header">Step {step_num}: {title}</div>', unsafe_allow_html=True)
    st.markdown(f"*{description}*")

def display_result_box(content: str, box_type: str = "info"):
    """Display content in a styled box"""
    css_class = f"result-box {box_type}-box"
    st.markdown(f'<div class="{css_class}">{content}</div>', unsafe_allow_html=True)

def format_concept_matrix(concept_matrix):
    """Format concept matrix for display"""
    if not concept_matrix:
        return "No concept matrix generated."
    
    content = ""
    for field, value in concept_matrix.items():
        field_name = field.replace('_', ' ').title()
        content += f"**{field_name}:** {value}<br><br>"
    return content

def format_seed_keywords(seed_keywords):
    """Format seed keywords for display"""
    if not seed_keywords:
        return "No keywords generated."
    
    content = ""
    for field, keywords in seed_keywords.items():
        field_name = field.replace('_', ' ').title()
        keyword_list = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
        content += f"**{field_name}:** {keyword_list}<br><br>"
    return content

def format_final_keywords(final_keywords):
    """Format final keywords with synonyms for display"""
    if not final_keywords:
        return "No final keywords generated."
    
    content = ""
    for keyword, synonyms in final_keywords.items():
        synonym_list = ", ".join(synonyms) if isinstance(synonyms, list) else str(synonyms)
        content += f"**{keyword}:** {synonym_list}<br><br>"
    return content

def format_queries(queries):
    """Format search queries for display"""
    if not queries:
        return "No queries generated."
    
    content = ""
    for i, query in enumerate(queries, 1):
        content += f"**Query {i}:** {query}<br><br>"
    return content

def format_urls_with_scores(final_urls):
    """Format URLs with evaluation scores for display"""
    if not final_urls:
        return "No URLs generated."
    
    content = ""
    for i, url_data in enumerate(final_urls, 1):
        if isinstance(url_data, dict):
            url = url_data.get('url', 'N/A')
            scenario_score = url_data.get('user_scenario', 0)
            problem_score = url_data.get('user_problem', 0)
            content += f"**Result {i}:**<br>"
            content += f"URL: <a href='{url}' target='_blank'>{url}</a><br>"
            content += f"Scenario Score: {scenario_score}<br>"
            content += f"Problem Score: {problem_score}<br><br>"
        else:
            content += f"**Result {i}:** {url_data}<br><br>"
    return content

def run_extraction_step(state, step_name, step_func):
    """Run a single extraction step and update session state"""
    try:
        with st.spinner(f"Processing {step_name}..."):
            result = step_func(state)
            state.update(result)
            return True, None
    except Exception as e:
        error_msg = f"Error in {step_name}: {str(e)}"
        st.error(error_msg)
        return False, error_msg

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Patent AI Agent - Keyword Extraction</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        model_name = st.selectbox(
            "Select LLM Model",
            ["qwen3:4b", "llama2", "mistral", "codellama"],
            index=0
        )
        
        if st.button("🔄 Reset Session"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
    
    # Main input section
    st.header("📝 Input Patent Idea")
    
    # Two input boxes as requested
    col1, col2 = st.columns(2)
    
    with col1:
        problem_text = st.text_area(
            "Problem Description",
            placeholder="Describe the technical problem or challenge that needs to be solved...",
            height=200,
            help="Enter the specific technical problem the invention aims to solve"
        )
    
    with col2:
        technical_text = st.text_area(
            "Technical Solution",
            placeholder="Describe the technical solution, method, or approach...",
            height=200,
            help="Enter the core technical solution, method, or approach proposed"
        )
    
    # Combine inputs for processing
    if problem_text and technical_text:
        combined_input = f"**Problem:** {problem_text}\n\n**Technical Solution:** {technical_text}"
        
        st.markdown("### 📋 Combined Input Preview")
        st.markdown(f'<div class="result-box info-box">{combined_input}</div>', unsafe_allow_html=True)
        
        # Process button
        if st.button("🚀 Start Extraction Process", type="primary", use_container_width=True):
            # Initialize extraction state
            extraction_state = {
                "input_text": combined_input,
                "summary_text": None,
                "ipcs": None,
                "concept_matrix": None,
                "seed_keywords": None,
                "validation_feedback": None,
                "final_keywords": None,
                "queries": None,
                "final_url": None
            }
            
            # Update extractor model if needed
            if st.session_state.extractor.llm.model != model_name:
                st.session_state.extractor = CoreConceptExtractor(model_name=model_name)
            
            # Progress tracking
            progress_bar = st.progress(0)
            steps_completed = 0
            total_steps = 8
            
            # Step 0: Initial step
            display_step_header(0, "Initialization", "Setting up the extraction process")
            success, error = run_extraction_step(extraction_state, "Step 0", st.session_state.extractor.step0)
            if success:
                steps_completed += 1
                progress_bar.progress(steps_completed / total_steps)
                display_result_box("✅ Initialization completed successfully", "success")
            
            # Step 1: Concept extraction
            if success:
                display_step_header(1, "Concept Matrix Extraction", "Extracting core concepts from the input")
                success, error = run_extraction_step(extraction_state, "Step 1", st.session_state.extractor.step1_concept_extraction)
                if success:
                    steps_completed += 1
                    progress_bar.progress(steps_completed / total_steps)
                    concept_content = format_concept_matrix(extraction_state["concept_matrix"].dict() if extraction_state["concept_matrix"] else None)
                    display_result_box(concept_content, "info")
            
            # Step 2: Keyword generation
            if success:
                display_step_header(2, "Seed Keyword Generation", "Generating initial keywords from concepts")
                success, error = run_extraction_step(extraction_state, "Step 2", st.session_state.extractor.step2_keyword_generation)
                if success:
                    steps_completed += 1
                    progress_bar.progress(steps_completed / total_steps)
                    keyword_content = format_seed_keywords(extraction_state["seed_keywords"].dict() if extraction_state["seed_keywords"] else None)
                    display_result_box(keyword_content, "info")
            
            # Step 3: Human evaluation (replaced with Streamlit interface)
            if success:
                display_step_header(3, "Keyword Validation", "Review and validate the extracted keywords")
                
                concept_matrix = extraction_state["concept_matrix"]
                seed_keywords = extraction_state["seed_keywords"]
                
                if concept_matrix and seed_keywords:
                    st.subheader("📋 Review Results")
                    
                    # Display results for review
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Concept Matrix:**")
                        for field, value in concept_matrix.dict().items():
                            st.write(f"• **{field.replace('_', ' ').title()}:** {value}")
                    
                    with col2:
                        st.markdown("**Generated Keywords:**")
                        for field, keywords in seed_keywords.dict().items():
                            st.write(f"• **{field.replace('_', ' ').title()}:** {', '.join(keywords)}")
                    
                    # User action selection
                    st.subheader("🎯 Choose Action")
                    action = st.radio(
                        "What would you like to do?",
                        ["✅ Approve - Continue with these keywords", 
                         "✏️ Edit - Modify the keywords", 
                         "❌ Reject - Regenerate keywords"],
                        index=0
                    )
                    
                    # Handle different actions
                    if action.startswith("✅"):
                        validation_feedback = ValidationFeedback(action="approve")
                        extraction_state["validation_feedback"] = validation_feedback
                        steps_completed += 1
                        progress_bar.progress(steps_completed / total_steps)
                        display_result_box("✅ Keywords approved - proceeding to next steps", "success")
                        
                    elif action.startswith("✏️"):
                        st.subheader("✏️ Edit Keywords")
                        edited_data = {}
                        
                        for field, keywords in seed_keywords.dict().items():
                            field_name = field.replace('_', ' ').title()
                            current_str = ", ".join(keywords)
                            
                            new_keywords = st.text_input(
                                f"{field_name}",
                                value=current_str,
                                help="Enter keywords separated by commas"
                            )
                            
                            if new_keywords:
                                edited_data[field] = [kw.strip() for kw in new_keywords.split(',') if kw.strip()]
                            else:
                                edited_data[field] = keywords
                        
                        if st.button("Save Edits"):
                            edited_keywords = SeedKeywords(**edited_data)
                            validation_feedback = ValidationFeedback(action="edit", edited_keywords=edited_keywords)
                            extraction_state["validation_feedback"] = validation_feedback
                            extraction_state["seed_keywords"] = edited_keywords
                            steps_completed += 1
                            progress_bar.progress(steps_completed / total_steps)
                            display_result_box("✏️ Keywords edited successfully", "success")
                    
                    elif action.startswith("❌"):
                        feedback_text = st.text_area("Optional feedback for improvement:")
                        if st.button("Regenerate Keywords"):
                            validation_feedback = ValidationFeedback(action="reject", feedback=feedback_text)
                            extraction_state["validation_feedback"] = validation_feedback
                            # This would trigger regeneration in a real implementation
                            st.warning("Keywords rejected - would regenerate in full implementation")
                    
                    # Continue only if approved or edited
                    if extraction_state.get("validation_feedback") and extraction_state["validation_feedback"].action in ["approve", "edit"]:
                        
                        # Generate synonyms and final keywords
                        if success:
                            display_step_header(4, "Synonym Generation", "Generating synonyms and related terms")
                            success, error = run_extraction_step(extraction_state, "Gen Key", st.session_state.extractor.gen_key)
                            if success:
                                steps_completed += 1
                                progress_bar.progress(steps_completed / total_steps)
                                final_keyword_content = format_final_keywords(extraction_state["final_keywords"])
                                display_result_box(final_keyword_content, "info")
                        
                        # Generate summary and IPCs
                        if success:
                            display_step_header(5, "Summary Generation", "Creating technical summary")
                            success, error = run_extraction_step(extraction_state, "Summary", st.session_state.extractor.summary_prompt_and_parser)
                            if success:
                                success, error = run_extraction_step(extraction_state, "IPC Classification", st.session_state.extractor.call_ipcs_api)
                                if success:
                                    steps_completed += 1
                                    progress_bar.progress(steps_completed / total_steps)
                                    summary_content = f"**Summary:** {extraction_state['summary_text']}<br><br>"
                                    if extraction_state.get('ipcs'):
                                        ipc_list = [f"{ipc.get('category', 'N/A')} (Score: {ipc.get('score', 'N/A')})" for ipc in extraction_state['ipcs']]
                                        summary_content += f"**IPC Classifications:** {', '.join(ipc_list)}"
                                    display_result_box(summary_content, "info")
                        
                        # Generate search queries
                        if success:
                            display_step_header(6, "Query Generation", "Creating search queries")
                            success, error = run_extraction_step(extraction_state, "Query Generation", st.session_state.extractor.genQuery)
                            if success:
                                steps_completed += 1
                                progress_bar.progress(steps_completed / total_steps)
                                query_content = format_queries(extraction_state["queries"].queries if extraction_state["queries"] else [])
                                display_result_box(query_content, "info")
                        
                        # Generate URLs and evaluate
                        if success:
                            display_step_header(7, "Patent Search & Evaluation", "Searching patents and evaluating relevance")
                            success, error = run_extraction_step(extraction_state, "URL Generation", st.session_state.extractor.genUrl)
                            if success:
                                success, error = run_extraction_step(extraction_state, "URL Evaluation", st.session_state.extractor.evalUrl)
                                if success:
                                    steps_completed += 1
                                    progress_bar.progress(steps_completed / total_steps)
                                    url_content = format_urls_with_scores(extraction_state["final_url"])
                                    display_result_box(url_content, "info")
                        
                        # Final results
                        if success:
                            st.session_state.extraction_results = extraction_state
                            progress_bar.progress(1.0)
                            st.success("🎉 Extraction process completed successfully!")
                            
                            # Export results
                            st.subheader("💾 Export Results")
                            
                            results = {
                                "timestamp": datetime.datetime.now().isoformat(),
                                "concept_matrix": extraction_state["concept_matrix"].dict() if extraction_state["concept_matrix"] else None,
                                "seed_keywords": extraction_state["seed_keywords"].dict() if extraction_state["seed_keywords"] else None,
                                "final_keywords": extraction_state["final_keywords"],
                                "summary": extraction_state["summary_text"],
                                "ipcs": extraction_state["ipcs"],
                                "queries": extraction_state["queries"].dict() if extraction_state["queries"] else None,
                                "final_urls": extraction_state["final_url"]
                            }
                            
                            json_str = json.dumps(results, indent=2, ensure_ascii=False)
                            
                            st.download_button(
                                label="📁 Download Results as JSON",
                                data=json_str,
                                file_name=f"patent_extraction_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                type="primary"
                            )
    else:
        st.info("👆 Please enter both problem description and technical solution to start the extraction process.")

if __name__ == "__main__":
    main()
