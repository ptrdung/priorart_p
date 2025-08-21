import streamlit as st
from src.core.extractor import CoreConceptExtractor, ValidationFeedback, SeedKeywords
import asyncio
from typing import Dict
import json

# Initialize session state
if 'extractor' not in st.session_state:
    st.session_state.extractor = CoreConceptExtractor()
if 'current_state' not in st.session_state:
    st.session_state.current_state = None
if 'keywords_displayed' not in st.session_state:
    st.session_state.keywords_displayed = False
if 'processing' not in st.session_state:
    st.session_state.processing = False

def display_concept_matrix(concept_matrix):
    st.subheader("Concept Matrix")
    if concept_matrix:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Problem/Purpose:")
            st.write(concept_matrix.problem_purpose)
        with col2:
            st.write("Object/System:")
            st.write(concept_matrix.object_system)
        with col3:
            st.write("Environment/Field:")
            st.write(concept_matrix.environment_field)

def display_keywords(seed_keywords):
    st.subheader("Generated Keywords")
    if seed_keywords:
        for field, keywords in seed_keywords.dict().items():
            st.markdown(f"**{field.replace('_', ' ').title()}:**")
            keywords_str = ", ".join(keywords)
            st.text_area(
                label=field,
                value=keywords_str,
                key=f"{field}_edit",
                help=f"Edit {field.replace('_', ' ')} keywords"
            )

def get_edited_keywords():
    from src.core.extractor import SeedKeywords
    return SeedKeywords(
        problem_purpose=[k.strip() for k in st.session_state.problem_purpose_edit.split(',')],
        object_system=[k.strip() for k in st.session_state.object_system_edit.split(',')],
        environment_field=[k.strip() for k in st.session_state.environment_field_edit.split(',')]
    )

def process_step(state):
    if not state:
        return None
    
    # Display current state information
    if state.get('concept_matrix'):
        display_concept_matrix(state['concept_matrix'])
    
    if state.get('seed_keywords'):
        display_keywords(state['seed_keywords'])
        st.session_state.keywords_displayed = True
        
        # Show action buttons with feedback
        st.markdown("### Review and Action")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✅ Approve"):
                st.session_state.processing = True
                feedback = ValidationFeedback(action="approve")
                state['validation_feedback'] = feedback
                return state
                
        with col2:
            if st.button("❌ Reject"):
                feedback_text = st.text_input("Provide feedback for rejection:")
                if st.button("Submit Rejection"):
                    st.session_state.processing = True
                    feedback = ValidationFeedback(action="reject", feedback=feedback_text)
                    state['validation_feedback'] = feedback
                    return state
                    
        with col3:
            if st.button("✏️ Edit"):
                st.session_state.processing = True
                edited_keywords = get_edited_keywords()
                feedback = ValidationFeedback(action="edit", edited_keywords=edited_keywords)
                state['validation_feedback'] = feedback
                return state
    
    # Display final results if available
    if state.get('final_url'):
        st.success("Processing completed!")
        st.subheader("Search Results")
        for url_data in state['final_url']:
            st.markdown(f"- [{url_data['url']}]({url_data['url']})")
            st.write(f"Scenario Score: {url_data['user_scenario']}")
            st.write(f"Problem Score: {url_data['user_problem']}")
            st.markdown("---")
    
    return state

def main():
    st.title("Patent AI Agent - Keyword Extraction System")
    
    # Input text area
    if 'current_state' not in st.session_state or st.session_state.current_state is None:
        input_text = st.text_area(
            "Enter your patent idea description:",
            height=200,
            key="input_text"
        )
        
        if st.button("Process"):
            st.session_state.processing = True
            initial_state = {
                'input_text': input_text,
                'problem': None,
                'technical': None,
                'concept_matrix': None,
                'seed_keywords': None,
                'validation_feedback': None,
                'final_keywords': None,
                'ipcs': None,
                'summary_text': None,
                'queries': None,
                'final_url': None
            }
            
            # Process until we need user input
            st.session_state.current_state = st.session_state.extractor.graph.invoke(initial_state)
    
    # Process current state and handle user actions
    if st.session_state.current_state is not None:
        updated_state = process_step(st.session_state.current_state)
        
        if updated_state and st.session_state.processing:
            # Continue processing with user feedback
            st.session_state.current_state = st.session_state.extractor.graph.invoke(updated_state)
            st.session_state.processing = False
            st.rerun()
            
        # Display final results if available
        if st.session_state.current_state.get('final_url'):
            st.subheader("Final Results")
            st.write("Patent Search URL:", st.session_state.current_state['final_url'])
            
            # Add reset button
            if st.button("Start New Search"):
                st.session_state.current_state = None
                st.session_state.keywords_displayed = False
                st.rerun()

if __name__ == "__main__":
    main()
