import streamlit as st
from src.core.extractor import CoreConceptExtractor
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
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Core Keywords:")
            st.text_area("Core", "\n".join(seed_keywords.core), key="core_edit")
        with col2:
            st.write("Technical Keywords:")
            st.text_area("Technical", "\n".join(seed_keywords.technical), key="tech_edit")
        with col3:
            st.write("Field Keywords:")
            st.text_area("Field", "\n".join(seed_keywords.field), key="field_edit")

def get_edited_keywords():
    from src.core.extractor import KeywordSet
    return KeywordSet(
        core=st.session_state.core_edit.split('\n'),
        technical=st.session_state.tech_edit.split('\n'),
        field=st.session_state.field_edit.split('\n')
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
        
        # Show action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Approve"):
                st.session_state.processing = True
                state['validation_feedback'] = {'action': 'approve'}
                return state
        with col2:
            if st.button("Reject"):
                st.session_state.processing = True
                state['validation_feedback'] = {'action': 'reject'}
                return state
        with col3:
            if st.button("Edit"):
                st.session_state.processing = True
                edited_keywords = get_edited_keywords()
                state['validation_feedback'] = {'action': 'edit', 'edited_keywords': edited_keywords}
                return state
    
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
