import streamlit as st
from src.core.extractor import CoreConceptExtractor
import json
from typing import Dict, Any

def display_concept_matrix(concept_matrix):
    st.subheader("Concept Matrix")
    cols = st.columns(3)
    with cols[0]:
        st.write("Problem/Purpose:")
        st.write(concept_matrix.problem_purpose)
    with cols[1]:
        st.write("Object/System:")
        st.write(concept_matrix.object_system)
    with cols[2]:
        st.write("Environment/Field:")
        st.write(concept_matrix.environment_field)

def display_keywords(seed_keywords):
    st.subheader("Generated Keywords")
    cols = st.columns(3)
    with cols[0]:
        st.write("Problem Keywords:")
        st.write("\n".join(seed_keywords.problem_purpose))
    with cols[1]:
        st.write("Technical Keywords:")
        st.write("\n".join(seed_keywords.object_system))
    with cols[2]:
        st.write("Field Keywords:")
        st.write("\n".join(seed_keywords.environment_field))

def main():
    st.set_page_config(
        page_title="Patent AI Agent",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Patent AI Agent - Keyword Extraction System")

    # Initialize session state for storing the extractor
    if 'extractor' not in st.session_state:
        st.session_state.extractor = CoreConceptExtractor()
        st.session_state.current_state = None
        st.session_state.processing = False
        st.session_state.edited_keywords = None
        st.session_state.waiting_for_input = False

    # Input text area
    if not st.session_state.processing:
        input_text = st.text_area(
            "Enter your patent idea description:",
            height=200,
            placeholder="Describe your patent idea here..."
        )

        if st.button("Process", type="primary"):
            if input_text:
                st.session_state.processing = True
                st.session_state.current_state = {
                    "input_text": input_text,
                    "problem": None,
                    "technical": None,
                    "concept_matrix": None,
                    "seed_keywords": None,
                    "validation_feedback": None,
                    "final_keywords": None,
                    "ipcs": None,
                    "summary_text": None,
                    "queries": None,
                    "final_url": None
                }
                st.rerun()

    # Process the input and show results
    if st.session_state.processing:
        if not st.session_state.waiting_for_input:
            # First run - go until we get keywords
            st.session_state.current_state = st.session_state.extractor.run_until_keywords(
                st.session_state.current_state["input_text"]
            )
            st.session_state.waiting_for_input = True
        elif st.session_state.current_state.get("validation_feedback"):
            # After user action, complete the pipeline
            st.session_state.current_state = st.session_state.extractor.complete_pipeline(
                st.session_state.current_state
            )

        if st.session_state.current_state.get("concept_matrix"):
            display_concept_matrix(st.session_state.current_state["concept_matrix"])

        if st.session_state.current_state.get("seed_keywords"):
            display_keywords(st.session_state.current_state["seed_keywords"])

            # Human evaluation interface
            st.subheader("Evaluate Generated Keywords")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("✅ Approve"):
                    st.session_state.current_state["validation_feedback"] = {"action": "approve"}
                    st.session_state.waiting_for_input = False
                    st.rerun()

            with col2:
                if st.button("🔄 Reject"):
                    st.session_state.current_state["validation_feedback"] = {"action": "reject"}
                    st.session_state.waiting_for_input = False
                    st.rerun()

            with col3:
                if st.button("✏️ Edit"):
                    st.session_state.editing = True

            if st.session_state.get("editing", False):
                st.subheader("Edit Keywords")
                edited_keywords = {
                    "problem_purpose": st.text_area(
                        "Problem/Purpose Keywords (one per line)",
                        value="\n".join(st.session_state.current_state["seed_keywords"].problem_purpose)
                    ).split("\n"),
                    "object_system": st.text_area(
                        "Object/System Keywords (one per line)",
                        value="\n".join(st.session_state.current_state["seed_keywords"].object_system)
                    ).split("\n"),
                    "environment_field": st.text_area(
                        "Environment/Field Keywords (one per line)",
                        value="\n".join(st.session_state.current_state["seed_keywords"].environment_field)
                    ).split("\n")
                }

                if st.button("Submit Edits"):
                    st.session_state.current_state["seed_keywords"] = edited_keywords
                    st.session_state.current_state["validation_feedback"] = {"action": "edit"}
                    st.session_state.editing = False
                    st.session_state.waiting_for_input = False
                    st.rerun()

        # Display final results if available
        if st.session_state.current_state.get("final_url"):
            st.subheader("Final Results")
            st.write("Generated Search URL:")
            st.write(st.session_state.current_state["final_url"])

            if st.button("Start New Search"):
                st.session_state.processing = False
                st.session_state.current_state = None
                st.session_state.waiting_for_input = False
                st.session_state.editing = False
                st.rerun()

if __name__ == "__main__":
    main()
