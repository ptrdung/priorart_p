import streamlit as st
from src.core.extractor import CoreConceptExtractor

st.set_page_config(page_title="Patent Keyword Extraction", layout="centered")

st.markdown(
    """
    <style>
    .main {background-color: #222; color: #fff;}
    .stTextInput, .stButton, .stMarkdown {background-color: #333; color: #fff;}
    .stTextInput input {background-color: #333; color: #fff;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Patent Keyword Extraction Tool")
st.markdown("#### Enter your patent idea below:")

problem = st.text_area("Problem", "", height=100)
technical = st.text_area("Technical", "", height=100)

if st.button("Run Extraction"):
    input_text = f"Problem: {problem}\nTechnical: {technical}"
    extractor = CoreConceptExtractor()
    st.info("Running extraction workflow...")
    results = extractor.extract_keywords(input_text)

    # Step 1: Concept Matrix
    st.subheader("Step 1: Concept Matrix")
    concept_matrix = results.get("concept_matrix")
    if concept_matrix:
        for k, v in concept_matrix.items():
            st.markdown(f"**{k.replace('_', ' ').title()}**: {v}")

    # Step 2: Seed Keywords
    st.subheader("Step 2: Seed Keywords")
    seed_keywords = results.get("seed_keywords")
    if seed_keywords:
        for k, v in seed_keywords.dict().items():
            st.markdown(f"**{k.replace('_', ' ').title()}**: {', '.join(v)}")

    # Step 3: Human Evaluation
    st.subheader("Step 3: Human Evaluation")
    st.markdown("Review the extracted keywords and choose your action:")
    action = st.radio("Choose action", ["Approve", "Reject", "Edit"])
    feedback = ""
    edited_keywords = {}

    if action == "Edit":
        st.markdown("Edit keywords below (comma separated):")
        for k, v in seed_keywords.dict().items():
            edited = st.text_input(f"Edit {k.replace('_', ' ').title()}", ", ".join(v))
            edited_keywords[k] = [kw.strip() for kw in edited.split(",") if kw.strip()]
        feedback = st.text_area("Feedback (optional)", "")
    elif action == "Reject":
        feedback = st.text_area("Feedback (optional)", "")

    if st.button("Submit Evaluation"):
        st.success(f"Action: {action}")
        if action == "Edit":
            st.write("Edited Keywords:", edited_keywords)
        if feedback:
            st.write("Feedback:", feedback)

    # Step 4: Results
    st.subheader("Step 4: Extraction Results")
    st.write(results)
