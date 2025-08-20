import streamlit as st
from src.core.extractor import CoreConceptExtractor, SeedKeywords

st.set_page_config(
    page_title="Patent AI Tool",
    layout="centered",
    initial_sidebar_state="auto"
)

st.markdown(
    """
    <style>
    body { background-color: #181818; color: #f5f5f5; }
    .stTextInput>div>div>input { background: #222; color: #f5f5f5; }
    .stButton>button { background: #0057b8; color: #fff; border-radius: 4px; }
    .result-box { background: #222; color: #f5f5f5; border-radius: 8px; padding: 1em; margin-top: 1em; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Patent AI Tool")
st.write("Nhập thông tin để trích xuất từ khóa sáng chế.")

problem = st.text_area("Problem", placeholder="Nhập vấn đề kỹ thuật hoặc mục tiêu...", height=100)
technical = st.text_area("Technical", placeholder="Nhập nội dung kỹ thuật hoặc bối cảnh...", height=100)

if "results" not in st.session_state:
    st.session_state.results = None
if "keywords_action" not in st.session_state:
    st.session_state.keywords_action = None
if "edit_data" not in st.session_state:
    st.session_state.edit_data = None
if "feedback_text" not in st.session_state:
    st.session_state.feedback_text = ""
if "confirmed" not in st.session_state:
    st.session_state.confirmed = False
if "related_keywords" not in st.session_state:
    st.session_state.related_keywords = None

if st.button("Extract Keywords"):
    if not problem and not technical:
        st.warning("Vui lòng nhập thông tin vào cả hai ô.")
    else:
        with st.spinner("Đang xử lý..."):
            input_text = f"Problem: {problem}\nTechnical: {technical}"
            extractor = CoreConceptExtractor()
            results = extractor.extract_keywords(input_text)
            st.session_state.results = results
            st.session_state.keywords_action = None
            st.session_state.edit_data = None
            st.session_state.feedback_text = ""
            st.session_state.confirmed = False
            st.session_state.related_keywords = None

if st.session_state.results and not st.session_state.confirmed:
    st.markdown("### Kết quả trích xuất", unsafe_allow_html=True)
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    for key, value in st.session_state.results.items():
        if value is None:
            continue
        st.write(f"**{key}**:")
        if hasattr(value, "dict"):
            for subkey, subval in value.dict().items():
                st.write(f"- {subkey}: {subval}")
        elif isinstance(value, dict):
            for subkey, subval in value.items():
                st.write(f"- {subkey}: {subval}")
        elif isinstance(value, list):
            for i, item in enumerate(value, 1):
                if isinstance(item, dict):
                    st.write(f"{i}. " + ", ".join([f"{k}: {v}" for k, v in item.items()]))
                else:
                    st.write(f"{i}. {item}")
        else:
            st.write(value)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Đánh giá kết quả")
    action = st.radio("Chọn hành động:", ["Phê duyệt", "Từ chối", "Chỉnh sửa"], key="keywords_action")
    if action == "Từ chối":
        st.session_state.feedback_text = st.text_area("Lý do từ chối", value=st.session_state.feedback_text)
    if action == "Chỉnh sửa":
        seed_keywords = st.session_state.results.get("seed_keywords")
        edit_data = {}
        if seed_keywords:
            for field in ["problem_purpose", "object_system", "environment_field"]:
                current = getattr(seed_keywords, field, [])
                edit_data[field] = st.text_input(
                    f"{field.replace('_', ' ').title()} (phân tách bằng dấu phẩy)",
                    value=", ".join(current),
                    key=f"edit_{field}"
                )
            st.session_state.edit_data = edit_data

    if st.button("Xác nhận đánh giá"):
        extractor = CoreConceptExtractor()
        seed_keywords = st.session_state.results.get("seed_keywords")
        concept_matrix = st.session_state.results.get("concept_matrix")
        state = {
            "concept_matrix": concept_matrix,
            "seed_keywords": seed_keywords,
            "validation_feedback": None
        }
        if st.session_state.keywords_action == "Phê duyệt":
            feedback = extractor.step3_human_evaluation(state, action="approve")
        elif st.session_state.keywords_action == "Từ chối":
            feedback = extractor.step3_human_evaluation(state, action="reject", feedback_text=st.session_state.feedback_text)
        elif st.session_state.keywords_action == "Chỉnh sửa":
            edited_data = {}
            for field in ["problem_purpose", "object_system", "environment_field"]:
                raw = st.session_state.edit_data[field]
                edited_data[field] = [kw.strip() for kw in raw.split(",") if kw.strip()]
            feedback = extractor.step3_human_evaluation(
                state,
                action="edit",
                edited_keywords=SeedKeywords(**edited_data)
            )
        st.success(f"Đã xác nhận: {feedback['validation_feedback'].action}")
        st.session_state.results["validation_feedback"] = feedback["validation_feedback"]
        st.session_state.confirmed = True

if st.session_state.results and st.session_state.confirmed:
    st.markdown("### Từ khoá liên quan")
    extractor = CoreConceptExtractor()
    seed_keywords = st.session_state.results.get("seed_keywords")
    concept_matrix = st.session_state.results.get("concept_matrix")
    state = {
        "concept_matrix": concept_matrix,
        "seed_keywords": seed_keywords,
        "validation_feedback": st.session_state.results.get("validation_feedback")
    }
    related = extractor.gen_key(state)
    st.session_state.related_keywords = related.get("final_keywords")
    if st.session_state.related_keywords:
        for key, value in st.session_state.related_keywords.items():
            st.write(f"**{key}**:")
            for v in value:
                st.write(f"- {v}")
