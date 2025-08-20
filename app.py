# app.py
import streamlit as st
from src.core.extractor import CoreConceptExtractor

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
st.write("Nhập thông tin để trích xuất từ khóa sáng chế. Giao diện tối, đơn giản, dễ nhìn.")

problem = st.text_area("Problem", placeholder="Nhập vấn đề kỹ thuật hoặc mục tiêu...", height=100)
technical = st.text_area("Technical", placeholder="Nhập nội dung kỹ thuật hoặc bối cảnh...", height=100)

if st.button("Extract Keywords"):
    if not problem and not technical:
        st.warning("Vui lòng nhập thông tin vào cả hai ô.")
    else:
        with st.spinner("Đang xử lý..."):
            input_text = f"Problem: {problem}\nTechnical: {technical}"
            extractor = CoreConceptExtractor()
            results = extractor.extract_keywords(input_text)
        st.markdown("### Kết quả trích xuất", unsafe_allow_html=True)
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        for key, value in results.items():
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
