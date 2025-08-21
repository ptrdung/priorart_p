import streamlit as st
import json
from src.core.extractor import CoreConceptExtractor

st.set_page_config(
    page_title="Patent AI Agent",
    page_icon="🔍",
    layout="wide"
)

def main():
    st.title("🔍 Patent AI Agent - Keyword Extraction System")
    st.markdown("""
    ### Tìm kiếm patents tương tự dựa trên mô tả ý tưởng
    Hệ thống sẽ phân tích ý tưởng của bạn và tìm các patents liên quan sử dụng AI
    """)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        model_name = st.selectbox(
            "Chọn mô hình",
            ["qwen2.5:3b-instruct"],
            index=0
        )
        use_checkpointer = st.checkbox("Sử dụng checkpointer", value=True)

    # Main input area
    st.header("📝 Nhập mô tả ý tưởng của bạn")
    
    # Template for input
    template = """
    **Idea title**: [Tên ý tưởng]

    **User scenario**: [Mô tả tình huống sử dụng]

    **User problem**: [Vấn đề cần giải quyết]
    """
    
    input_text = st.text_area(
        "Mô tả ý tưởng của bạn theo template sau:",
        template,
        height=300
    )

    if st.button("🚀 Bắt đầu phân tích"):
        if input_text == template or len(input_text.strip()) < 50:
            st.error("Vui lòng nhập mô tả ý tưởng chi tiết!")
            return

        # Initialize session state if not exists
        if 'extractor' not in st.session_state:
            st.session_state.extractor = CoreConceptExtractor(
                model_name=model_name,
                use_checkpointer=use_checkpointer
            )
            st.session_state.current_state = None
            st.session_state.phase = "initial"  # Track current phase
            st.session_state.validation_feedback = None

        with st.spinner("🤖 Đang phân tích ý tưởng..."):
            try:
                if st.session_state.phase == "initial":
                    # First run until keyword generation
                    results = st.session_state.extractor.extract_keywords(input_text)
                    st.session_state.current_state = results
                    st.session_state.phase = "evaluation"
                elif st.session_state.phase == "continue" and st.session_state.validation_feedback:
                    action = st.session_state.validation_feedback["action"]
                    
                    # Update current state with feedback
                    st.session_state.current_state["validation_feedback"] = st.session_state.validation_feedback
                    
                    # Process based on action type
                    results = st.session_state.extractor.extract_keywords(
                        input_text,
                        continue_from_state=st.session_state.current_state,
                        action=action
                    )
                    
                    if action == "reject":
                        # Stay in evaluation phase for new keywords
                        st.session_state.current_state = results
                        st.session_state.phase = "evaluation"
                        st.session_state.validation_feedback = None
                    else:
                        # Continue to completion for approve/edit
                        st.session_state.current_state = results
                        st.session_state.phase = "completed"
                else:
                    results = st.session_state.current_state
                
                # Display results in tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Phân tích cơ bản",
                    "🎯 Từ khóa chính",
                    "👤 Đánh giá & Chỉnh sửa",
                    "🔍 Kết quả tìm kiếm",
                    "📑 Chi tiết"
                ])
                
                with tab1:
                    st.subheader("Phân tích cơ bản")
                    if results.get("problem"):
                        st.markdown("### 🎯 Vấn đề chính")
                        st.write(results["problem"])
                    if results.get("technical"):
                        st.markdown("### 💡 Đặc điểm kỹ thuật")
                        st.write(results["technical"])
                    if results.get("ipcs"):
                        st.markdown("### 📑 Phân loại IPC")
                        st.write(results["ipcs"])

                with tab2:
                    st.subheader("Từ khóa trích xuất")
                    if results.get("seed_keywords"):
                        seed_keywords = results["seed_keywords"]
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("### 🎯 Vấn đề & Mục đích")
                            for kw in seed_keywords.problem_purpose:
                                st.markdown(f"- {kw}")
                                
                        with col2:
                            st.markdown("### 🔧 Đối tượng & Hệ thống")
                            for kw in seed_keywords.object_system:
                                st.markdown(f"- {kw}")
                                
                        with col3:
                            st.markdown("### 🌍 Môi trường & Lĩnh vực")
                            for kw in seed_keywords.environment_field:
                                st.markdown(f"- {kw}")

                with tab3:
                    st.subheader("Đánh giá và Chỉnh sửa Từ khóa")
                    if results.get("seed_keywords"):
                        seed_keywords = results["seed_keywords"]
                        
                        st.markdown("### 🤖 Từ khóa được đề xuất")
                        st.markdown("Vui lòng xem xét và đánh giá các từ khóa được trích xuất:")
                        
                        # Display current keywords in editable text areas
                        col1, col2, col3 = st.columns(3)
                        
                        edited_keywords = {}
                        with col1:
                            st.markdown("#### 🎯 Vấn đề & Mục đích")
                            edited_keywords["problem_purpose"] = st.text_area(
                                "Chỉnh sửa từ khóa (phân cách bằng dấu phẩy)",
                                value=", ".join(seed_keywords.problem_purpose),
                                key="edit_problem"
                            ).split(",")
                            
                        with col2:
                            st.markdown("#### 🔧 Đối tượng & Hệ thống")
                            edited_keywords["object_system"] = st.text_area(
                                "Chỉnh sửa từ khóa (phân cách bằng dấu phẩy)",
                                value=", ".join(seed_keywords.object_system),
                                key="edit_object"
                            ).split(",")
                            
                        with col3:
                            st.markdown("#### 🌍 Môi trường & Lĩnh vực")
                            edited_keywords["environment_field"] = st.text_area(
                                "Chỉnh sửa từ khóa (phân cách bằng dấu phẩy)",
                                value=", ".join(seed_keywords.environment_field),
                                key="edit_environment"
                            ).split(",")
                        
                        st.markdown("---")
                        col1, col2, col3 = st.columns([1,1,2])
                        
                        # Add rejection feedback field if needed
                        reject_feedback = None
                        if "show_reject_feedback" not in st.session_state:
                            st.session_state.show_reject_feedback = False

                        with col1:
                            approve_button = st.button(
                                "✅ Chấp nhận", 
                                type="primary", 
                                disabled=st.session_state.phase == "completed",
                                key="approve_button"
                            )
                            if approve_button:
                                # Cập nhật validation feedback
                                validation_feedback = {
                                    "action": "approve",
                                    "feedback": None,
                                    "edited_keywords": None
                                }
                                
                                # Cập nhật state hiện tại với feedback
                                current_state = st.session_state.current_state.copy()
                                current_state["validation_feedback"] = validation_feedback
                                
                                # Chạy tiếp pipeline với state đã cập nhật
                                results = st.session_state.extractor.extract_keywords(
                                    input_text,
                                    continue_from_state=current_state,
                                    action="approve"
                                )
                                
                                # Cập nhật session state
                                st.session_state.current_state = results
                                st.session_state.validation_feedback = validation_feedback
                                st.session_state.phase = "completed"
                                st.rerun()
                                
                        with col2:
                            reject_button = st.button(
                                "❌ Từ chối & Tạo lại", 
                                disabled=st.session_state.phase == "completed",
                                key="reject_button"
                            )
                            if reject_button:
                                st.session_state.show_reject_feedback = True
                                
                            if st.session_state.show_reject_feedback:
                                reject_feedback = st.text_area(
                                    "Phản hồi cho việc tạo lại:",
                                    key="reject_feedback_input"
                                )
                                if st.button("Xác nhận từ chối", key="confirm_reject"):
                                    st.session_state.validation_feedback = {
                                        "action": "reject",
                                        "feedback": reject_feedback,
                                        "edited_keywords": None
                                    }
                                    st.session_state.phase = "continue"
                                    st.session_state.show_reject_feedback = False
                                    st.rerun()
                                
                        with col3:
                            edit_button = st.button(
                                "✏️ Lưu chỉnh sửa", 
                                disabled=st.session_state.phase == "completed",
                                key="edit_button"
                            )
                            if edit_button:
                                # Validate edited keywords
                                valid_edits = all(
                                    any(k.strip() for k in keywords)
                                    for keywords in edited_keywords.values()
                                )
                                if valid_edits:
                                    st.session_state.validation_feedback = {
                                        "action": "edit",
                                        "feedback": None,
                                        "edited_keywords": {
                                            "problem_purpose": [k.strip() for k in edited_keywords["problem_purpose"] if k.strip()],
                                            "object_system": [k.strip() for k in edited_keywords["object_system"] if k.strip()],
                                            "environment_field": [k.strip() for k in edited_keywords["environment_field"] if k.strip()]
                                        }
                                    }
                                    st.session_state.phase = "continue"
                                    st.rerun()
                                else:
                                    st.error("Vui lòng đảm bảo mỗi danh mục có ít nhất một từ khóa")
                
                with tab4:
                    st.subheader("Kết quả tìm kiếm Patents")
                    if results.get("final_url"):
                        for idx, url in enumerate(results["final_url"], 1):
                            st.markdown(f"### Patent {idx}")
                            st.markdown(f"[Xem chi tiết]({url})")
                            
                with tab5:
                    st.subheader("Chi tiết kết quả")
                    st.json(results)

            except Exception as e:
                st.error(f"❌ Có lỗi xảy ra: {str(e)}")

if __name__ == "__main__":
    main()
