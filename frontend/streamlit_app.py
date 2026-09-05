import os
import time
import requests
import streamlit as st

api_url = os.getenv("API_URL", "http://localhost:8000")

# st.cache_data
@st.cache_data(show_spinner=False, ttl=3600) # cache 1 tiếng
def call_fastapi_backend(question: str):
    response = requests.post(f"{api_url}/api/chat/ask", json={"question": question})
    if response.status_code == 200:
        return response.json().get("answer", "Lỗi: Không có câu trả lời.")
    else:
        return f"Lỗi Server: {response.status_code}"

# Streamed response emulator
def response_generator(answer):
    for word in answer.split(" "):
        yield word + " "
        time.sleep(0.05)

st.set_page_config(page_title="LawRAG Chatbot", page_icon="⚖️", layout="centered")
st.title("⚖️ Hệ Thống Hỏi Đáp Pháp Luật")
st.markdown('''
    :red[Chào] :orange[mừng] :green[bạn] :blue[đến] :violet[với]
    :yellow[Chatbot] :rainbow[hỏi] đáp :blue[văn] :yellow[bản] 
    :green[pháp] :rainbow[luật].''')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
    {"role": "assistant", "content": "Xin chào bạn, tôi có thể giúp được gì cho bạn?"}
]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar:
    st.header("📁 Thêm văn bản pháp luật")
    uploaded_file = st.file_uploader(
        "Upload file để hệ thống xử lý tài liệu bạn muốn",
        accept_multiple_files=False,
        type=["pdf","docx"]
    )
    if st.button("Xác nhận xử lý File"):
        if uploaded_file is not None:
            with st.spinner("Đang xử lý dữ liệu",show_time=True):
                # FastAPI
                try:
                    rq = requests.post(
                        f"{api_url}/api/upload/uploadfile",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue()
                            )
                        }
                    )
                    if rq.status_code == 200:
                        st.success(f"Đã xử lý thành công: {uploaded_file.name}!", icon="✅")
                    else:
                        st.error(f"Lỗi hệ thống ({rq.status_code}): {rq.text}", icon="❌")
                except Exception as e:
                    st.error(e)


if prompt := st.chat_input("Xin nhập câu hỏi của bạn"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    # FastAPI
    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ...",show_time=True):
            try:
                bot_reply = call_fastapi_backend(prompt)
                st.write_stream(response_generator(bot_reply))
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            except Exception as e:
                st.error(e)



