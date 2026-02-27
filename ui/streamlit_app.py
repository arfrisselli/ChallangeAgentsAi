"""
Streamlit UI: chat with text input, message history, and streaming from FastAPI.
No dedicated buttons for tools; the model decides when to call each tool.
"""
import os
import json
import streamlit as st
import httpx

API_URL = os.getenv("API_URL", "http://localhost:8000")


def stream_chat(message: str, conversation_id: str | None, placeholder=None) -> str:
    """Call POST /chat/stream; if placeholder is given, update it with each token. Returns full response."""
    full = []
    with httpx.stream(
        "POST",
        f"{API_URL}/chat/stream",
        json={"message": message, "conversation_id": conversation_id},
        timeout=120.0,
    ) as r:
        for line in r.iter_lines():
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("type") == "token" and obj.get("content"):
                    full.append(obj["content"])
                    if placeholder is not None:
                        placeholder.markdown("".join(full) + "▌")
            except json.JSONDecodeError:
                pass
    return "".join(full)


def chat_no_stream(message: str, conversation_id: str | None) -> str:
    """Call POST /chat and return response."""
    r = httpx.post(
        f"{API_URL}/chat",
        json={"message": message, "conversation_id": conversation_id},
        timeout=120.0,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")


st.set_page_config(page_title="ChallangeAgentsAi",
                   page_icon="🤖", layout="wide")
st.title("ChallangeAgentsAi – Assistente RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Digite sua pergunta (busca web, docs, SQL, tempo)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        # Show loading indicator
        with placeholder.container():
            with st.spinner("Processando sua pergunta..."):
                try:
                    # Stream response token-by-token
                    full_response = stream_chat(
                        prompt, st.session_state.conversation_id, None)
                    if not full_response:
                        full_response = chat_no_stream(
                            prompt, st.session_state.conversation_id)
                except Exception as e:
                    full_response = f"Erro: {e}. Verifique se a API está em " + API_URL

        # Clear loading and show final response
        placeholder.markdown(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response})

with st.sidebar:
    st.header("Info")
    st.caption(
        "O modelo escolhe quando usar: busca web, documentos, SQL ou previsão do tempo.")
    if st.button("Limpar histórico"):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.rerun()
