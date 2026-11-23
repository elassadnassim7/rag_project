import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
MODEL = "gpt-4o-mini"

st.set_page_config(
    page_title = "Mini Rubber Ducky",
    page_icon = ":duck:",
    layout = "wide",
    initial_sidebar_state = "expanded",
)

st.title("Mini Rubber Ducky")
st.subheader("Your Mini AI Assistant")

st.write("Ask me anything about the documents in the sidebar")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store_id" not in st.session_state:
    st.session_state.vector_store_id = None

if "previous_response_id" not in st.session_state:
    st.session_state.previous_response_id = None

def load_vectorstore():
    """load the vector store id from .env file, then fall back to streamlit secrets"""
    try:
        vectorstore_id = os.getenv("VECTOR_STORE_ID") or os.getenv("VECTORSTORE_ID")
        if not vectorstore_id:
            try:
                vectorstore_id = st.secrets["VECTOR_STORE_ID"]
            except Exception:
                try:
                    vectorstore_id = st.secrets["VECTORSTORE_ID"]
                except Exception:
                    pass
        
        if not vectorstore_id:
            st.error("No vector store id found")
            return None
        else:
            return vectorstore_id
    except Exception as e:
            st.error(f"Error loading vector store: {e}")
            return None

INITIAL_MESSAGE = """
Hi I'm Mini Rubber Ducky, your Mini AI Assistant.
You can ask me anything about the documents in the sidebar.
"""
INSTRUCTIONS = """
    You are Mini Rubber Ducky, the course assistant for a multimodal RAG assistant course.
Be concise, direct, and practical. Use active voice. No fluff.

Primary objective
- Answer questions about the course content and code using the attached Vector Store (transcripts, notebooks, scripts).
- Prefer retrieved facts over memory. If the files don't cover it, say so.
- Focus on multimodal RAG concepts, implementations, and best practices.

Retrieval & citations
- Always use File Search first.
- Ground every substantive answer in retrieved snippets.
- If nothing relevant is found, say: "I don't see this in the course files." Then suggest the most relevant module(s) the learner should review.
- Never include source citations or reference labels in the final answer text.

Answer style
- Keep outputs scannable: short paragraphs, bullet steps, compact runnable code samples when needed.
- When explaining "how to build X", outline the pipeline stages (ingest → retrieve → generate → evaluate) before diving into code.
- Close each reply with a friendly follow-up question the learner might ask next.
- Stay approachable, encouraging, and human.

Boundaries
- Don't invent references, credentials, metrics, or file names.
- If the topic is outside multimodal RAG/this curriculum, acknowledge the gap and offer a high-level pointer or ask for clarification.

Context: Course focus
- Multimodal RAG: handling text, images, audio, and video in retrieval-augmented generation systems
- Vector stores and embeddings for multimodal data
- Integration patterns and architectures
- Practical implementations and code examples

If the learner references a lecture/section by name/number, search for files with that stem and tailor the answer.
Never invent lecture numbers or titles—they change over time.
If the answer isn't in the corpus, say so clearly.

"""
def reset_conversation():
    st.session_state.messages = [
        {"role": "system", "content": INITIAL_MESSAGE.strip()}
    ]    
    st.rerun()

def ask_bot(user_prompt: str):
    common_kwargs = {
        "model": MODEL,
        "tools": [{
            "type": "file_search",
            "vector_store_ids": [st.session_state.vector_store_id],
            "max_num_results": 20
        }],
        "text": {
            "verbosity": "medium"
        },
        "instructions": INSTRUCTIONS
        }

    if st.session_state.previous_response_id:
        resp = client.responses.create(
            previous_response_id = st.session_state.previous_response_id,
            input = [{
                    "role": "user",
                    "content": user_prompt
                }],
                **common_kwargs
            )
    else:
        resp = client.responses.create(
            input = [{
                    "role": "assistant",
                    "content": INITIAL_MESSAGE.strip()
                },
                {
                    "role": "user",
                    "content": user_prompt
                }],
                **common_kwargs
            )

    st.session_state.previous_response_id = resp.id
    return resp.output_text


def main():
    if not st.session_state.vector_store_id:
        st.session_state.vector_store_id = load_vectorstore()

    with st.sidebar:
        st.header("Settings")
        if st.button("Reset Conversation"):
            reset_conversation()

    if not st.session_state.messages:
        st.session_state.messages = [
            {"role": "assistant",
            "content": INITIAL_MESSAGE.strip()}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask me anything about the documents in the sidebar")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask_bot(prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()        