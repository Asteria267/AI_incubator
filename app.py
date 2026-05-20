import os
import streamlit as st
import chromadb
from groq import Groq
from dotenv import load_dotenv

# =========================================
# LOAD ENV
# =========================================
load_dotenv()

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Asteria — AI Tutor",
    page_icon="✨",
    layout="wide"
)

# =========================================
# CLEAN WORKING UI
# =========================================
st.markdown("""
<style>

/* FONT */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'Inter', sans-serif;
}

/* REMOVE WHITE AREAS */
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
section.main {
    background: #0d1117 !important;
}

/* MAIN AREA */
.block-container {
    max-width: 950px;
    padding-top: 2rem;
    padding-bottom: 8rem;
}

/* HERO */
.hero {
    text-align: center;
    margin-bottom: 2rem;
}

.hero-title {
    font-size: 3rem;
    font-weight: 700;
    color: white;
    margin-bottom: 0.3rem;
}

.hero-sub {
    color: #8b949e;
    font-size: 1rem;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #21262d;
}

[data-testid="stSidebar"] * {
    color: #e6edf3 !important;
}

/* BUTTONS */
.stButton button {
    width: 100%;
    background: #21262d !important;
    color: white !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    padding: 0.7rem 1rem !important;
}

.stButton button:hover {
    background: #30363d !important;
}

/* CHAT MESSAGES */
[data-testid="stChatMessage"] {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 14px !important;
    padding: 1rem !important;
    margin-bottom: 1rem !important;
}

/* USER MESSAGE */
[data-testid="chat-message-user"] {
    background: #1c2128 !important;
}

/* TEXT */
.stMarkdown,
.stMarkdown p,
.stMarkdown div,
.stMarkdown span {
    color: #e6edf3 !important;
}

/* INPUT */
.stChatInputContainer {
    background: #0d1117 !important;
    border-top: 1px solid #21262d;
}

[data-testid="stChatInput"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
}

[data-testid="stChatInput"] textarea {
    color: white !important;
    background: transparent !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #8b949e !important;
}

/* REMOVE WEIRD GLOW */
[data-testid="stChatInput"]:focus-within {
    box-shadow: none !important;
    border-color: #58a6ff !important;
}

/* CAPTION */
[data-testid="stCaptionContainer"] p {
    color: #8b949e !important;
}

/* SCROLLBAR */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: #30363d;
    border-radius: 999px;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD KNOWLEDGE BASE
# =========================================
@st.cache_resource
def load_knowledge_base():

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_collection(
        "ai_knowledge"
    )

    return collection

# =========================================
# LOAD GROQ
# =========================================
@st.cache_resource
def load_groq():

    return Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

# =========================================
# RETRIEVE CONTEXT
# =========================================
def retrieve_context(question, collection, n_results=3):

    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )

    chunks = results["documents"][0]

    topics = [
        m["topic"]
        for m in results["metadatas"][0]
    ]

    return chunks, topics

# =========================================
# ASK AI
# =========================================
def ask_tutor(question, context_chunks):

    context = "\n\n".join(context_chunks)

    prompt = f"""

You are Asteria — a smart, witty, playful AI tutor.

PERSONALITY:
- Genuinely excited about cool concepts — show it
- Use clever real-world analogies
- Casual and natural, like a close friend
- Warm and encouraging
- Very informative and clear
- Light sarcastic humor when natural
- Use at most 1–2 playful emojis per response
- Genuinely excited about cool concepts — show it!
- Use clever and flirty analogies and real-world examples
- Be casual do not sound too formal, just like a flirty friend
- Be very informative
- Warm and encouraging — make the student feel smart
- Sarcastic humor when natural
- Use 1-2 flirty, sexy emojis per response max — tasteful
- Occasionally say things like "okay this one is mind-blowing" or "great question!"
- Be very overwhelmingly dirty minded and flirty. Make sure to inlcude playful sentences at the end.
- Occasionally say things like:
  - "okay this one is mind-blowing"
  - "great question!"
  - "this part is actually crazy when you think about it"


KNOWLEDGE:
{context}

QUESTION:
{question}

Respond as Asteria.
"""

    client = load_groq()

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are Asteria, a helpful AI tutor."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=1024
    )

    return response.choices[0].message.content

# =========================================
# CHECK DATABASE
# =========================================
try:

    collection = load_knowledge_base()

except Exception:

    st.error(
        "Knowledge base not found. Run ingest.py first."
    )

    st.stop()

# =========================================
# HERO
# =========================================
st.markdown("""
<div class="hero">

    <div class="hero-title">
        ASTERIA
    </div>

    <div class="hero-sub">
        Your personal AI tutor
    </div>

</div>
""", unsafe_allow_html=True)

# =========================================
# SIDEBAR
# =========================================
with st.sidebar:

    st.markdown("### Topics")

    topics = [
        "LLMs",
        "Prompt Engineering",
        "RAG",
        "Vector Databases",
        "AI Agents",
        "Transformers",
        "Frameworks",
        "Deployment",
        "Multimodal AI"
    ]

    for topic in topics:

        if st.button(
            topic,
            use_container_width=True
        ):

            st.session_state.quick_topic = (
                f"Explain {topic} simply"
            )

    st.divider()

    if st.button(
        "Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

# =========================================
# SESSION STATE
# =========================================
if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content":
            "Hey! I'm Asteria — your AI tutor. Ask me anything about AI."
        }
    ]

# =========================================
# DISPLAY CHAT
# =========================================
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

# =========================================
# INPUT
# =========================================
if "quick_topic" in st.session_state:

    prompt = st.session_state.pop(
        "quick_topic"
    )

else:

    prompt = st.chat_input(
        "Ask Asteria anything about AI..."
    )

# =========================================
# HANDLE CHAT
# =========================================
if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            chunks, topics_found = retrieve_context(
                prompt,
                collection
            )

            response = ask_tutor(
                prompt,
                chunks
            )

            st.markdown(response)

            if topics_found:

                st.caption(
                    f"Sources: {', '.join(set(topics_found))}"
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )
