import os
import json
import time
import streamlit as st
import chromadb
from groq import Groq
from dotenv import load_dotenv

# =========================================
# ENVIRONMENT
# =========================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =========================================
# APP CONFIG
# =========================================
st.set_page_config(
    page_title="Asteria AI Tutor Pro",
    page_icon="🧠",
    layout="wide"
)

# =========================================
# THEME (ADVANCED UI SYSTEM)
# =========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp {
    background: #0d1117;
    color: #e6edf3;
    font-family: 'Inter';
}

.block-container {
    max-width: 1000px;
    padding-top: 2rem;
    padding-bottom: 6rem;
}

/* CHAT CARDS */
[data-testid="stChatMessage"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: #161b22;
}

/* BUTTONS */
.stButton button {
    width: 100%;
    border-radius: 10px;
    background: #21262d;
    color: white;
}
.stButton button:hover {
    background: #30363d;
}

/* INPUT */
[data-testid="stChatInput"] textarea {
    background: #161b22;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =========================================
# CLIENTS
# =========================================
@st.cache_resource
def get_clients():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection("ai_knowledge")
    llm = Groq(api_key=GROQ_API_KEY)
    return collection, llm

collection, llm = get_clients()

# =========================================
# SESSION STATE INIT
# =========================================
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "progress" not in st.session_state:
        st.session_state.progress = {
            "topics_learned": [],
            "questions_asked": 0,
            "quiz_score": []
        }

    if "mode" not in st.session_state:
        st.session_state.mode = "tutor"

init_state()

# =========================================
# RAG PIPELINE (CLEAN ARCHITECTURE)
# =========================================
def retrieve(query, k=4):
    results = collection.query(
        query_texts=[query],
        n_results=k
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    return docs, metas

# =========================================
# CHAT HISTORY ENGINE
# =========================================
def build_history(limit=10):
    msgs = st.session_state.messages[-limit:]
    return "\n".join([f"{m['role']}: {m['content']}" for m in msgs])

# =========================================
# LESSON ENGINE (IMPORTANT UPGRADE)
# =========================================
def lesson_mode_prompt(topic, context):
    return f"""
You are Asteria, an elite AI tutor.

TASK: Teach the topic step-by-step.

TOPIC: {topic}

CONTEXT:
{context}

INSTRUCTIONS:
1. Start with a simple explanation
2. Then give structured breakdown
3. Then provide analogy
4. Then give 3 key bullet points
5. End with a mini-check question

Be clear, structured, and educational.
"""

# =========================================
# QUIZ ENGINE
# =========================================
def generate_quiz(topic):
    return f"""
Create a short quiz (3 questions) on: {topic}

Rules:
- Mix conceptual + reasoning questions
- Do NOT provide answers yet
- Make it beginner-friendly
"""

# =========================================
# MAIN AI ENGINE
# =========================================
def ask_ai(user_input, context, history, mode):

    base_system = """
You are Asteria, an advanced AI tutor system.

CORE RULE:
- You teach, not just answer
- You guide thinking step-by-step
- You are structured and logical
- You avoid hallucination
"""

    if mode == "quiz":
        user_prompt = generate_quiz(user_input)

    elif mode == "lesson":
        user_prompt = lesson_mode_prompt(user_input, context)

    else:
        user_prompt = f"""
CONTEXT:
{context}

HISTORY:
{history}

QUESTION:
{user_input}
"""

    try:
        res = llm.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": base_system},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=900
        )

        return res.choices[0].message.content

    except Exception as e:
        return f"AI Error: {str(e)}"

# =========================================
# SIDEBAR CONTROL PANEL
# =========================================
with st.sidebar:

    st.title("🧠 Control Panel")

    st.session_state.mode = st.selectbox(
        "Learning Mode",
        ["tutor", "lesson", "quiz"]
    )

    st.divider()

    topics = ["LLMs", "RAG", "Transformers", "Agents", "Prompt Engineering"]

    st.markdown("### Quick Topics")

    for t in topics:
        if st.button(t):
            st.session_state.quick = t

    st.divider()

    if st.button("Reset Progress"):
        st.session_state.progress = {
            "topics_learned": [],
            "questions_asked": 0,
            "quiz_score": []
        }

    if st.button("Clear Chat"):
        st.session_state.messages = []

# =========================================
# HEADER
# =========================================
st.markdown("""
<h1 style='text-align:center;'>🧠 ASTERIA PRO</h1>
<p style='text-align:center; color:#8b949e;'>
AI Tutor with Memory • RAG • Lessons • Quiz System
</p>
""", unsafe_allow_html=True)

# =========================================
# SHOW CHAT
# =========================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================================
# INPUT SYSTEM
# =========================================
if "quick" in st.session_state:
    user_input = f"Teach me about {st.session_state.pop('quick')}"
else:
    user_input = st.chat_input("Ask Asteria...")

# =========================================
# CHAT EXECUTION
# =========================================
if user_input:

    st.session_state.progress["questions_asked"] += 1

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing knowledge..."):

            context, metas = retrieve(user_input)
            history = build_history()

            response = ask_ai(
                user_input,
                "\n\n".join(context),
                history,
                st.session_state.mode
            )

            st.markdown(response)

            if metas:
                topics = list(set([m.get("topic", "") for m in metas if m]))
                st.caption(f"Sources: {', '.join(topics)}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
