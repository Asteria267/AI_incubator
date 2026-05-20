import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document

# =========================================================
# LOAD ENV
# =========================================================
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Asteria AI Academy",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# SAFE CSS (READABLE + DARK THEME)
# =========================================================
st.markdown("""
<style>

html, body, .stApp {
    background-color: #0b1220;
    color: #e5e7eb;
    font-family: Inter, sans-serif;
}

h1, h2, h3 {
    color: #ffffff !important;
}

p, div, span {
    color: #e5e7eb !important;
    line-height: 1.6;
}

.lesson-card {
    background: #111827;
    padding: 2rem;
    border-radius: 18px;
    border: 1px solid #1f2937;
    margin-bottom: 2rem;
}

.stButton button {
    background: #2563eb;
    color: white !important;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    border: none;
}

.stButton button:hover {
    background: #3b82f6;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================
if "xp" not in st.session_state:
    st.session_state.xp = 0

if "uploaded_text" not in st.session_state:
    st.session_state.uploaded_text = ""

# =========================================================
# LESSON DATA (LONG + CLEAN + SAFE)
# =========================================================
LESSONS = {
    "Intro to LLMs": {
        "image": "https://images.unsplash.com/photo-1677442136019-21780ecad995",
        "hook": "Imagine an AI that has read almost the entire internet… and learned to talk like a human.",
        "content": """Large Language Models (LLMs) are AI systems trained on massive datasets of text.

They do not “think” like humans.
Instead, they predict the next word based on patterns.

This simple idea becomes powerful because:
- they see billions of examples
- they learn language structure
- they adapt to context

That’s why they can:
- write essays
- answer questions
- generate code
- explain concepts
"""
    },

    "Transformers": {
        "image": "https://images.unsplash.com/photo-1620712943543-bcc4688e7485",
        "hook": "Transformers are the architecture that made modern AI possible.",
        "content": """Before Transformers, AI struggled with long text.

Transformers introduced ATTENTION.

Attention allows AI to:
- focus on important words
- ignore irrelevant ones
- understand relationships

This is why AI now understands context so well.

It is the foundation of:
- ChatGPT
- Claude
- Gemini
"""
    },

    "RAG Systems": {
        "image": "https://images.unsplash.com/photo-1555949963-ff9fe0c870eb",
        "hook": "RAG helps AI use real documents instead of guessing.",
        "content": """RAG = Retrieval Augmented Generation.

Instead of relying only on memory, AI:
1. searches documents
2. finds relevant info
3. uses it to answer

This makes AI:
- more accurate
- less hallucinating
- more useful in real apps
"""
    },

    "Embeddings": {
        "image": "https://images.unsplash.com/photo-1551288049-bebda4e38f71",
        "hook": "Embeddings turn language into math.",
        "content": """Embeddings convert words into numbers.

Similar meanings → close vectors
Different meanings → far vectors

This enables:
- semantic search
- recommendations
- RAG systems
"""
    },

    "AI Agents": {
        "image": "https://images.unsplash.com/photo-1531746790731-6c087fecd65a",
        "hook": "AI agents don’t just answer — they act.",
        "content": """AI agents can:
- plan tasks
- use tools
- execute steps

Example:
Instead of answering, they can:
- search
- analyze
- decide
- complete tasks

They are the future of AI systems.
"""
    }
}

# =========================================================
# QUIZZES
# =========================================================
QUIZZES = {
    "Intro to LLMs": ("What do LLMs learn?", ["Patterns", "Memories", "Images"], "Patterns"),
    "Transformers": ("What powers Transformers?", ["Attention", "Storage", "CPU"], "Attention"),
    "RAG Systems": ("What does RAG do?", ["Search docs", "Delete files", "Random guess"], "Search docs"),
    "Embeddings": ("Embeddings represent?", ["Meaning", "Colors", "Sound"], "Meaning"),
    "AI Agents": ("Agents can?", ["Act", "Only chat", "Only store"], "Act")
}

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.title("🧠 Asteria AI Academy")
    st.metric("XP", st.session_state.xp)

    selected_lesson = st.radio("Lessons", list(LESSONS.keys()))

# =========================================================
# LESSON DISPLAY (SAFE + CLEAN)
# =========================================================
lesson = LESSONS[selected_lesson]

st.markdown(f"""
<div class="lesson-card">

<img src="{lesson['image']}" style="
width:100%;
border-radius:14px;
margin-bottom:1rem;
">

<h1>{selected_lesson}</h1>

<h3 style="color:#60a5fa;">{lesson['hook']}</h3>

<div style="white-space:pre-wrap; font-size:17px;">
{lesson['content']}
</div>

</div>
""", unsafe_allow_html=True)

# =========================================================
# QUIZ SYSTEM (FIXED LOGIC)
# =========================================================
q, options, answer = QUIZZES[selected_lesson]

st.subheader("🧠 Quiz")

user_ans = st.radio(q, options, key=selected_lesson)

if st.button("Submit Quiz"):
    if user_ans == answer:
        st.success("Correct +10 XP 🎉")
        st.session_state.xp += 10
    else:
        st.error(f"Wrong ❌ Answer: {answer}")

# =========================================================
# FILE UPLOAD SYSTEM
# =========================================================
st.markdown("---")
st.subheader("📂 Upload File")

file = st.file_uploader("Upload PDF / TXT / DOCX", type=["pdf", "txt", "docx"])

def extract_text(file):
    text = ""

    if file.name.endswith("txt"):
        text = file.read().decode("utf-8")

    elif file.name.endswith("pdf"):
        pdf = PdfReader(file)
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()

    elif file.name.endswith("docx"):
        doc = Document(file)
        for p in doc.paragraphs:
            text += p.text + "\n"

    return text

if file:
    st.session_state.uploaded_text = extract_text(file)
    st.success("File processed successfully ✔")

# =========================================================
# FILE Q&A
# =========================================================
if st.session_state.uploaded_text:

    st.subheader("💬 Ask your file")

    q2 = st.text_input("Ask something from file")

    if q2:

        context = st.session_state.uploaded_text[:12000]

        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "Answer ONLY from the provided file."
                },
                {
                    "role": "user",
                    "content": context + "\n\nQuestion: " + q2
                }
            ],
            max_tokens=500
        )

        st.markdown(res.choices[0].message.content)

# =========================================================
# QUIZ FROM FILE
# =========================================================
if st.session_state.uploaded_text:

    if st.button("Generate Quiz From File"):

        context = st.session_state.uploaded_text[:10000]

        quiz = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "Create 5 beginner quiz questions."
                },
                {
                    "role": "user",
                    "content": context
                }
            ],
            max_tokens=700
        )

        st.markdown(quiz.choices[0].message.content)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("🧠 Asteria AI Academy — built with Streamlit + Groq")
