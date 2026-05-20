import os
import chromadb
from dotenv import load_dotenv

load_dotenv()

AI_KNOWLEDGE = [
    {
        "id": "llm_1",
        "text": "A Large Language Model (LLM) is an AI system trained on massive amounts of text data. It learns patterns in language and can generate human-like text, answer questions, summarize content, write code, and much more. Examples include GPT-4, Claude, and Gemini.",
        "topic": "LLMs"
    },
    {
        "id": "llm_2",
        "text": "LLMs work using a neural network architecture called the Transformer. It uses a mechanism called 'attention' which lets the model focus on the most relevant words when generating a response. This is why LLMs understand context so well.",
        "topic": "LLMs"
    },
    {
        "id": "llm_3",
        "text": "Prompt engineering is the skill of writing instructions to an LLM to get the best possible output. About 90% of AI failures come from poor prompting. Key techniques include: zero-shot prompting (just ask), few-shot prompting (give examples), and chain-of-thought prompting (ask the model to think step by step).",
        "topic": "Prompt Engineering"
    },
    {
        "id": "llm_4",
        "text": "Zero-shot prompting means asking the AI a question with no examples. Few-shot prompting means giving 2-3 examples before your question so the AI understands the pattern you want. Chain-of-thought prompting asks the AI to reason step by step before giving a final answer, which greatly improves accuracy on complex tasks.",
        "topic": "Prompt Engineering"
    },
    {
        "id": "rag_1",
        "text": "RAG stands for Retrieval-Augmented Generation. It is a technique where, before the LLM answers a question, the system first searches a knowledge base for relevant information and provides that context to the LLM. This grounds the AI's answers in real, specific data rather than just its general training.",
        "topic": "RAG"
    },
    {
        "id": "rag_2",
        "text": "RAG vs Retraining: Instead of expensively retraining an LLM on your private data, RAG lets you plug your own documents into any existing LLM. The model reads your documents at query time and answers based on them. This saves enormous cost and time for companies.",
        "topic": "RAG"
    },
    {
        "id": "rag_3",
        "text": "A RAG pipeline has 4 steps: 1) Ingest - load your documents and split them into chunks. 2) Embed - convert each chunk into a vector (a list of numbers representing meaning). 3) Store - save vectors in a vector database like ChromaDB. 4) Retrieve - when a user asks a question, find the most similar chunks and send them to the LLM as context.",
        "topic": "RAG"
    },
    {
        "id": "vector_1",
        "text": "A vector database stores text as embeddings — numerical representations of meaning. When you search a vector database, it finds text that is semantically similar (same meaning) rather than just keyword matching. This is called semantic search. ChromaDB and Pinecone are popular vector databases.",
        "topic": "Vector Databases"
    },
    {
        "id": "vector_2",
        "text": "Embeddings are numerical representations of text. Similar meanings produce similar numbers. For example, 'dog' and 'puppy' would have very similar embeddings, while 'dog' and 'airplane' would be very different. This allows semantic search to find relevant content even when the exact words don't match.",
        "topic": "Vector Databases"
    },
    {
        "id": "agent_1",
        "text": "An AI agent is an autonomous AI system that can plan and execute multi-step tasks on its own. Unlike a basic chatbot that just answers questions, an agent can decide what tools to use, search the web, write and run code, and take actions to complete a goal. Agents are more dynamic than static RAG implementations.",
        "topic": "AI Agents"
    },
    {
        "id": "agent_2",
        "text": "Multi-agent systems involve multiple AI agents working together, each with a specific role. For example, one agent might act as a teacher, another as a student, and another as an evaluator. They communicate with each other to automate complex workflows that would be hard for a single agent.",
        "topic": "AI Agents"
    },
    {
        "id": "context_1",
        "text": "A context window is the maximum amount of text an LLM can process at once. Larger context windows allow more information but can cause confusion if the data is unstructured or irrelevant. It is important to send only clean, relevant information to the model rather than dumping everything into the context.",
        "topic": "LLM Concepts"
    },
    {
        "id": "data_1",
        "text": "Data quality is the most important factor in AI success. Clean, structured, and balanced data produces much better AI results than large amounts of messy data. This is called the data-centric approach — focusing on improving your data rather than just changing the model.",
        "topic": "Data"
    },
    {
        "id": "transformer_1",
        "text": "The Transformer architecture, introduced in the 2017 paper 'Attention is All You Need', is the foundation of all modern LLMs. It processes entire sequences of text in parallel using self-attention mechanisms, which allows it to understand relationships between all words in a sentence simultaneously.",
        "topic": "Architecture"
    },
    {
        "id": "langchain_1",
        "text": "LangChain and LlamaIndex are Python frameworks that make it easier to build RAG pipelines and AI agents. They provide ready-made components for loading documents, creating embeddings, connecting to vector databases, and chaining LLM calls together. They save a lot of boilerplate code.",
        "topic": "Frameworks"
    },
    {
        "id": "deploy_1",
        "text": "Deploying an AI app means making it publicly accessible on the internet. Streamlit Cloud offers free hosting for Python web apps. You push your code to GitHub, connect it to Streamlit Cloud, and get a public URL anyone can visit. This is essential for showcasing your project.",
        "topic": "Deployment"
    },
    {
        "id": "mlops_1",
        "text": "MLOps stands for Machine Learning Operations. It covers the practices of deploying, monitoring, and maintaining AI systems in production. Key concepts include CI/CD (Continuous Integration/Continuous Deployment), model monitoring, and version control for both code and data.",
        "topic": "MLOps"
    },
    {
        "id": "multimodal_1",
        "text": "Multimodal AI can process multiple types of data — text, images, audio, and video. Models like Gemini and GPT-4V can look at an image and answer questions about it. This opens up applications like visual question answering, document understanding, and image-based tutoring.",
        "topic": "Multimodal AI"
    },
]

def build_knowledge_base():
    print("Building AI knowledge base...")
    client = chromadb.PersistentClient(path="./chroma_db")
    
    try:
        client.delete_collection("ai_knowledge")
        print("Cleared existing knowledge base.")
    except:
        pass
    
    collection = client.create_collection(
        name="ai_knowledge",
        metadata={"hnsw:space": "cosine"}
    )
    
    texts = [item["text"] for item in AI_KNOWLEDGE]
    ids = [item["id"] for item in AI_KNOWLEDGE]
    metadatas = [{"topic": item["topic"]} for item in AI_KNOWLEDGE]
    
    collection.add(
        documents=texts,
        ids=ids,
        metadatas=metadatas
    )
    
    print(f"Successfully loaded {len(AI_KNOWLEDGE)} knowledge chunks!")
    print("Topics covered:")
    topics = list(set(item["topic"] for item in AI_KNOWLEDGE))
    for topic in topics:
        print(f"  - {topic}")
    print("\nKnowledge base is ready. Now run: streamlit run app.py")

if __name__ == "__main__":
    build_knowledge_base()
