# IIT_RAG_Blog_Application

A retrieval-augmented generation (RAG) system that answers questions about specific web content using LangChain and OpenAI

# 🌟 Features
Web Document Intelligence: Loads and understands web pages (blogs, articles, documentation)

Precise Q&A: Answers questions strictly from document context (no hallucinations)

Efficient Retrieval: Chroma vector database for fast semantic search

Streamlit UI: Clean, interactive interface for end users

Production-Ready: LangChain tracing and project management

# 🛠️ Tech Stack
Framework: LangChain

LLM: OpenAI GPT-3.5-turbo

Vector DB: Chroma

Web Interface: Streamlit

Environment: Python + dotenv


# 📚 Customization Guide
Change Source Document: Modify web_paths in setup_rag_chain()

Adjust Chunking: Edit chunk_size and chunk_overlap parameters

Upgrade LLM: Switch to GPT-4 by changing model="gpt-3.5-turbo"

# 📊 Advanced Features
LangSmith tracing for debugging

Customizable prompt engineering

Support for multiple document sources

