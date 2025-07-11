import streamlit as st  
import os  
import bs4 
from dotenv import load_dotenv 


load_dotenv()

print("OPENAI_API_KEY exists:", os.getenv("OPENAI_API_KEY") is not None)



# LangChain and related imports for RAG pipeline
from langchain.chains import create_retrieval_chain  # For combining retrieval and answer generation
from langchain.chains.combine_documents import create_stuff_documents_chain  # For combining retrieved docs
from langchain_community.vectorstores import Chroma  # For storing and searching text chunks as vectors
from langchain_community.document_loaders import WebBaseLoader  # For loading web pages as documents
from langchain_core.prompts import ChatPromptTemplate  # For formatting prompts to the language model
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting text into chunks
from langchain_core.prompts import MessagesPlaceholder  # (Not used here, but often for chat history)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # For embeddings and language model


# Set environment variables for LangChain and OpenAI
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "RAG"


# Streamlit UI setup
st.title("LLM Powered Autonomous Agents") 
st.write("Ask anything about the loaded document—get instant, context-aware answers.")  

# Define a function to set up the RAG pipeline and cache it so it only runs once
@st.cache_resource  # Streamlit decorator to cache the result for faster reloads
def setup_rag_chain():
    embeddings = OpenAIEmbeddings()  
    model = ChatOpenAI(model="gpt-3.5-turbo")  

    # Load the web page and extract only relevant parts (post content, title, header)
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))),
    )

    doc = loader.load()  # Download and parse the web page

    # Split the document into manageable chunks for embedding and retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(doc)

    # Create a Chroma vector store from the text chunks
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # Create a retriever object to search for relevant chunks based on user queries
    retriever = vectorstore.as_retriever()

    # Defining the system prompt for the language model (how it should answer)
    system_prompt = (
        "Answer **ONLY** from the provided context. "
        "If the answer is not in the context, say 'I don't know.' "
        "Do not use any outside knowledge.\n\n"
        "Context:\n{context}"
    )




    # Create a chat prompt template with system and user messages
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Combine the model and prompt into a question-answering chain
    question_answering_chain = create_stuff_documents_chain(model, chat_prompt)

    # Combine the retriever and QA chain into a RAG chain
    rag_chain = create_retrieval_chain(retriever, question_answering_chain)

    return rag_chain  # Return the ready-to-use RAG chain

# Initialize the RAG chain (runs only once due to caching)
rag_chain = setup_rag_chain()

# Create a text input box for the user to type their question
user_question = st.text_input("Your question:")

# If the user has entered a question, process it
if user_question:
    with st.spinner("Thinking..."):  # Show a spinner while processing
        # Pass the user's question to the RAG chain and get the answer
        response = rag_chain.invoke({"input": user_question})
        # Show the answer in a green box, or the whole response if 'answer' key is missing
        st.success(response["answer"] if "answer" in response else response)
