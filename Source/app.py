import os
import streamlit as st
from dotenv import load_dotenv
from langsmith import traceable
from config.settings_loader import load_config
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Roman Empire RAG Assistant",
    page_icon="🏛️",
    layout="centered"
)

# ============================================================================
# CONFIGURATION & INITIALIZATION
# ============================================================================

@st.cache_resource
def initialize_rag_agent():
    """Initialize the RAG agent with all necessary components."""
    load_dotenv()
    config = load_config("config/config.yaml")
    
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "evaluation_roman_rag"
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model=config["embedding"]["embedding_model"],
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Load vector store
    loaded_vector_store = FAISS.load_local(
        r"C:\vscode\graph-rag\Source\data\iterations\chunk_size_300_chunk_overlap_25\vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Initialize retriever
    retriever = loaded_vector_store.as_retriever(
        search_type=config["retriever"]["search_type"],
        search_kwargs={"k": config["retriever"]["k"]}
    )
    
    # Initialize LLM with streaming enabled
    llm = ChatOpenAI(
        model=config["llm"]["model_name"],
        api_key=os.getenv("OPENAI_API_KEY"),
        streaming=True
    )
    
    # Prompt template
    SYSTEM_PROMPT = """You are a historical assistant specializing in the Roman Empire.

Answer the question using ONLY information that is explicitly stated in the provided context.
Ignore any context that does not directly help answer the question.

Requirements:
- Base every claim on concrete details from the context (e.g., practices, locations, roles, deployments).
- Prefer specific examples over general statements.
- Do NOT introduce people, events, or interpretations that are not clearly supported by the context.
- Do NOT generalize beyond what the context shows.
- Do NOT mention context numbers, the author, or phrases like "the text says".
- Write a clear, factual paragraph that directly answers the question.

Context:
{context}

Question:
{question}

Answer:"""
    
    prompt_template = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["context", "question"]
    )
    
    # Build RAG chain
    def get_documents(query: str) -> str:
        """Retrieve and format documents for a given query."""
        docs = retriever.invoke(query)
        context_parts = [f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, 1)]
        return "\n\n".join(context_parts)
    
    rag_chain = (
        {
            "context": lambda x: get_documents(x["question"]),
            "question": RunnablePassthrough()
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, config

# Initialize the agent
rag_chain, config = initialize_rag_agent()

@traceable(
    run_type="chain",
    name="RAG_Chat_Streamlit",
    metadata={
        "llm_model": config["llm"]["model_name"],
        "embedding_model": config["embedding"]["embedding_model"],
        "chunk_size": config["chunking"]["chunk_size"],
        "chunk_overlap": config["chunking"]["chunk_overlap"],
        "search_type": config["retriever"]["search_type"],
        "top_k": config["retriever"]["k"],
        "vector_store": config["retriever"]["vector_store_path"]
    }
)
def chat_stream(query: str):
    """Generate answer for a single query using the RAG pipeline with streaming."""
    for chunk in rag_chain.stream({"question": query}):
        yield chunk

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================================
# UI LAYOUT
# ============================================================================

st.title("🏛️ Roman Empire RAG Assistant")
st.markdown("Ask questions about the Roman Empire and get answers based on historical documents.")

# Display configuration info in sidebar
with st.sidebar:
    st.header("Configuration")
    st.write(f"**Model:** {config['llm']['model_name']}")
    st.write(f"**Embedding:** {config['embedding']['embedding_model']}")
    st.write(f"**Chunk Size:** {config['chunking']['chunk_size']}")
    st.write(f"**Top K:** {config['retriever']['k']}")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================================================================
# CHAT INPUT & RESPONSE
# ============================================================================

if prompt := st.chat_input("Ask a question about the Roman Empire..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response with streaming
    with st.chat_message("assistant"):
        response = st.write_stream(chat_stream(prompt))
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
