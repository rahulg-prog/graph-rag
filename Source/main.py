import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from config.settings_loader import load_config

# Load environment variables and configuration
load_dotenv()
config = load_config("config/config.yaml")

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model=config["embedding"]["embedding_model"],
    api_key=os.getenv("OPENAI_API_KEY")
)

# Load FAISS vector store
loaded_vector_store = FAISS.load_local(
    f"C:/vscode/graph-rag/Source/data/vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)

# Create retriever
retriever = loaded_vector_store.as_retriever(
    search_type=config["retriever"]["search_type"],
    search_kwargs={"k": config["retriever"]["k"]}
)

def get_documents(query: str) -> str:
    docs = retriever.invoke(query)
    context_parts = [f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, 1)]
    return "\n\n".join(context_parts)

def chat(query: str) -> str:
    # Initialize LLM
    llm = ChatOpenAI(
        model=config["llm"]["model_name"],
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Get relevant context from retrieved documents
    context = get_documents(query)
    print("Context:", context)
    # Create prompt template
    prompt = PromptTemplate(
        template="""You are a helpful assistant. Answer the question based solely on the provided context.
If the answer cannot be found in the context, say "I don't have enough information to answer this question."
Use only relevant information from the context to answer the question and ignore the irrelevant parts.

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )
    
    # Format prompt with context and question
    formatted_prompt = prompt.format(context=context, question=query)
    
    # Get response from LLM
    response = llm.invoke(formatted_prompt)
    return response.content

if __name__ == "__main__":
    user_query = "How does the author argue that Roman persecution of early Christians was driven more by concerns for public order, political stability, and social conformity than by consistent religious hatred, and what evidence does he use to support this interpretation?"
    answer = chat(user_query)
    print("Answer:", answer)