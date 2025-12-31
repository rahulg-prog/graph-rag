import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config.settings_loader import load_config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

config = load_config("config/config.yaml")

load_dotenv()

# retriever related code
embeddings = OpenAIEmbeddings(
    model=config["embedding"]["embedding_model"],
    api_key=os.getenv("OPENAI_API_KEY")
)
loaded_vector_store = FAISS.load_local(
    config["retriever"]["vector_store_path"],
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = loaded_vector_store.as_retriever(
    search_type=config["retriever"]["search_type"],
    search_kwargs={"k": config["retriever"]["k"]}
)

def get_documents(query: str):
    docs = retriever.invoke(query)
    context_parts = [f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, 1)]
    return "\n\n".join(context_parts)

def chat(query: str):
    llm = ChatOpenAI(
        model=config["llm"]["model"],
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Get relevant context from retrieved documents
    context = get_documents(query)
    
    # Create prompt template
    prompt = PromptTemplate(
        template="""You are a helpful assistant. Answer the question based solely on the provided context. 
        If the answer cannot be found in the context, say "I don't have enough information to answer this question.
        use only relevant information from the context to answer the question and ignore the irrelevant parts."

        Context:
        {context}

        Question: {question}

        Answer:""",
                input_variables=["context", "question"]
        )
    
    formatted_prompt = prompt.format(context=context, question=query)

    response = llm.invoke(formatted_prompt)
    return response.content