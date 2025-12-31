import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config.settings_loader import load_config

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
    return retriever.invoke(query, k=5)




