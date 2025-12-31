import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from config.settings_loader import load_config

load_dotenv()
config = load_config("config/config.yaml")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "graph-rag"

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

def get_documents(query: str) -> str:
    docs = retriever.invoke(query)
    context_parts = [f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, 1)]
    return "\n\n".join(context_parts)

@traceable(
    run_type="chain",
    name="RAG Pipeline",
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
def chat(query: str) -> str:
    # Initialize LLM
    llm = ChatOpenAI(
        model=config["llm"]["model_name"],
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    prompt = PromptTemplate(
        template="""You are a historical assistant specializing in the Roman Empire.

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

    Answer:""",
        input_variables=["context", "question"]
    )

    
    rag_chain = (
        {
            "context": lambda x: get_documents(x["question"]),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = rag_chain.invoke({"question": query})
    return response

import json
from typing import List, Dict

def evaluate(jsonl_path: str, output_json_path: str = "evaluation/evaluation_results.json") -> List[Dict]:
    """
    Reads a JSONL file, processes it through the RAG pipeline, and saves evaluation data to JSON.
    
    Args:
        jsonl_path: Path to the JSONL file containing ground truth data
        output_json_path: Path where the evaluation results JSON will be saved
        
    Returns:
        List of dictionaries with the format:
        {
            "input": str,
            "actual_output": str,
            "expected_output": str,
            "retrieval_context": List[str]
        }
    """
    evaluation_data = []
    llm = ChatOpenAI(
        model=config["llm"]["model_name"],
        api_key=os.getenv("OPENAI_API_KEY")
    )
    prompt = PromptTemplate(
        template="""You are a historical assistant specializing in the Roman Empire.

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

    Answer:""",
        input_variables=["context", "question"]
    )
    rag_chain = (
        {
            "context": lambda x: get_documents(x["question"]),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            question = item["inputs"]["question"]
            docs = retriever.invoke(question)
            retrieval_context = [doc.page_content for doc in docs]
            actual_output = rag_chain.invoke({"question": question})
            expected_output = item["outputs"]["answer"]
            eval_point = {
                "input": question,
                "actual_output": actual_output,
                "expected_output": expected_output,
                "retrieval_context": retrieval_context
            }
            evaluation_data.append(eval_point)
    
    # Save to JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
    
    print(f"Evaluation results saved to {output_json_path}")
    return evaluation_data

if __name__ == "__main__":
    eval_results = evaluate(
        "evaluation/ground_truth/ground_truth.jsonl",
        "evaluation/evaluation_results.json"
    )
    print(f"Processed {len(eval_results)} questions")