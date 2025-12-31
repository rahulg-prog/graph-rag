import os
import json
from typing import List, Dict
from dotenv import load_dotenv
from langsmith import traceable
from config.settings_loader import load_config
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ============================================================================
# CONFIGURATION & INITIALIZATION
# ============================================================================

load_dotenv()
config = load_config("config/config.yaml")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "graph-rag-evaluation"

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model=config["embedding"]["embedding_model"],
    api_key=os.getenv("OPENAI_API_KEY")
)

# Load vector store
loaded_vector_store = FAISS.load_local(
    config["retriever"]["vector_store_path"],
    embeddings,
    allow_dangerous_deserialization=True
)

# Initialize retriever
retriever = loaded_vector_store.as_retriever(
    search_type=config["retriever"]["search_type"],
    search_kwargs={"k": config["retriever"]["k"]}
)

# Initialize LLM
llm = ChatOpenAI(
    model=config["llm"]["model_name"],
    api_key=os.getenv("OPENAI_API_KEY")
)

# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

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

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_documents(query: str) -> str:
    """Retrieve and format documents for a given query."""
    docs = retriever.invoke(query)
    context_parts = [f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, 1)]
    return "\n\n".join(context_parts)

def get_retrieval_context(query: str) -> List[str]:
    """Retrieve documents as a list of strings."""
    docs = retriever.invoke(query)
    return [doc.page_content for doc in docs]

# ============================================================================
# RAG CHAIN
# ============================================================================

rag_chain = (
    {
        "context": lambda x: get_documents(x["question"]),
        "question": RunnablePassthrough()
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

@traceable(
    run_type="chain",
    name="Evaluate_Single_Question"
)
def evaluate_single_question(question: str, expected_output: str, question_idx: int) -> Dict:
    """
    Evaluate a single question through the RAG pipeline.
    
    Args:
        question: The input question
        expected_output: The ground truth answer
        question_idx: Index of the question for tracking
        
    Returns:
        Dictionary with evaluation data for this question
    """
    print(f"Processing question {question_idx}: {question[:80]}...")
    
    # Get retrieval context
    retrieval_context = get_retrieval_context(question)
    
    # Get actual output from RAG chain
    actual_output = rag_chain.invoke({"question": question})
    
    # Create evaluation data point
    eval_point = {
        "input": question,
        "actual_output": actual_output,
        "expected_output": expected_output,
        "retrieval_context": retrieval_context
    }
    
    return eval_point


def evaluate(
    jsonl_path: str = None, 
    output_json_path: str = None
) -> List[Dict]:
    """
    Evaluate the RAG pipeline using ground truth data and save results as JSON.
    
    Args:
        jsonl_path: Path to the JSONL file containing ground truth data
        output_json_path: Path where the evaluation results JSON will be saved
        
    Returns:
        List of dictionaries with evaluation results
    """
    # Use config defaults if not provided
    if jsonl_path is None:
        jsonl_path = config["evaluation"]["ground_truth_path"]
    if output_json_path is None:
        output_json_path = config["evaluation"]["output_path"]
    
    evaluation_data = []
    
    print(f"Starting evaluation from {jsonl_path}...")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            item = json.loads(line.strip())
            question = item["inputs"]["question"]
            expected_output = item["outputs"]["answer"]
            
            # Each question runs as a separate trace
            eval_point = evaluate_single_question(question, expected_output, idx)
            evaluation_data.append(eval_point)
    
    # Save to JSON file
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Evaluation complete!")
    print(f"✓ Processed {len(evaluation_data)} questions")
    print(f"✓ Results saved to {output_json_path}")
    
    return evaluation_data

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run evaluation
    eval_results = evaluate()