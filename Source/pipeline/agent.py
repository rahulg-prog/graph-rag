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
from evaluation.deepeval_evaluation import RetrieverEvaluator


class RAGAgent:
    def __init__(self, config_path: str = "config/config.yaml"):
        load_dotenv()
        self.config = load_config(config_path)
        
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "graph-rag-evaluation"
        
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        self.llm = None
        self.rag_chain = None
        self.prompt_template = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        self.embeddings = OpenAIEmbeddings(
            model=self.config["embedding"]["embedding_model"],
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.vector_store = FAISS.load_local(
            self.config["retriever"]["vector_store_path"],
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        self.retriever = self.vector_store.as_retriever(
            search_type=self.config["retriever"]["search_type"],
            search_kwargs={"k": self.config["retriever"]["k"]}
        )
        
        self.llm = ChatOpenAI(
            model=self.config["llm"]["model_name"],
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        system_prompt = """You are a historical assistant specializing in the Roman Empire.

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
        
        self.prompt_template = PromptTemplate(
            template=system_prompt,
            input_variables=["context", "question"]
        )
        
        self.rag_chain = (
            {
                "context": lambda x: self._get_documents(x["question"]),
                "question": RunnablePassthrough()
            }
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
    
    def _get_documents(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        context_parts = [f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, 1)]
        return "\n\n".join(context_parts)
    
    def _get_retrieval_context(self, query: str) -> List[str]:
        docs = self.retriever.invoke(query)
        return [doc.page_content for doc in docs]
    
    @traceable(run_type="chain", name="Evaluate_Single_Question")
    def evaluate_single_question(self, question: str, expected_output: str, question_idx: int) -> Dict:
        print(f"Processing question {question_idx}: {question[:80]}...")
        
        retrieval_context = self._get_retrieval_context(question)
        actual_output = self.rag_chain.invoke({"question": question})
        
        eval_point = {
            "input": question,
            "actual_output": actual_output,
            "expected_output": expected_output,
            "retrieval_context": retrieval_context
        }
        
        return eval_point
    
    def evaluate(self, jsonl_path: str = None, output_json_path: str = None) -> List[Dict]:
        if jsonl_path is None:
            jsonl_path = self.config["evaluation"]["ground_truth_path"]
        if output_json_path is None:
            output_json_path = self.config["evaluation"]["output_path"]
        
        evaluation_data = []
        
        print(f"Starting evaluation from {jsonl_path}...")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, 1):
                item = json.loads(line.strip())
                question = item["inputs"]["question"]
                expected_output = item["outputs"]["answer"]
                
                eval_point = self.evaluate_single_question(question, expected_output, idx)
                evaluation_data.append(eval_point)
        
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Evaluation complete!")
        print(f"✓ Processed {len(evaluation_data)} questions")
        print(f"✓ Results saved to {output_json_path}")
        
        return evaluation_data
    
    def run_deepeval_metrics(self, json_path: str = None, csv_path: str = None):
        if json_path is None:
            json_path = self.config["evaluation"]["output_path"]
        if csv_path is None:
            csv_path = self.config["evaluation"]["evaluation_csv"]
        
        with open(json_path, 'r', encoding='utf-8') as f:
            evaluation_data = json.load(f)
        
        print(f"\n🔍 Running DeepEval metrics on {len(evaluation_data)} questions...")
        evaluator = RetrieverEvaluator(csv_path)
        
        for idx, eval_point in enumerate(evaluation_data, 1):
            print(f"Evaluating with DeepEval {idx}/{len(evaluation_data)}...")
            evaluator.evaluate(eval_point)
        
        print(f"✓ DeepEval results saved to {csv_path}")
        return csv_path
    
    def chat(self, question: str) -> str:
        return self.rag_chain.invoke({"question": question})


if __name__ == "__main__":
    agent = RAGAgent()
    
    run_evaluation = False
    
    if run_evaluation:
        agent.evaluate()
    
    agent.run_deepeval_metrics()