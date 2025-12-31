from data_embedding import DataEmbedding
from agent import RAGAgent

embedder = DataEmbedding()
embedder.chunk_data()
vector_store, path = embedder.create_embeddings()

agent = RAGAgent(vector_store=vector_store, path=path)

# Run evaluation first, then DeepEval metrics
agent.evaluate()
agent.run_deepeval_metrics()