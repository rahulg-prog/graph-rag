import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import List
from dotenv import load_dotenv
from config.settings_loader import load_config
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

class DataEmbedding:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.chunks = []
        load_dotenv()
    
    def chunk_data(
        self, 
        chunk_size: int = None,
        chunk_overlap: int = None,
        separators: List[str] = None,
        volume_paths: List[tuple] = None
    ) -> List[Document]:
        chunk_size = chunk_size or self.config["chunking"]["chunk_size"]
        chunk_overlap = chunk_overlap or self.config["chunking"]["chunk_overlap"]
        separators = separators or [". ", "."]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators
        )
        
        if volume_paths is None:
            volume_paths = [
                ("volume_1", os.path.join(self.base_dir, self.config["data_source"]["jsonified_data"]["volume_1"])),
                ("volume_2", os.path.join(self.base_dir, self.config["data_source"]["jsonified_data"]["volume_2"]))
            ]
        
        print(f"Chunk size: {chunk_size}")
        print(f"Chunk overlap: {chunk_overlap}")
        
        for volume_name, file_path in volume_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = [
                Document(page_content=entry['text'], metadata=entry['metadata'])
                for entry in data
            ]
            
            volume_chunks = text_splitter.split_documents(documents)
            self.chunks.extend(volume_chunks)
            
            print(f"Original entries for {volume_name}: {len(data)}")
            print(f"New chunks for {volume_name}: {len(volume_chunks)}")
        
        print(f"\nTotal chunks: {len(self.chunks)}")
        return self.chunks
    
    def create_embeddings(
        self,
        embedding_model: str = None,
        vector_store_path: str = None,
        api_key: str = None
    ):
        embedding_model = embedding_model or self.config["embedding"]["embedding_model"]
        
        new_dir_name = f"chunk_size_{self.config['chunking']['chunk_size']}_chunk_overlap_{self.config['chunking']['chunk_overlap']}"

        # Create the full path
        base_path = r"C:\vscode\graph-rag\Source\data\iterations"
        new_dir_path = os.path.join(base_path, new_dir_name)

        # Create vector_store subdirectory
        vector_store_dir = os.path.join(new_dir_path, "vector_store")
        os.makedirs(vector_store_dir, exist_ok=True)

        vector_store_path = vector_store_path or vector_store_dir
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # Check if vector store already exists
        faiss_index_path = os.path.join(vector_store_path, "index.faiss")
        if os.path.exists(faiss_index_path):
            print(f"\nVector store already exists at {vector_store_path}")
            print("Loading existing vector store...")
            embeddings = OpenAIEmbeddings(
                model=embedding_model,
                api_key=api_key
            )
            vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
            print(f"Loaded existing vector store with {vector_store.index.ntotal} vectors")
            return vector_store, new_dir_path
                
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=api_key
        )
        
        vector_store = FAISS.from_documents(self.chunks, embeddings)
        vector_store.save_local(vector_store_path)
        
        print(f"\nVector store created and saved to {vector_store_path}")
        print(f"Total vectors stored: {len(self.chunks)}")
        
        return vector_store, new_dir_path
    
if __name__ == "__main__":
    embedder = DataEmbedding()
    embedder.chunk_data()
    embedder.create_embeddings()