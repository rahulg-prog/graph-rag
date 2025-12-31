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
        vector_store_path = vector_store_path or self.config["retriever"]["vector_store_path"]
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=api_key
        )
        
        vector_store = FAISS.from_documents(self.chunks, embeddings)
        vector_store.save_local(vector_store_path)
        
        print(f"\nVector store created and saved to {vector_store_path}")
        print(f"Total vectors stored: {len(self.chunks)}")
        
        return vector_store
    
if __name__ == "__main__":
    embedder = DataEmbedding()
    embedder.chunk_data()
    embedder.create_embeddings()