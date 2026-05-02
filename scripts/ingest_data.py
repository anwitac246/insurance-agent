import os
import glob
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import time

# Get project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load environment variables from .env in project root
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment.")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Use HuggingFace embeddings (Free, runs locally, fast)
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
print(f"Loading embedding model: {embedding_model}...")
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
dimension = 384

def recreate_index(index_name: str):
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name in existing_indexes:
        print(f"Deleting existing index: {index_name}")
        pc.delete_index(index_name)
    
    print(f"Creating new index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    # Wait for index to be ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
    
    return pc.Index(index_name)

def process_and_upsert(data_dir: str, index_name: str):
    print(f"\nProcessing {data_dir} into {index_name}...")
    
    # Load all markdown files
    md_files = glob.glob(os.path.join(data_dir, "*.md"))
    
    # Define headers to split on to maintain context
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # Secondary splitter for large chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    all_chunks = []
    
    for file_path in md_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Split by markdown headers
        md_splits = markdown_splitter.split_text(text)
        
        # Add metadata source
        for split in md_splits:
            split.metadata["source"] = os.path.basename(file_path)
            
        # Split further if chunks are too big
        splits = text_splitter.split_documents(md_splits)
        all_chunks.extend(splits)
        print(f"Processed {os.path.basename(file_path)}: {len(splits)} chunks")

    print(f"Total chunks to insert: {len(all_chunks)}")
    
    # Recreate index to start fresh
    recreate_index(index_name)
    
    # Upsert using PineconeVectorStore
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    vectorstore.add_documents(all_chunks)
    print(f"Successfully upserted data to {index_name}.")

if __name__ == "__main__":
    policy_dir = os.path.join(PROJECT_ROOT, "data", "policy_data")
    fraud_dir = os.path.join(PROJECT_ROOT, "data", "fraud_data")
    
    process_and_upsert(policy_dir, "insurance-policies")
    process_and_upsert(fraud_dir, "fraud-cases")
    
    print("\nData Ingestion Complete!")
