from data_converter import convert_data
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_astradb import AstraDBVectorStore
from dotenv import load_dotenv
import os

def ingest_data(status=None):
    load_dotenv()

    ASTRA_API_KEY = os.getenv("ASTRA_API_KEY")
    DB_ENDPOINT = os.getenv("DB_ENDPOINT")

    DB_ID = "Medical"

    hf_embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = AstraDBVectorStore(
        embedding=hf_embedding,
        api_endpoint=DB_ENDPOINT,
        namespace="default_keyspace",
        token=ASTRA_API_KEY,
        collection_name="medical_v2",
    )

    print("Database initialized")

    if status is None:
        pdf_path = r"/Users/aashutoshkumar/Documents/Projects/healthgraph-assistant/data/8205Oxford Handbook of Clinical Medicine 10th 2017 Edition_SamanSarKo - Copy (1).pdf"
        text_chunks = convert_data(pdf_path)
        print(f"Upserting {len(text_chunks)} documents into DB...")
        inserted_ids = vector_store.add_documents(text_chunks)
        return vector_store, inserted_ids
    else:
        return vector_store, []

if __name__ == "__main__":
    vector_store, inserted_ids = ingest_data(None)
    if inserted_ids:
        print(f"Inserted {len(inserted_ids)} documents")
    print(" DB has been initialized")
