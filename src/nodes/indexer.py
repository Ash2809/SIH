from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from src.state import HealthGraphState

def node_build_faiss_index(state: HealthGraphState) -> HealthGraphState:
    hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_dir = Path(state.index_dir).parent
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss_file = Path(state.index_dir)
    if faiss_file.exists():
        state.local_vectorstore = FAISS.load_local(
            str(index_dir), hf_embedding, allow_dangerous_deserialization=True
        )
        print(f"Loaded FAISS index from {index_dir}")
        return state

    docs = state.outbreak_docs or []
    if not docs:
        print("No outbreak documents found to index.")
        return state

    vectorstore = FAISS.from_documents(docs, embedding=hf_embedding)
    vectorstore.save_local(str(index_dir))
    state.local_vectorstore = vectorstore
    print(f"Built new FAISS index with {len(docs)} docs and saved to {index_dir}")
    return state
