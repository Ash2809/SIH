from langchain_community.document_loaders import JSONLoader
from langchain.document_loaders import PyPDFLoader
from src.state import HealthGraphState
from langchain_community.document_loaders import PyPDFLoader


def node_load_vaccination_json(state: HealthGraphState) -> HealthGraphState:
    loader = JSONLoader(file_path=state.vaccination_json_path)
    state.vaccination_docs = loader.load()
    return state

def node_load_outbreak_pdf(state: HealthGraphState) -> HealthGraphState:
    loader = PyPDFLoader(state.outbreak_pdf_path)
    state.outbreak_docs = loader.load_and_split()
    return state
