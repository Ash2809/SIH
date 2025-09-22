import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langgraph.graph import StateGraph, START, END
from src.state import HealthGraphState
from src.nodes.ingress import node_twilio_ingress
from src.nodes.loaders import node_load_vaccination_json, node_load_outbreak_pdf
from src.nodes.indexer import node_build_faiss_index
from src.nodes.router import node_router, node_route_dispatcher
from src.nodes.handlers import node_emergency_outbreak, node_symptom, node_vaccination_schedule, node_general_query
from src.nodes.translation import node_translation

def build_workflow():
    workflow = StateGraph(HealthGraphState)
    workflow.add_node("twilio_ingress", node_twilio_ingress)
    workflow.add_node("load_vaccination_json", node_load_vaccination_json)
    workflow.add_node("load_outbreak_pdf", node_load_outbreak_pdf)
    workflow.add_node("build_faiss_index", node_build_faiss_index)
    workflow.add_node("router", node_router)
    workflow.add_node("route_dispatcher", node_route_dispatcher)
    workflow.add_node("emergency_outbreak", node_emergency_outbreak)
    workflow.add_node("symptom", node_symptom)
    workflow.add_node("vaccination_schedule", node_vaccination_schedule)
    workflow.add_node("general_query", node_general_query)
    workflow.add_node("translation", node_translation)

    workflow.add_edge(START, "twilio_ingress")
    workflow.add_edge("twilio_ingress", "router")
    workflow.add_edge("router", "route_dispatcher")
    workflow.add_edge("load_vaccination_json", "build_faiss_index")
    workflow.add_edge("load_outbreak_pdf", "build_faiss_index")
    workflow.add_edge("build_faiss_index", "route_dispatcher")
    workflow.add_edge("route_dispatcher", "translation")
    workflow.add_edge("translation", END)

    print("Compilation successfull.")
    return workflow.compile()

if __name__ == "__main__":
    build_workflow()
