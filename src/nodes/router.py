from src.state import HealthGraphState
from src.utils.llm import llm
from src.nodes.handlers import (
    node_emergency_outbreak,
    node_symptom,
    node_vaccination_schedule,
    node_general_query,
)


def node_router(state: HealthGraphState) ->  HealthGraphState:
    message = state.user_message or ""
    print("----Router node----")
    if not message.strip():
        state.route_decision = {"route": "general_query", "reason": "empty_message"}
        return state

    prompt = f"""
    You are a classifier. 
    Categorize the user's query into one of these routes:
    if the query is asking about outbreaks in his/her area route to emergency_outbreak.
    if the query is regarding any medical condition then route to symptom for efficient RAG.
    if the query is regrading the vaccination schedule then route to vaccination_schedule.
    if you cant classify the query in any of the given classes then route to general_query.

    - emergency_outbreak
    - symptom
    - vaccination_schedule
    - general_query

    Query: "{message}"

    Reply ONLY with a valid route name.
    """

    result = llm.invoke(prompt).content.strip().lower()
    print(result)

    valid_routes = {"emergency_outbreak", "symptom", "vaccination_schedule", "general_query"}
    route = result if result in valid_routes else "general_query"

    state.route_decision = {"route": route, "reason": "llm_classified"}
    return state


def node_route_dispatcher(state: HealthGraphState) -> HealthGraphState:
    route = state.route_decision["route"] if state.route_decision else "general_query"
    if route == "emergency_outbreak":
        return node_emergency_outbreak(state)
    if route == "symptom":
        return node_symptom(state)
    if route == "vaccination_schedule":
        return node_vaccination_schedule(state)
    return node_general_query(state)