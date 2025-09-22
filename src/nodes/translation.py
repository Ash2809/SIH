import json
from src.state import HealthGraphState
from src.utils.llm import llm

def node_translation(state: HealthGraphState) -> HealthGraphState:
    if not state.response:
        state.response = "No response to translate."
        return state

    prompt = f"""
    You are a translation assistant. 
    Take the following text and translate it into **Odia** and **Telugu**.

    TEXT: {state.response}
    """

    result = llm.invoke(prompt)
    try:
        state.response = json.loads(result.content) 
    except Exception:
        state.response = {"original": state.response, "translation_raw": result.content}

    return state