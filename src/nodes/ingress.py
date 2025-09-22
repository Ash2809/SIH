from src.state import HealthGraphState

def node_twilio_ingress(state: HealthGraphState) -> HealthGraphState:
    payload = state.twilio_payload
    if not payload:
        return state

    text = payload.get("Body") or payload.get("Message") or payload.get("text")
    sender = payload.get("From") or payload.get("from")

    state.user_message = text
    state.user_meta = {"sender": sender, "raw_payload": payload}
    return state
