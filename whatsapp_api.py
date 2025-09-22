import os
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

from src.state import HealthGraphState
from src.nodes.loaders import node_load_outbreak_pdf
from src.nodes.indexer import node_build_faiss_index
from src.memory import DiskConversationMemory
from workflow import build_workflow

# app_flask = Flask(__name__)

# disk_mem = DiskConversationMemory("chat_memory.pkl")

# @app_flask.route("/", methods=["GET"])
# def home():
#     return "HealthGraph WhatsApp bot is running!"

# @app_flask.route("/whatsapp", methods=["POST"])
# def whatsapp_webhook():
#     incoming_msg = request.values.get("Body", "").strip()
#     sender = request.values.get("From", "")
#     print("Incoming:", incoming_msg, sender)

#     resp = MessagingResponse()
#     try:
#         graph_app = build_workflow()
#         state = HealthGraphState(
#             twilio_payload={"Body": incoming_msg, "From": sender},
#             disk_memory=disk_mem
#         )

#         state = node_load_outbreak_pdf(state)
#         state = node_build_faiss_index(state)

#         final_state = graph_app.invoke(state)

#         response_text = final_state["response"]
#         print(response_text)
#         if isinstance(response_text, dict):  
#             response_text = response_text["original"]

#         resp.message(response_text)

#     except Exception as e:
#         print("Error:", e)
#         resp.message("Oops! Something went wrong.")

#     return str(resp)

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5050))
#     app_flask.run(host="0.0.0.0", port=port, debug=True)


from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

app_flask = Flask(__name__)

@app_flask.route("/", methods=["GET"])
def home():
    return "Bot is running!"

@app_flask.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    incoming_msg = request.values.get("Body", "").strip()
    sender = request.values.get("From", "")
    print("Incoming:", incoming_msg, sender)

    resp = MessagingResponse()
    try:
        graph_app = build_workflow()
        disk_mem = DiskConversationMemory("chat_memory.pkl")
        state = HealthGraphState(
            twilio_payload={"Body": incoming_msg, "From": sender},
            disk_memory=disk_mem
        )

        state = node_load_outbreak_pdf(state)  

        state = node_build_faiss_index(state)  
        final_state = graph_app.invoke(state)
        response_text = final_state["response"]
        print(response_text)
        if hasattr(response_text, "original"):
            response_text = response_text.original
        resp.message(response_text)
    except Exception as e:
        print("Error:", e)
        resp.message("Oops! Something went wrong.")

    return str(resp)

if __name__ == "__main__":
    app_flask.run(host="0.0.0.0", port=5050)  
