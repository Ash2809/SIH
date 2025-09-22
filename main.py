import os
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

from src.state import HealthGraphState
from src.nodes.loaders import node_load_outbreak_pdf
from src.nodes.indexer import node_build_faiss_index
from src.memory import DiskConversationMemory
from workflow import build_workflow   # <-- import the function

# build the compiled graph once
graph_app = build_workflow()          # <-- call it to get the compiled workflow

disk_mem = DiskConversationMemory()

state = HealthGraphState(
    disk_memory=disk_mem
)
state = node_load_outbreak_pdf(state)  
state = node_build_faiss_index(state)  

state.twilio_payload = {"Body": "the pain is very bad and i have headache since 1 hour"}
final_state = graph_app.invoke(state)   # <-- works now

print("Final Response:\n", final_state["response"]["original"])
print("Final Response:\n", final_state["response"]["translation_raw"])
