import os
import pickle
import json
from pathlib import Path
from typing import Any, Dict, Optional
from collections import Counter
import re


from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import JSONLoader
from langchain.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ASTRA_API_KEY = os.getenv("ASTRA_API_KEY")
DB_ENDPOINT = os.getenv("DB_ENDPOINT")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.2,
)


from langchain.memory import ConversationBufferMemory
class DiskConversationMemory:
    def __init__(self, filename="chat_memory.pkl"):
        self.filename = Path(filename)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self._load()

    def _load(self):
        if self.filename.exists():
            try:
                with open(self.filename, "rb") as f:
                    self.memory = pickle.load(f)
                print(f"Loaded memory from {self.filename}")
            except Exception as e:
                print("Failed to load memory, starting fresh:", e)

    def persist(self):
        try:
            with open(self.filename, "wb") as f:
                pickle.dump(self.memory, f)
                print(f"Persisted memory to {self.filename}")
        except Exception as e:
            print("Failed to persist memory:", e)

class HealthGraphState(BaseModel):
    twilio_payload: Optional[Dict[str, Any]] = None
    user_message: Optional[str] = None
    user_meta: Optional[Dict[str, Any]] = None
    vaccination_docs: Optional[Any] = None
    outbreak_docs: Optional[Any] = None
    local_vectorstore: Optional[Any] = None
    disk_memory: Optional[Any] = None
    route_decision: Optional[Dict[str, str]] = None
    response: Optional[str] = None
    vaccination_json_path: Optional[str] = (
        r"/Users/aashutoshkumar/Documents/Projects/healthgraph-assistant/data/vaccination_schedule.json"
    )
    outbreak_pdf_path: Optional[str] = (
        r"/Users/aashutoshkumar/Documents/Projects/healthgraph-assistant/latest_weekly_outbreak/31st_weekly_outbreak.pdf"
    )
    index_dir: Optional[str] = (
        r"/Users/aashutoshkumar/Documents/Projects/healthgraph-assistant/exp/faiss_index/index.faiss"
    )


def node_twilio_ingress(state: HealthGraphState) -> HealthGraphState:
    payload = state.twilio_payload
    if not payload:
        return state

    text = payload.get("Body") or payload.get("Message") or payload.get("text")
    sender = payload.get("From") or payload.get("from")

    state.user_message = text
    state.user_meta = {"sender": sender, "raw_payload": payload}
    return state


def node_load_vaccination_json(state: HealthGraphState) -> HealthGraphState:
    loader = JSONLoader(file_path=state.vaccination_json_path)
    docs = loader.load()
    state.vaccination_docs = docs
    return state


def node_load_outbreak_pdf(state: HealthGraphState) -> HealthGraphState:
    loader = PyPDFLoader(state.outbreak_pdf_path)
    docs = loader.load_and_split()
    state.outbreak_docs = docs
    return state


from langchain.vectorstores import FAISS

def node_build_faiss_index(state: HealthGraphState) -> HealthGraphState:
    hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    index_dir = Path(state.index_dir).parent
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss_file = Path(state.index_dir)

    if faiss_file.exists():
        state.local_vectorstore = FAISS.load_local(str(index_dir), hf_embedding, allow_dangerous_deserialization=True)
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


# def node_router(state: HealthGraphState) -> HealthGraphState:
#     message = state.user_message or ""
#     if not message.strip():
#         state.route_decision = {"route": "general_query", "reason": "empty_message"}
#         return state

#     prompt = f"""
#     You are a classifier. 
#     Categorize the user's query into one of these routes:
#     if the query is asking about outbreaks in his/her area route to emergency_outbreak.
#     if the query is regarding any medical condition then route to symptom for efficient RAG.
#     if the query is regrading the vaccination schedule then route to vaccination_schedule.
#     if you cant classify the query in any of the given classes then route to general_query.

#     - emergency_outbreak
#     - symptom
#     - vaccination_schedule
#     - general_query

#     Query: "{message}"

#     Reply ONLY with a valid route name.
#     """

#     result = llm.invoke(prompt).content.strip().lower()
#     print(result)

#     valid_routes = {"emergency_outbreak", "symptom", "vaccination_schedule", "general_query"}
#     route = result if result in valid_routes else "general_query"

#     state.route_decision = {"route": route, "reason": "llm_classified"}
#     return state

SYMPTOM_KEYWORDS = [
    "fever", "cough", "pain", "headache", "vomit", "sore throat", "stomach ache",
    "rash", "fatigue", "tired", "dizzy", "symptom", "disease", "infection",
    "flu", "cold", "chills", "breathing problem", "allergy", "diarrhea"
]

OUTBREAK_KEYWORDS = [
    "outbreak", "epidemic", "pandemic", "cases in", "spread", "number of cases",
    "hotspot", "disease outbreak", "situation report", "alert", "cluster", "local spread"
]

VACCINE_KEYWORDS = [
    "vaccine", "vaccination", "immunization", "dose", "booster", "injection",
    "shot", "schedule", "eligibility", "age group", "when should i take",
    "side effect of vaccine"
]

def node_router(state: HealthGraphState) -> HealthGraphState:
    print("===router===")
    message = (state.user_message or "").lower().strip()

    if not message:
        state.route_decision = {"route": "general_query", "reason": "empty_message"}
        return state

    # Match against expanded corpus
    if any(kw in message for kw in OUTBREAK_KEYWORDS):
        route = "emergency_outbreak"
    elif any(kw in message for kw in SYMPTOM_KEYWORDS):
        route = "symptom"
    elif any(kw in message for kw in VACCINE_KEYWORDS):
        route = "vaccination_schedule"
    else:
        route = "general_query"

    print(route)

    state.route_decision = {"route": route, "reason": "expanded_rule_based"}
    return state



def node_emergency_outbreak(state: HealthGraphState) -> HealthGraphState:
    if not state.user_message:
        state.response = "No message provided."
        return state
    if not state.local_vectorstore:
        state.response = "Outbreak data not indexed."
        return state
    
    print("inside emergency")

    retriever = state.local_vectorstore.as_retriever(search_kwargs={"k": 5})
    conv = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=state.disk_memory.memory if state.disk_memory else None,
        return_source_documents=False,
    )

    print("running llm")
    result = conv.run(question=state.user_message)
    state.response = result
    if state.disk_memory:
        state.disk_memory.persist()
    return state




def node_symptom(state: HealthGraphState) -> HealthGraphState:
    hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = AstraDBVectorStore(
        embedding=hf_embedding,
        api_endpoint=DB_ENDPOINT,
        namespace="default_keyspace",
        token=ASTRA_API_KEY,
        collection_name="medical_v2",
    )
    retriever = vector_store.as_retriever()
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="You are a medical assistant.\nContext:\n{context}\n\nQuestion:\n{question}\nAnswer clearly and answer in simple terms."
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
    )
    state.response = qa_chain.run(state.user_message)
    return state

def node_vaccination_schedule(state):
    message = state.user_message
    if not message:
        return state

    vaccination_json_path = state.vaccination_json_path

    try:
        with open(vaccination_json_path, "r", encoding="utf-8") as f:
            schedule_data = json.load(f)
        docs_json_str = json.dumps(schedule_data, ensure_ascii=False, indent=2)
    except Exception as e:
        state.response = f"(unable to load vaccination schedule JSON: {e})"
        return state

    prompt = f"""
        You are an assistant that knows how to infer vaccination due-dates from a vaccination schedule JSON.
        Use the provided schedule to answer the question as precisely as possible and, when appropriate, return a short checklist.

        SCHEDULE_JSON:
        {docs_json_str}

        QUESTION:
        {message}
    """

    answer = llm.invoke(prompt)
    state.response = answer.content  
    return state


def node_general_query(state: HealthGraphState) -> HealthGraphState:
    resp = llm.invoke([{"role": "user", "content": state.user_message}]).content
    state.response = resp
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

app_graph = workflow.compile()   # <-- renamed so it doesn’t clash with FastAPI app
print("✅ Workflow compiled successfully.")


from fastapi import FastAPI, Form
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import threading

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Twilio credentials
TWILIO_SID = "AC187e0d53b247d8845e8580b09ba28b80"
TWILIO_AUTH = "a806fc81d3079787f692825417663cd6"
TWILIO_WHATSAPP_NUMBER = "whatsapp:+14155238886"
client = Client(TWILIO_SID, TWILIO_AUTH)

# ----------------------------
# Background worker
# ----------------------------
def process_healthgraph_and_reply(sender: str, incoming_msg: str):
    try:
        disk_mem = DiskConversationMemory()

        state = HealthGraphState(
            twilio_payload={"Body": incoming_msg, "From": sender},
            disk_memory=disk_mem
        )

        # Load outbreak docs & FAISS
        state = node_load_outbreak_pdf(state)
        state = node_build_faiss_index(state)

        # Run workflow
        final_state = app_graph.invoke(state)
        response_text = final_state["response"]

        if isinstance(response_text, dict):
            original = response_text.get("original", "")
            translation = response_text.get("translation_raw", "")
        else:
            original = str(response_text)
            translation = ""

        combined_reply = original
        if translation:
            combined_reply += "\n\n--- Translation/Raw ---\n" + translation

        # Send via Twilio
        max_len = 1500
        for i in range(0, len(combined_reply), max_len):
            chunk = combined_reply[i:i+max_len]
            client.messages.create(
                from_=TWILIO_WHATSAPP_NUMBER,
                to=sender,
                body=chunk
            )
        print(f"✅ Sent WhatsApp reply to {sender}")

    except Exception as e:
        print("❌ Error in async reply:", e)
        client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            to=sender,
            body="Oops! Something went wrong processing your request."
        )

# ----------------------------
# Routes
# ----------------------------
@app.get("/")
async def home():
    return {"status": "Bot is running!"}

@app.post("/whatsapp")
async def whatsapp_webhook(
    Body: str = Form(...),
    From: str = Form(...)
):
    print(f"📩 Incoming from {From}: {Body}")

    threading.Thread(target=process_healthgraph_and_reply, args=(From, Body)).start()

    resp = MessagingResponse()
    resp.message("✅ Got your message! Processing your request, you will receive a reply shortly...")
    return PlainTextResponse(str(resp), media_type="application/xml")

# ----------------------------
# Entry
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=4000, reload=True)