import json
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_astradb import AstraDBVectorStore

from src.state import HealthGraphState
from src.utils.llm import llm
from src.config import ASTRA_API_KEY, DB_ENDPOINT


def node_emergency_outbreak(state: HealthGraphState) -> HealthGraphState:
    if not state.user_message:
        state.response = "No message provided."
        return state
    if not state.local_vectorstore:
        state.response = "Outbreak data not indexed."
        return state

    retriever = state.local_vectorstore.as_retriever(search_kwargs={"k": 5})
    conv = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=state.disk_memory.memory if state.disk_memory else None,
        return_source_documents=False,
    )
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
        template=(
            "You are a medical assistant.\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n"
            "Answer clearly and in simple terms. "
            "Never provide a formal diagnosis, but explain possible causes and next steps. "
            "Also keep the answer in simple terms"
        ),
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
    )
    state.response = qa_chain.run(state.user_message)
    return state


def node_vaccination_schedule(state: HealthGraphState) -> HealthGraphState:
    message = state.user_message
    if not message:
        return state

    try:
        with open(state.vaccination_json_path, "r", encoding="utf-8") as f:
            schedule_data = json.load(f)
        docs_json_str = json.dumps(schedule_data, ensure_ascii=False, indent=2)
    except Exception as e:
        state.response = f"(unable to load vaccination schedule JSON: {e})"
        return state

    prompt = f"""
    You are an assistant that infers vaccination due-dates from a vaccination schedule JSON.
    Use the provided schedule to answer precisely. When appropriate, return a short checklist.

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
