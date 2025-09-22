from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.2,
)

def get_llm(model: str = None, temperature: float = 0.2):
    """
    Factory function to get a configured LLM.
    Useful if some nodes need a different model/temperature.
    """
    return ChatGoogleGenerativeAI(
        model = "gemini-2.5-flash",
        temperature=temperature,
    )
