import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ASTRA_API_KEY = os.getenv("ASTRA_API_KEY")
DB_ENDPOINT = os.getenv("DB_ENDPOINT")

VACCINATION_JSON_PATH = r"/Users/aashutoshkumar/Documents/Projects/healthgraph-assistant/data/vaccination_schedule.json"
OUTBREAK_PDF_PATH = r"/Users/aashutoshkumar/Documents/Projects/healthgraph-assistant/latest_weekly_outbreak/31st_weekly_outbreak.pdf"
INDEX_DIR = r"/Users/aashutoshkumar/Documents/Projects/healthgraph-assistant/exp/faiss_index/index.faiss"
