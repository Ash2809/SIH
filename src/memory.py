import pickle
from pathlib import Path
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
