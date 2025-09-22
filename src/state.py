from pydantic import BaseModel
from typing import Any, Dict, Optional
import config

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
    vaccination_json_path: Optional[str] = config.VACCINATION_JSON_PATH
    outbreak_pdf_path: Optional[str] = config.OUTBREAK_PDF_PATH
    index_dir: Optional[str] = config.INDEX_DIR
