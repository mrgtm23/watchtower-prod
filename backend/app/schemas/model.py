from pydantic import BaseModel
from uuid import UUID
from typing import Dict, Any
from datetime import datetime

class ModelRegister(BaseModel):
    name: str
    framework: str
    owner_id: UUID

class ModelOut(BaseModel):
    id: UUID
    name: str
    framework: str
    artifact_path: str
    sha256: str
    status: str
    meta_info: Dict[str, Any]
    created_at: datetime
    class Config:
        orm_mode = True