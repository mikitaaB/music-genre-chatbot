from typing import List, Optional
from pydantic import BaseModel


class AudioMetadata(BaseModel):
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    file_size: Optional[int] = None
    format: Optional[str] = None


class ClassificationResponse(BaseModel):
    genre: str
    recommendations: List[str]