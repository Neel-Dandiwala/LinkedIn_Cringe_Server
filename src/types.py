from pydantic import BaseModel

class CringeRequest(BaseModel):
    text: str

class CringeResponse(BaseModel):
    score: float
    rating: str
    timestamp: str
    processing_time: float
