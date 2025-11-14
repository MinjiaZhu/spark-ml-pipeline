from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class CampaignCreate(BaseModel):
    name: str
    model_path: str  # e.g., "models/campaign_model_v1.pkl"
    audience_filter: str  # SQL-like: "country='US' AND age > 25"
    features: List[str]  # ["recency", "frequency", "monetary"]

class CampaignResponse(CampaignCreate):
    id: int
    created_at: datetime
    is_active: bool
    
    class Config:
        from_attributes = True

class CampaignUpdate(BaseModel):
    name: Optional[str] = None
    model_path: Optional[str] = None
    is_active: Optional[bool] = None