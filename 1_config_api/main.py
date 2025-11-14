from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List
from . import models, database

app = FastAPI(title="Campaign Config API")

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    database.init_db()

#basic CRUD operations, basic setting
# Create campaign config
@app.post("/campaigns/", response_model=models.CampaignResponse)
def create_campaign(campaign: models.CampaignCreate, db: Session = Depends(database.get_db)):
    """
    Create a new campaign configuration.
    
    In production:
    - This config gets read by the Spark predict job
    - Defines which model to use, which audience to target, etc.
    """
    db_campaign = database.Campaign(
        name=campaign.name,
        model_path=campaign.model_path,
        audience_filter=campaign.audience_filter,
        features=campaign.features  # Which features to use for prediction
    )
    db.add(db_campaign)
    db.commit()
    db.refresh(db_campaign)
    return db_campaign

# Get campaign config (predict job reads this)
@app.get("/campaigns/{campaign_id}", response_model=models.CampaignResponse)
def get_campaign(campaign_id: int, db: Session = Depends(database.get_db)):
    campaign = db.query(database.Campaign).filter(database.Campaign.id == campaign_id).first()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return campaign

# List all campaigns
@app.get("/campaigns/", response_model=List[models.CampaignResponse])
def list_campaigns(db: Session = Depends(database.get_db)):
    return db.query(database.Campaign).all()

# Update campaign
@app.put("/campaigns/{campaign_id}")
def update_campaign(campaign_id: int, campaign: models.CampaignUpdate,
                    db: Session = Depends(database.get_db)):
    db_campaign = db.query(database.Campaign).filter(database.Campaign.id == campaign_id).first()
    if not db_campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    for key, value in campaign.dict(exclude_unset=True).items():
        setattr(db_campaign, key, value)

    db.commit()
    return db_campaign

# Delete campaign
@app.delete("/campaigns/{campaign_id}")
def delete_campaign(campaign_id: int, db: Session = Depends(database.get_db)):
    campaign = db.query(database.Campaign).filter(database.Campaign.id == campaign_id).first()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    db.delete(campaign)
    db.commit()
    return {"message": "Campaign deleted"}