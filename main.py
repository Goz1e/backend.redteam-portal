



from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import models, schemas
from database import engine, get_db
from mock_data import challenges as mock_challenges, miners as mock_miners
from pydantic import ValidationError


models.Base.metadata.create_all(bind=engine)
app = FastAPI()

@app.post("/seed-database")
def seed_database(db: Session = Depends(get_db)):
    try:
        # Validate challenges
        validated_challenges = [schemas.Challenge.model_validate(c) for c in mock_challenges]
        # Validate miners - ignoring 'recentChallenges'
        validated_miners = []
        for m in mock_miners:
            miner_data = {k: v for k, v in m.items() if k != "recentChallenges"}
            validated_miners.append(schemas.Miner.model_validate(miner_data))

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors())

    # Clear existing data
    db.query(models.Miner).delete()
    db.query(models.Challenge).delete()
    
    # Insert new data
    db.bulk_insert_mappings(models.Challenge, [c.model_dump() for c in validated_challenges])
    db.bulk_insert_mappings(models.Miner, [m.model_dump() for m in validated_miners])
    
    db.commit()
    
    return {"status": "Database seeded successfully"}


@app.get("/")
def read_root():
    return {"status": "ok"}

@app.get("/challenges", response_model=List[schemas.Challenge])
def get_challenges(db: Session = Depends(get_db)):
    challenges = db.query(models.Challenge).all()
    return challenges

@app.get("/miners", response_model=List[schemas.Miner])
def get_miners(db: Session = Depends(get_db)):
    miners = db.query(models.Miner).all()
    return miners