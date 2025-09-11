



import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Optional
import models, schemas
from database import engine, get_db
from mock_data import challenges as mock_challenges, miners as mock_miners, submissions as mock_submissions
from pydantic import ValidationError
import threading
import time
import random


models.Base.metadata.create_all(bind=engine)

ENV = os.getenv("ENV")
FRONTEND_DOMAIN = os.getenv("FRONTEND_DOMAIN")
ALLOWED_ORIGINS = "*" if ENV == "development" else [FRONTEND_DOMAIN]

app = FastAPI()

# Mock scoring system
def mock_score_submission(submission_id: str):
    """Mock function to score a submission after 60 seconds"""
    def score_after_delay():
        time.sleep(60)  # Wait 60 seconds
        
        # Get a new database session
        db = next(get_db())
        try:
            # Find the submission
            submission = db.query(models.Submission).filter(models.Submission.id == submission_id).first()
            if submission and submission.status == "pending":
                # Generate a random score between 0.1 and 1.0
                random_score = round(random.uniform(0.1, 1.0), 2)
                
                # Update the submission
                submission.score = random_score
                submission.status = "scored"
                
                db.commit()
                print(f"Scored submission {submission_id} with score {random_score}")
        except Exception as e:
            print(f"Error scoring submission {submission_id}: {e}")
            db.rollback()
        finally:
            db.close()
    
    # Start the scoring task in a separate thread
    thread = threading.Thread(target=score_after_delay)
    thread.daemon = True  # Dies when main thread dies
    thread.start()

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/seed-database")
def seed_database(db: Session = Depends(get_db)):
    try:
        # Validate challenges
        validated_challenges = [schemas.Challenge.model_validate(c) for c in mock_challenges]
        # Validate miners - ignoring 'recentChallenges'
        validated_miners = []
        for m in mock_miners:
            miner_data = {k: v for k, v in m.items() if k != "recentChallenges"}
            validated_miners.append(schemas.MinerProfile.model_validate(miner_data))
        # Validate submissions
        validated_submissions = [schemas.Submission.model_validate(s) for s in mock_submissions]

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors())

    # Clear existing data (in dependency order to avoid foreign key constraints)
    db.query(models.Submission).delete()     # Clear first (has foreign keys to both tables)
    db.query(models.MinerProfile).delete()   # Clear second
    db.query(models.Challenge).delete()      # Clear third
    
    # Insert new data
    db.bulk_insert_mappings(models.Challenge, [c.model_dump() for c in validated_challenges])
    db.bulk_insert_mappings(models.MinerProfile, [m.model_dump() for m in validated_miners])
    db.bulk_insert_mappings(models.Submission, [s.model_dump() for s in validated_submissions])
    
    db.commit()
    
    return {"status": "Database seeded successfully"}


@app.get("/")
def read_root():
    return {"status": "ok"}


@app.get("/challenges", response_model=List[schemas.Challenge])
def get_challenges(db: Session = Depends(get_db)):
    challenges = db.query(models.Challenge).all()
    return challenges


@app.get("/miners", response_model=List[schemas.MinerProfile])
def get_miners(db: Session = Depends(get_db)):
    # For backward compatibility, return miner profiles
    profiles = db.query(models.MinerProfile).all()
    return profiles


@app.get("/challenges/{challenge_id}", response_model=schemas.Challenge)
def get_challenge(challenge_id: str, db: Session = Depends(get_db)):
    challenge = db.query(models.Challenge).filter(models.Challenge.id == challenge_id).first()
    if challenge is None:
        raise HTTPException(status_code=404, detail="Challenge not found")
    return challenge


# MinerProfile endpoints
@app.get("/miner-profiles", response_model=List[schemas.MinerProfile])
def get_miner_profiles(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    profiles = db.query(models.MinerProfile).offset(skip).limit(limit).all()
    return profiles

@app.get("/miner-profiles/{profile_id}", response_model=schemas.MinerProfile)
def get_miner_profile(profile_id: str, db: Session = Depends(get_db)):
    profile = db.query(models.MinerProfile).filter(models.MinerProfile.id == profile_id).first()
    if profile is None:
        raise HTTPException(status_code=404, detail="Miner profile not found")
    return profile

@app.get("/miner-profiles/by-user/{user_id}", response_model=schemas.MinerProfile)
def get_miner_profile_by_user(user_id: str, db: Session = Depends(get_db)):
    profile = db.query(models.MinerProfile).filter(models.MinerProfile.user_id == user_id).first()
    if profile is None:
        raise HTTPException(status_code=404, detail="Miner profile not found for user")
    return profile

@app.post("/miner-profiles", response_model=schemas.MinerProfile, status_code=201)
def create_miner_profile(profile: schemas.MinerProfileCreate, db: Session = Depends(get_db)):
    # Check if profile already exists for this user
    existing = db.query(models.MinerProfile).filter(models.MinerProfile.user_id == profile.user_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Miner profile already exists for this user")
    
    # Create new profile
    profile_id = f"miner_{profile.user_id}"
    db_profile = models.MinerProfile(
        id=profile_id,
        **profile.model_dump()
    )
    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)
    return db_profile

@app.post("/submissions", response_model=schemas.Submission, status_code=201)
def create_submission(submission: schemas.SubmissionCreate, db: Session = Depends(get_db)):
    # Verify miner profile and challenge exist
    miner_profile = db.query(models.MinerProfile).filter(models.MinerProfile.id == submission.miner).first()
    if not miner_profile:
        raise HTTPException(status_code=404, detail="Miner profile not found")
    
    challenge = db.query(models.Challenge).filter(models.Challenge.id == submission.challenge).first()
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge not found")
    
    # Generate unique submission ID
    import uuid
    submission_id = f"sub_{uuid.uuid4().hex[:12]}"
    
    # Create submission with generated ID and challenge name
    submission_data = submission.model_dump()
    db_submission = models.Submission(
        id=submission_id,
        miner=submission.miner,
        challenge=submission.challenge,
        challenge_name=challenge.name,  # Get challenge name from the challenge object
        code=submission.code,
        score=0.0,  # Default score is 0
        status=submission.status,
        time=submission.time
    )
    db.add(db_submission)
    db.commit()
    db.refresh(db_submission)
    
    # Start mock scoring process (60 seconds delay)
    mock_score_submission(submission_id)
    
    return db_submission

# Miner profile submissions
@app.get("/miner-profiles/{profile_id}/submissions", response_model=List[schemas.Submission])
def get_miner_profile_submissions(profile_id: str, db: Session = Depends(get_db)):
    # Check if miner profile exists first
    profile = db.query(models.MinerProfile).filter(models.MinerProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Miner profile not found")
    
    # Get submissions for the miner profile (returns empty list if no submissions)
    submissions = db.query(models.Submission).filter(models.Submission.miner == profile_id).all()
    return submissions

@app.get("/users/{user_id}/submissions", response_model=List[schemas.Submission])
def get_user_submissions(user_id: str, db: Session = Depends(get_db)):
    try:
        # Check if user exists in Neon Auth schema
        result = db.execute(text("SELECT id FROM neon_auth.users_sync WHERE id = :user_id"), {"user_id": user_id})
        user_exists = result.fetchone()
        if not user_exists:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get user's miner profile to find submissions
        miner_profile = db.query(models.MinerProfile).filter(models.MinerProfile.user_id == user_id).first()
        if not miner_profile:
            # User exists but has no miner profile, so no submissions
            return []
        
        # Get submissions for the miner profile (returns empty list if no submissions)
        submissions = db.query(models.Submission).filter(models.Submission.miner == miner_profile.id).all()
        return submissions
    except Exception as e:
        # If neon_auth schema doesn't exist yet, try to find submissions by user_id pattern
        # Look for miner profiles for this user
        miner_profile = db.query(models.MinerProfile).filter(models.MinerProfile.user_id == user_id).first()
        if not miner_profile:
            return []
        submissions = db.query(models.Submission).filter(models.Submission.miner == miner_profile.id).all()
        return submissions

@app.get("/users/{user_id}/complete-profile")
def get_complete_user_profile(user_id: str, db: Session = Depends(get_db)):
    """Get complete user data from Neon Auth + Miner Profile"""
    try:
        # Query Neon Auth user data and miner profile in one go
        result = db.execute(text("""
            SELECT 
                u.id as user_id,
                u.name,
                u.email,
                u.created_at as user_created_at,
                m.id as miner_profile_id,
                m.fullAddress,
                m.totalScore,
                m.rank,
                m.submissions,
                m.successRate,
                m.totalEarned,
                m.joinDate,
                m.lastActive,
                m.trustTier,
                m.publicProfile
            FROM neon_auth.users_sync u
            LEFT JOIN miner_profiles m ON u.id = m.user_id
            WHERE u.id = :user_id
        """), {"user_id": user_id})
        
        user_data = result.fetchone()
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "user": {
                "id": user_data.user_id,
                "name": user_data.name,
                "email": user_data.email,
                "created_at": user_data.user_created_at
            },
            "miner_profile": {
                "id": user_data.miner_profile_id,
                "fullAddress": user_data.fullAddress,
                "totalScore": user_data.totalScore,
                "rank": user_data.rank,
                "submissions": user_data.submissions,
                "successRate": user_data.successRate,
                "totalEarned": user_data.totalEarned,
                "joinDate": user_data.joinDate,
                "lastActive": user_data.lastActive,
                "trustTier": user_data.trustTier,
                "publicProfile": user_data.publicProfile
            } if user_data.miner_profile_id else None
        }
    except Exception as e:
        # If neon_auth schema doesn't exist yet, return error
        raise HTTPException(status_code=500, detail=f"Error querying user data: {str(e)}")

# Challenge endpoints with submissions
@app.get("/challenges/{challenge_id}/submissions", response_model=List[schemas.Submission])
def get_challenge_submissions(challenge_id: str, db: Session = Depends(get_db)):
    # Check if challenge exists first
    challenge = db.query(models.Challenge).filter(models.Challenge.id == challenge_id).first()
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge not found")
    
    # Get submissions for the challenge (returns empty list if no submissions)
    submissions = db.query(models.Submission).filter(models.Submission.challenge == challenge_id).all()
    return submissions

@app.post("/healthz")
def check_health():
   return {"status": "ok"}