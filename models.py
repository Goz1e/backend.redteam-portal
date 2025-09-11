from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class Challenge(Base):
    __tablename__ = "challenges"

    id = Column(String, primary_key=True)
    name = Column(String)
    category = Column(String)
    weight = Column(Float)
    status = Column(String)
    submissions = Column(Integer)  # Keep this for quick count
    avgScore = Column(Float)
    timeRemaining = Column(String)
    description = Column(String)
    template = Column(String)
    testingGuide = Column(String)

class MinerProfile(Base):
    __tablename__ = "miner_profiles"
    
    id = Column(String, primary_key=True)  # e.g., "miner_{neon_auth_user_id}"
    user_id = Column(String, nullable=False)  # References neon_auth.users_sync(id)
    
    # Miner-specific fields
    walletAddress = Column(String)  # Wallet address
    totalScore = Column(Float, default=0.0)
    rank = Column(Integer)
    submissions = Column(Integer, default=0)  # Keep for quick count
    successRate = Column(Integer, default=0)
    totalEarned = Column(String, default="0.0")
    joinDate = Column(DateTime, default=datetime.utcnow)
    lastActive = Column(DateTime, default=datetime.utcnow)
    trustTier = Column(String, default="bronze")
    publicProfile = Column(Boolean, default=True)
    
class Submission(Base):
    __tablename__ = "submissions"

    id = Column(String, primary_key=True)
    miner = Column(String, nullable=False)  # Maps to miner_profile_id in API
    challenge = Column(String, nullable=False)  # Maps to challenge_id in API  
    challenge_name = Column(String, nullable=False)  # Challenge name for easy display
    score = Column(Float)
    time = Column(String)  # Maps to submission_time in API
    status = Column(String)  # "pending", "processing", "completed", "failed", "scored"
    code = Column(String, nullable=False)  # Code submitted by the miner