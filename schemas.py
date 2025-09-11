from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class Challenge(BaseModel):
    id: str
    name: str
    category: str
    weight: float
    status: str
    submissions: int
    avgScore: float
    timeRemaining: str
    description: str
    template: str
    testingGuide: str

class MinerProfile(BaseModel):
    id: str
    user_id: str  # References neon_auth.users_sync(id)
    walletAddress: Optional[str] = None
    totalScore: float = 0.0
    rank: Optional[int] = None
    submissions: int = 0
    successRate: int = 0
    totalEarned: str = "0.0"
    joinDate: datetime
    lastActive: datetime
    trustTier: str = "bronze"
    publicProfile: bool = True
    
    class Config:
        from_attributes = True

class MinerProfileCreate(BaseModel):
    user_id: str
    walletAddress: Optional[str] = None
    trustTier: str = "bronze"
    publicProfile: bool = True

class SubmissionBase(BaseModel):
    miner: str  # Maps to miner_profile_id in API logic
    challenge: str  # Maps to challenge_id in API logic
    challenge_name: str  # Challenge name for easy display
    code: str  # Code submitted by the miner
    score: Optional[float] = None
    status: str = "pending"
    time: Optional[str] = None  # Maps to submission_time in API logic

class SubmissionCreate(BaseModel):
    miner: str  # Maps to miner_profile_id in API logic
    challenge: str  # Maps to challenge_id in API logic
    code: str  # Code submitted by the miner
    score: Optional[float] = 0.0  # Default score is 0
    status: str = "pending"
    time: Optional[str] = None  # Maps to submission_time in API logic
    # Note: challenge_name is populated automatically by the backend

class Submission(SubmissionBase):
    id: str
    
    class Config:
        from_attributes = True