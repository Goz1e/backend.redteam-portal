from pydantic import BaseModel

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

class Miner(BaseModel):
    id: str
    fullAddress: str
    name: str
    totalScore: float
    rank: int
    submissions: int
    successRate: int
    totalEarned: str
    joinDate: str
    lastActive: str
    trustTier: str
    publicProfile: bool