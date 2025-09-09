from sqlalchemy import Column, Integer, String, Float, Boolean
from database import Base

class Challenge(Base):
    __tablename__ = "challenges"

    id = Column(String, primary_key=True)
    name = Column(String)
    category = Column(String)
    weight = Column(Float)
    status = Column(String)
    submissions = Column(Integer)
    avgScore = Column(Float)
    timeRemaining = Column(String)
    description = Column(String)
    template = Column(String)
    testingGuide = Column(String)

class Miner(Base):
    __tablename__ = "miners"

    id = Column(String, primary_key=True)
    fullAddress = Column(String)
    name = Column(String)
    totalScore = Column(Float)
    rank = Column(Integer)
    submissions = Column(Integer)
    successRate = Column(Integer)
    totalEarned = Column(String)
    joinDate = Column(String)
    lastActive = Column(String)
    trustTier = Column(String)
    publicProfile = Column(Boolean)
