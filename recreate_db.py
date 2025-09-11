#!/usr/bin/env python3

import os
from database import Base, engine
from models import Challenge, MinerProfile, Submission
from mock_data import challenges, miners, submissions

def recreate_database():
    """Drop all tables and recreate them with fresh data"""
    
    # Drop all tables
    print("Dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    
    # Create all tables
    print("Creating all tables...")
    Base.metadata.create_all(bind=engine)
    
    # Seed with mock data
    print("Seeding database with mock data...")
    
    from database import SessionLocal
    db = SessionLocal()
    
    try:
        # Add challenges
        for challenge_data in challenges:
            challenge = Challenge(**challenge_data)
            db.add(challenge)
        
        # Add miner profiles
        for miner_data in miners:
            miner = MinerProfile(**miner_data)
            db.add(miner)
        
        # Add submissions
        for submission_data in submissions:
            submission = Submission(**submission_data)
            db.add(submission)
        
        db.commit()
        print("Database recreated and seeded successfully!")
        
    except Exception as e:
        print(f"Error seeding database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    recreate_database()