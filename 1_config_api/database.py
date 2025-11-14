from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Database URL - can be overridden by environment variable
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://campaign_user:campaign_pass@localhost:5432/campaigns"
)

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create SessionLocal class for dependency injection
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for ORM models
Base = declarative_base()


# Campaign ORM Model
class Campaign(Base):
    __tablename__ = "campaigns"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    model_path = Column(String, nullable=False)
    audience_filter = Column(String, nullable=False)
    features = Column(JSON, nullable=False)  # List of feature names
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<Campaign(id={self.id}, name='{self.name}', is_active={self.is_active})>"


# Dependency for FastAPI routes
def get_db():
    """
    Dependency that provides a database session for each request.
    Automatically closes the session after the request is complete.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Create all tables (call this on app startup)
def init_db():
    """Initialize database by creating all tables."""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully")
