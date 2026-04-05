from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Boolean
from database import Base
import datetime

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(String, primary_key=True, index=True)
    tenant_slug = Column(String, index=True)
    prediction_name = Column(String)
    timestamp = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    mode = Column(String)                    # "SINGLE" or "BATCH"
    records_analyzed = Column(Integer)       # how many rows were classified
    theft_detected = Column(Integer)         # how many came back as Theft
    accuracy_score = Column(String)          # "99.61%"
    execution_time = Column(String)          # "1.42s"
    has_full_records = Column(Boolean)       # False if batch exceeded threshold
    theft_predictions = Column(JSON)         # rows classified as Theft
    all_predictions = Column(JSON)           # ALL rows, only saved if under threshold
    
    
class Client(Base):
    __tablename__ = "clients"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    slug = Column(String, unique=True, index=True, nullable=False)
    region = Column(String, default="Global")
    status = Column(String, default="PROVISIONING")  # ACTIVE, PROVISIONING, TESTING, SUSPENDED
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))