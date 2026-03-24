from sqlalchemy import Column, String, Integer, Float, DateTime, JSON
from database import Base
import datetime

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(String, primary_key=True, index=True)
    tenant_slug = Column(String, index=True)  # Links the scan to a specific client
    prediction_name = Column(String)          # e.g., "Sector 7 Audit"
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    mode = Column(String)                     # "SINGLE" or "BATCH"
    meters_scanned = Column(Integer)
    flags_detected = Column(Integer)
    accuracy_score = Column(String)           # e.g., "99.61%"
    execution_time = Column(String)           # e.g., "1.42s"
    
    # We store the actual flagged anomalies as JSON so the snapshot page can render the table
    anomalies = Column(JSON)