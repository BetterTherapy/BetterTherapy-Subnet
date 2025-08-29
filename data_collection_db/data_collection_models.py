from sqlalchemy import Column, Integer, String, Text, DateTime, func, Index
from sqlalchemy.ext.declarative import declarative_base
from BetterTherapy.db.models import TimestampMixin

DataCollectionBase = declarative_base()

class DataCollectionLog(DataCollectionBase, TimestampMixin):
    __tablename__ = "data_collection_log"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    sn = Column(Integer, nullable=False, autoincrement=True)  # Serial number
    query_id = Column(String(255), nullable=False, unique=True)  # request_id from validator
    query = Column(Text, nullable=False)  # The prompt (question from base_query_response)
    
    __table_args__ = (
        Index('idx_data_collection_query_id', 'query_id'),
        Index('idx_data_collection_sn', 'sn'),
    )