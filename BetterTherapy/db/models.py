from sqlalchemy import Column, Integer, String, Float, ForeignKey, Text, DateTime, func
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.ext.declarative import declared_attr


Base = declarative_base()


class TimestampMixin(object):
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Request(Base, TimestampMixin):
    __tablename__ = "requests"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True)
    openai_batch_id = Column(String(255), nullable=False, unique=False)
    prompt = Column(Text, nullable=False)
    base_response = Column(Text, nullable=False)
    responses = relationship(
        "MinerResponse", backref="request", cascade="all, delete-orphan"
    )


class BlacklistedMiners(Base):
    __tablename__ = "blacklisted_miners"

    id = Column(Integer, primary_key=True, autoincrement=True)
    miner_id = Column(Integer, nullable=False)
    hotkey = Column(String, nullable=False, unique=True)
    coldkey = Column(String, nullable=False)
    blacklist_count = Column(Integer, default=1)
    reason = Column(String, nullable=False)
    created_at = Column(String)
    updated_at = Column(String)


class MinerResponse(Base, TimestampMixin):
    __tablename__ = "miner_responses"
    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(Integer, ForeignKey("requests.id"), nullable=False)
    miner_id = Column(Integer, nullable=False)
    response_text = Column(Text, nullable=True)
    response_time = Column(Float, nullable=True)
    time_score = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)
    total_score = Column(Float, nullable=True)
