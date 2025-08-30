from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

from .models import Base, DatasetBase

DATABASE_URL = "sqlite:///bettertherapy.db"

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


DATASET_DB_URL = os.environ.get("DATASET_DB_URL")
if not DATASET_DB_URL:
    raise ValueError("DATASET_DB_URL is not set")
dataset_engine = create_engine(DATASET_DB_URL, echo=False, future=True)
DatasetSessionLocal = sessionmaker(
    bind=dataset_engine, autoflush=False, autocommit=False
)


def init_main_db():
    Base.metadata.create_all(bind=engine)


def init_dataset_db():
    DatasetBase.metadata.create_all(bind=dataset_engine)
