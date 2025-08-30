from functools import wraps
from .connection import SessionLocal, DatasetSessionLocal


def session(func):
    @wraps(func)
    def session_wrapper(*args, **kwargs):
        with SessionLocal() as session:
            return func(session, *args, **kwargs)

    return session_wrapper


def dataset_session(func):
    @wraps(func)
    def dataset_session_wrapper(*args, **kwargs):
        with DatasetSessionLocal() as session:
            return func(session, *args, **kwargs)

    return dataset_session_wrapper
