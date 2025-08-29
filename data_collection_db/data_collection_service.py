import typing
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.orm import sessionmaker
import bittensor as bt
from data_collection_db.data_collection_models import DataCollectionLog, DataCollectionBase
import os
from dotenv import load_dotenv

load_dotenv()



class DataCollectionDatabaseService:
    def __init__(self, database_url: str = f"postgresql://postgres:{os.getenv('POSTGRES_DC_PASSWORD')}@localhost:5432/data_collection_db"):
        """
        Initialize data collection database service.
        
        Args:
            database_url: PostgreSQL connection string
                         Format: postgresql://user:password@host:port/database
        """
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.metadata = MetaData()
        self._known_miners = set()
        self._setup_database()
    
    def _setup_database(self):
        """Create tables and initial schema."""
        try:
            # Create the base table
            DataCollectionBase.metadata.create_all(bind=self.engine)
            bt.logging.info("Data collection database tables created successfully")
        except Exception as e:
            bt.logging.error(f"Error setting up data collection database: {e}")
            raise
    
    def _ensure_miner_column(self, miner_id: int):
        """Dynamically add miner column if it doesn't exist."""
        column_name = f"miner_{miner_id}"
        
        if miner_id in self._known_miners:
            return column_name
            
        try:
            with self.engine.connect() as conn:
                # Check if column exists
                result = conn.execute(text(f"""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'data_collection_log' AND column_name = '{column_name}'
                """))
                
                if not result.fetchone():
                    # Add the column
                    conn.execute(text(f"ALTER TABLE data_collection_log ADD COLUMN {column_name} TEXT"))
                    conn.commit()
                    bt.logging.info(f"Added new miner column: {column_name}")
                
                self._known_miners.add(miner_id)
                
        except Exception as e:
            bt.logging.error(f"Error adding miner column {column_name}: {e}")
            
        return column_name
    
    def _get_next_serial_number(self) -> int:
        """Get the next available serial number."""
        try:
            with self.SessionLocal() as session:
                result = session.execute(text("SELECT COALESCE(MAX(sn), 0) + 1 FROM data_collection_log"))
                return result.scalar()
        except Exception as e:
            bt.logging.error(f"Error getting next serial number: {e}")
            return 1  # Start from 1 if error
    
    def log_query_and_responses(
        self, 
        query_id: str, 
        query: str, 
        miner_responses: typing.Dict[int, str]
    ):
        """
        Log a query and all miner responses.
        
        Args:
            query_id: Unique query identifier (e.g., "btai_01JGXXX...")
            query: The actual prompt/query text (the question from base_query_response)
            miner_responses: Dict mapping miner_id -> actual miner response text
        """
        try:
            with self.SessionLocal() as session:
                # Ensure all miner columns exist
                for miner_id in miner_responses.keys():
                    self._ensure_miner_column(miner_id)
                
                # Check if entry already exists
                existing = session.query(DataCollectionLog).filter_by(query_id=query_id).first()
                if existing:
                    bt.logging.warning(f"Query ID {query_id} already exists in data collection log")
                    return
                
                # Create base entry
                log_entry = DataCollectionLog(
                    sn = self._get_next_serial_number(),
                    query_id=query_id,
                    query=query  # This is the prompt (question)
                )
                
                session.add(log_entry)
                session.flush()  # Get the ID
                
                # Update with miner responses using raw SQL for dynamic columns
                if miner_responses:
                    set_clauses = []
                    params = {"log_id": log_entry.id}
                    
                    for miner_id, response in miner_responses.items():
                        column_name = f"miner_{miner_id}"
                        set_clauses.append(f"{column_name} = :response_{miner_id}")
                        params[f"response_{miner_id}"] = response
                    
                    if set_clauses:
                        update_query = f"""
                            UPDATE data_collection_log 
                            SET {', '.join(set_clauses)}
                            WHERE id = :log_id
                        """
                        session.execute(text(update_query), params)
                
                session.commit()
                bt.logging.info(f"Logged query {query_id} with {len(miner_responses)} miner responses")
                
        except Exception as e:
            bt.logging.error(f"Error logging to data collection database: {e}")
            if 'session' in locals():
                session.rollback()
    
    def get_query_log(self, query_id: str) -> typing.Optional[dict]:
        """Get a specific query log entry."""
        try:
            with self.SessionLocal() as session:
                # Get the entry with all columns
                result = session.execute(text(f"SELECT * FROM data_collection_log WHERE query_id = '{query_id}'"))
                row = result.fetchone()
                
                if row:
                    return dict(row._mapping)
                return None
                
        except Exception as e:
            bt.logging.error(f"Error retrieving query log: {e}")
            return None
    
    def get_recent_logs(self, limit: int = 10) -> typing.List[dict]:
        """Get recent query logs."""
        try:
            with self.SessionLocal() as session:
                result = session.execute(text(f"""
                    SELECT * FROM data_collection_log 
                    ORDER BY created_at DESC 
                    LIMIT {limit}
                """))
                
                return [dict(row._mapping) for row in result.fetchall()]
                
        except Exception as e:
            bt.logging.error(f"Error retrieving recent logs: {e}")
            return []
    
    def get_miner_activity(self) -> typing.Dict[str, int]:
        """Get activity count for each miner."""
        try:
            with self.SessionLocal() as session:
                # Get all miner columns
                result = session.execute(text("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'data_collection_log' 
                    AND column_name LIKE 'miner_%'
                """))
                
                miner_columns = [row[0] for row in result.fetchall()]
                activity = {}
                
                for column in miner_columns:
                    count_result = session.execute(text(f"""
                        SELECT COUNT(*) FROM data_collection_log 
                        WHERE {column} IS NOT NULL AND {column} != ''
                    """))
                    activity[column] = count_result.scalar()
                
                return activity
                
        except Exception as e:
            bt.logging.error(f"Error getting miner activity: {e}")
            return {}