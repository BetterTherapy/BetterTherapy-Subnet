#!/usr/bin/env python3
"""
Script to set up the data_collection PostgreSQL database.
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import sys
from dotenv import load_dotenv
import os

load_dotenv()


def create_database():
    """Create the data_collection database if it doesn't exist."""
    try:
        # Connect to PostgreSQL server (default database)
        conn = psycopg2.connect(
            host="localhost",
            port="5432",
            user="postgres",  # Change as needed
            password=os.environ.get('POSTGRES_DC_PASSWORD'),      # Add your password
            database="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname='data_collection_db'")
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute("CREATE DATABASE data_collection_db")
            print("Created data_collection_db database successfully")
        else:
            print("data_collection_db database already exists")
            
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error creating database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    create_database()