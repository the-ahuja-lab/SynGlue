from fastapi import FastAPI, HTTPException, File, UploadFile, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi_limiter.depends import RateLimiter
from fastapi_limiter import FastAPILimiter
import sqlite3  # Use SQLite instead of pickle
import os
import shutil
import logging
import numpy as np
from slowapi import Limiter
from slowapi.util import get_remote_address

# Initialize FastAPI app
app = FastAPI(debug=True)

# CORS Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3001",
        "http://192.168.30.53:5050",
        "http://192.168.30.53:8000/",
        "*",  # Allow all origins
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
                                    
# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate Limiter Configuration
limiter = Limiter(key_func=get_remote_address)

# SynGlue Service Configuration
SYNGLUE_URL = "http://192.168.30.53:8000"

# SQLite Database Configuration
DB_PATH = "SynGlue/data/All_data.db"

# Check if the database exists
if not os.path.exists(DB_PATH):
    logger.error(f"Database not found at {DB_PATH}")
    raise FileNotFoundError(f"Database not found at {DB_PATH}")

# Function to connect to SQLite database
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable dictionary-like access
    return conn

# Load models and data at startup
@app.on_event("startup")
async def startup_event():
    try:
        conn = get_db_connection()
        logger.info("Connected to SQLite database successfully.")
        conn.close()
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Startup error: {str(e)}")

# Root Endpoint
@app.get("/", summary="Welcome Endpoint")
async def root():
    return {"message": "Welcome to SynGlue API"}

# Search Targets
@app.get("/search/target/", summary="Search Targets")
@limiter.limit("10/minute")
async def search_target(request: Request, compound: str, species: str = "human"):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = "SELECT * FROM targets WHERE compound_name = ? AND species = ?"
        cursor.execute(query, (compound, species))
        results = cursor.fetchall()
        conn.close()

        if not results:
            raise HTTPException(status_code=404, detail="No targets found")

        return {"status": "success", "targets": [dict(row) for row in results]}
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

# Search Compounds
@app.get("/search/compound/", summary="Search Compounds")
@limiter.limit("10/minute")
async def search_compound(request: Request, target: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = "SELECT * FROM compounds WHERE target_id = ?"
        cursor.execute(query, (target,))
        results = cursor.fetchall()
        conn.close()

        if not results:
            raise HTTPException(status_code=404, detail="No compounds found")

        return {"status": "success", "compounds": [dict(row) for row in results]}
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

# Retrieve Data (Example: Fetch all from a table)
@app.get("/data/{table_name}", summary="Retrieve Data")
@limiter.limit("5/minute")
async def get_data(request: Request, table_name: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = f"SELECT * FROM {table_name}"
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()

        if not results:
            raise HTTPException(status_code=404, detail=f"No data found in {table_name}")

        return {"status": "success", "table": table_name, "data": [dict(row) for row in results]}
    except sqlite3.OperationalError:
        raise HTTPException(status_code=400, detail="Invalid table name")
    except Exception as e:
        logger.error(f"Error retrieving data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving data: {str(e)}")

# Run Generator Module
@app.post("/generator/", summary="Run Generator Module")
@limiter.limit("10/minute")
async def run_generator(request: Request, input_data: dict):
    try:
        logger.info(f"Received data for generator module: {input_data}")
        # Replace with actual logic
        generated_output = {"generated": "example_data"}
        return {"status": "success", "generated_data": generated_output}
    except Exception as e:
        logger.error(f"Generator processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generator processing failed: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
