
from fastapi import FastAPI, BackgroundTasks, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional


# ===================
# Imports (organized)
# ===================
import os
import uuid
import shutil
import zipfile
import time
import pandas as pd
from fastapi import FastAPI, BackgroundTasks, Query, Request, APIRouter, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
from starlette.background import BackgroundTask
import module_4_update as module_4  # Assuming this is the main SynGlue logic module
from pathos.multiprocessing import ProcessPool  
import multiprocess_synglue
from db_utils import get_db, init_db, get_next_queue_no

def get_config():
    if hasattr(module_4, 'CONFIG'):
        return module_4.CONFIG.copy()
    import ast
    import inspect
    import types
    source = inspect.getsource(module_4)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'CONFIG':
                    code = compile(ast.Module([node], []), '<ast>', 'exec')
                    config_ns = {}
                    exec(code, module_4.__dict__, config_ns)
                    return config_ns['CONFIG'].copy()
    raise AttributeError('CONFIG not found in module_4')




import uuid
import shutil
from fastapi import UploadFile, File
from starlette.background import BackgroundTask
import multiprocess_synglue

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# On startup, resume any queued jobs
@app.on_event("startup")
def resume_queued_jobs():
    import threading
    with get_db() as conn:
        cur = conn.execute("SELECT job_id, target, threshold FROM jobs WHERE status = 'queued' ORDER BY queue_no ASC LIMIT 1")
        row = cur.fetchone()
    if row:
        threading.Thread(target=run_pipeline, args=(row[0], row[1], row[2])).start()

design_router = APIRouter(prefix="/synglue/api/design")

# Job submission model
class JobRequest(BaseModel):
    target: str
    threshold: Optional[float] = 75.0




# Helper to run the pipeline in background and update DB
def run_pipeline(job_id, target, threshold):
    # Mark job as running
    with get_db() as conn:
        conn.execute("UPDATE jobs SET status=? WHERE job_id=?", ("running", job_id))
    output_dir = os.path.join("outputs", "Design_Runs", job_id)
    os.makedirs(output_dir, exist_ok=True)
    CONFIG = get_config()
    CONFIG["output_dir"] = output_dir
    E3 = pd.read_csv(CONFIG["e3_db_path"])
    AA = pd.read_pickle(CONFIG["fragments_db_path"])
    selector = module_4.SynGlueSelector(E3)
    subset_AA = AA[(AA['Protein'].str.upper() == target.upper()) & (AA['percentage'] >= threshold)]
    if subset_AA.empty:
        with get_db() as conn:
            conn.execute("UPDATE jobs SET status=?, error=? WHERE job_id=?", ("failed", "No fragments found for target above threshold.", job_id))
        _start_next_queued_job()
        return
    payload = selector.run_selection(target, subset_AA)
    if "Error" in payload:
        with get_db() as conn:
            conn.execute("UPDATE jobs SET status=?, error=? WHERE job_id=?", ("failed", payload["Error"], job_id))
        _start_next_queued_job()
        return
    wh_smi = payload['Warhead_SMILES']
    e3_smi = payload['E3_Tagged_SMILES']
    module_4.visualize_exit_vectors(wh_smi, e3_smi, output_dir)
    pair_string = f"{wh_smi}|{e3_smi}"
    generated_df, out_path = module_4.run_link_invent(pair_string, CONFIG)
    if generated_df is not None:
        predicted_df = module_4.run_ai_predictions(generated_df, out_path, CONFIG)
        classified_df = module_4.run_linker_classification(predicted_df, wh_smi, e3_smi, out_path, CONFIG)
        module_4.visualize_top_protacs(classified_df, out_path, top_n=3)
        module_4.run_admet_ai(classified_df, out_path, CONFIG, top_n=20)
        zip_path = os.path.join(output_dir, f"{job_id}_results.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    if file.endswith('.zip') or file.endswith('.json') or file == 'progress.log':
                        continue
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)
        with get_db() as conn:
            conn.execute("UPDATE jobs SET status=?, output_dir=?, zip_path=?, error=NULL WHERE job_id=?", ("completed", output_dir, zip_path, job_id))
    else:
        with get_db() as conn:
            conn.execute("UPDATE jobs SET status=?, error=? WHERE job_id=?", ("failed", "Link-INVENT failed.", job_id))
    _start_next_queued_job()

# Helper to start the next queued job (if any)
def _start_next_queued_job():
    import threading
    with get_db() as conn:
        cur = conn.execute("SELECT job_id, target, threshold FROM jobs WHERE status = 'queued' ORDER BY queue_no ASC LIMIT 1")
        row = cur.fetchone()
    if row:
        # Use thread to avoid blocking
        threading.Thread(target=run_pipeline, args=(row[0], row[1], row[2])).start()





# --- Per-endpoint rate limiting config and logic ---

import time

RATE_LIMITS = {
    'submit_job': {'per_second': 5, 'per_minute': 10},
    'job_status': {'per_second': 5, 'per_minute': 20},
    'download_results': {'per_second': 5, 'per_minute': 20},
}

def is_rate_limited(ip: str, endpoint: str) -> (bool, str):
    now = time.time()
    per_second = RATE_LIMITS[endpoint]['per_second']
    per_minute = RATE_LIMITS[endpoint]['per_minute']
    with get_db() as conn:
        # Clean up old entries for this IP/endpoint
        conn.execute("DELETE FROM rate_limit WHERE timestamp < ?", (now - 60,))
        # Insert this request
        conn.execute("INSERT INTO rate_limit (ip_address, endpoint, timestamp) VALUES (?, ?, ?)", (ip, endpoint, now))
        # Count requests in last 1 second
        cur1 = conn.execute("SELECT COUNT(*) FROM rate_limit WHERE ip_address=? AND endpoint=? AND timestamp > ?", (ip, endpoint, now - 1))
        count_1s = cur1.fetchone()[0]
        if count_1s > per_second:
            return True, f"Rate limit exceeded: >{per_second} req/s"
        # Count requests in last 60 seconds
        cur2 = conn.execute("SELECT COUNT(*) FROM rate_limit WHERE ip_address=? AND endpoint=? AND timestamp > ?", (ip, endpoint, now - 60))
        count_60s = cur2.fetchone()[0]
        if count_60s > per_minute:
            return True, f"Rate limit exceeded: >{per_minute} req/min"
    return False, ""

@design_router.post("/submit/")
async def submit_job(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    job_req = JobRequest(**body)
    ip = request.client.host
    limited, reason = is_rate_limited(ip, 'submit_job')
    if limited:
        return JSONResponse(status_code=429, content={"error": reason})
    job_id = str(uuid.uuid4())
    queue_no = get_next_queue_no()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, target, threshold, status, queue_no, ip_address) VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, job_req.target, job_req.threshold, "queued", queue_no, ip)
        )
        # Check if any job is running or queued before this one
        cur = conn.execute("SELECT COUNT(*) FROM jobs WHERE (status = 'running' OR status = 'queued') AND queue_no < ?", (queue_no,))
        jobs_before = cur.fetchone()[0]
    # If no jobs before, start this job
    if jobs_before == 0:
        background_tasks.add_task(run_pipeline, job_id, job_req.target, job_req.threshold)
    return {"job_id": job_id, "status": "queued"}


@design_router.get("/status/")
async def job_status(request: Request, job_id: str = Query(...)):
    ip = request.client.host
    limited, reason = is_rate_limited(ip, 'job_status')
    if limited:
        return JSONResponse(status_code=429, content={"error": reason})
    with get_db() as conn:
        cur = conn.execute("SELECT job_id, status, error, output_dir, zip_path, queue_no FROM jobs WHERE job_id=?", (job_id,))
        row = cur.fetchone()
        if not row:
            return JSONResponse(status_code=404, content={"error": "Job not found"})
        queue_no = row[5]
        # Calculate position: count jobs with lower queue_no and status queued/running
        cur2 = conn.execute("SELECT COUNT(*) FROM jobs WHERE (status = 'queued' OR status = 'running') AND queue_no < ?", (queue_no,))
        position = cur2.fetchone()[0] + 1 if row[1] in ("queued", "running") else None
    job = {
        "job_id": row[0],
        "status": row[1],
        "error": row[2],
        #"output_dir": row[3],
        #"zip_path": row[4],
        #"queue_no": queue_no,
        "queue_position": position,
    }
    return job


@design_router.get("/download/")
async def download_results(request: Request, job_id: str = Query(...)):
    ip = request.client.host
    limited, reason = is_rate_limited(ip, 'download_results')
    if limited:
        return JSONResponse(status_code=429, content={"error": reason})
    with get_db() as conn:
        cur = conn.execute("SELECT zip_path, status FROM jobs WHERE job_id=?", (job_id,))
        row = cur.fetchone()
    if not row or row[1] != "completed":
        return JSONResponse(status_code=404, content={"error": "Results not available"})
    zip_path = row[0]
    if not zip_path or not os.path.exists(zip_path):
        return JSONResponse(status_code=404, content={"error": "Zip file not found"})
    return FileResponse(zip_path, filename=os.path.basename(zip_path))



# =====================
# --- Screen Mapping API ---
# =====================
from db_utils import get_db, get_next_queue_no

screen_router = APIRouter(prefix="/synglue/api/screen")


# For screen jobs, accept a list of molecules (name, smiles)
from typing import List, Dict
class ScreenJobRequest(BaseModel):
    molecules: List[Dict[str, str]]  # Each dict: {"name": str, "smiles": str}


# Existing endpoint: submit a list of molecules (name, smiles) for hybrid mapping
@screen_router.post("/submit/")
async def submit_screen_job(background_tasks: BackgroundTasks, req: ScreenJobRequest):
    """Submit a list of molecules (name, smiles) for hybrid mapping. Returns job_id."""
    job_id = str(uuid.uuid4())
    queue_no = get_next_queue_no('screen')
    input_dir = os.path.join("outputs", "Screen_Runs", job_id)
    os.makedirs(input_dir, exist_ok=True)
    input_path = os.path.join(input_dir, "input.csv")
    # Write the molecules to CSV
    import csv
    with open(input_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "SMILES"])
        for mol in req.molecules:
            writer.writerow([mol["name"], mol["smiles"]])
    with get_db() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, job_type, target, threshold, status, queue_no, ip_address, output_dir, zip_path, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (job_id, 'screen', None, None, "queued", queue_no, None, input_dir, None, None)
        )
    # Start job if no jobs before
    with get_db() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM jobs WHERE job_type='screen' AND (status = 'running' OR status = 'queued') AND queue_no < ?", (queue_no,))
        jobs_before = cur.fetchone()[0]
    if jobs_before == 0:
        background_tasks.add_task(run_hybrid_mapping, job_id, input_path, input_dir)
    return {"job_id": job_id, "status": "queued"}

# New endpoint: accept a CSV upload for screen jobs (columns: SMILES, NAME)
from fastapi import UploadFile, File
import csv

@screen_router.post("/submit_csv/")
async def submit_screen_csv(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Accept a CSV file with columns SMILES and NAME for hybrid mapping."""
    # Read and validate CSV
    import io
    content = await file.read()
    decoded = content.decode('utf-8')
    reader = csv.DictReader(io.StringIO(decoded))
    # Accept both SMILES/Name and Name/SMILES (case-insensitive)
    fieldnames = [f.strip().lower() for f in reader.fieldnames]
    if not (('smiles' in fieldnames and 'name' in fieldnames) or ('smiles' in fieldnames and 'name' in fieldnames)):
        return JSONResponse(status_code=400, content={"error": "CSV must have columns 'SMILES' and 'NAME' (case-insensitive)"})
    # Prepare molecules list
    molecules = []
    for row in reader:
        name = row.get('NAME') or row.get('Name') or row.get('name')
        smiles = row.get('SMILES') or row.get('Smiles') or row.get('smiles')
        if not name or not smiles:
            continue
        molecules.append({"name": name, "smiles": smiles})
    if not molecules:
        return JSONResponse(status_code=400, content={"error": "No valid molecules found in CSV."})
    # Reuse the logic from submit_screen_job
    job_id = str(uuid.uuid4())
    queue_no = get_next_queue_no('screen')
    input_dir = os.path.join("outputs", "Screen_Runs", job_id)
    os.makedirs(input_dir, exist_ok=True)
    input_path = os.path.join(input_dir, "input.csv")
    with open(input_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "SMILES"])
        for mol in molecules:
            writer.writerow([mol["name"], mol["smiles"]])
    with get_db() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, job_type, target, threshold, status, queue_no, ip_address, output_dir, zip_path, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (job_id, 'screen', None, None, "queued", queue_no, None, input_dir, None, None)
        )
    # Start job if no jobs before
    with get_db() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM jobs WHERE job_type='screen' AND (status = 'running' OR status = 'queued') AND queue_no < ?", (queue_no,))
        jobs_before = cur.fetchone()[0]
    if jobs_before == 0:
        background_tasks.add_task(run_hybrid_mapping, job_id, input_path, input_dir)
    return {"job_id": job_id, "status": "queued"}

def run_hybrid_mapping(job_id, input_path, output_dir):
    """Background task to run hybrid mapping and update DB."""
    with get_db() as conn:
        conn.execute("UPDATE jobs SET status=? WHERE job_id=?", ("running", job_id))
    try:
        # Call run_hybrid_engine with the input CSV path for automation
        multiprocess_synglue.run_hybrid_engine(
            db_dir="data_copy", output_dir=output_dir, num_workers=10, csv_path=input_path
        )
        # Always zip both the input query and Hybrid_Mapping_Results.csv
        result_path = os.path.join(output_dir, "Hybrid_Mapping_Results.csv")
        zip_path = os.path.join(output_dir, f"{job_id}_results.zip")
        if os.path.exists(result_path):
            import zipfile
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                # Add the query CSV (input_path)
                if os.path.exists(input_path):
                    zipf.write(input_path, arcname="query.csv")
                # Add the results CSV
                zipf.write(result_path, arcname="Hybrid_Mapping_Results.csv")
            with get_db() as conn:
                conn.execute("UPDATE jobs SET status=?, zip_path=?, error=NULL WHERE job_id=?", ("completed", zip_path, job_id))
        else:
            with get_db() as conn:
                conn.execute("UPDATE jobs SET status=?, error=? WHERE job_id=?", ("failed", "No results generated", job_id))
    except Exception as e:
        with get_db() as conn:
            conn.execute("UPDATE jobs SET status=?, error=? WHERE job_id=?", ("failed", str(e), job_id))

@screen_router.get("/status/")
async def screen_job_status(job_id: str = Query(...)):
    with get_db() as conn:
        cur = conn.execute("SELECT job_id, status, error, zip_path, queue_no FROM jobs WHERE job_id=?", (job_id,))
        row = cur.fetchone()
        if not row:
            return JSONResponse(status_code=404, content={"error": "Job not found"})
        queue_no = row[4]
        cur2 = conn.execute("SELECT COUNT(*) FROM jobs WHERE (status = 'queued' OR status = 'running') AND queue_no < ?", (queue_no,))
        position = cur2.fetchone()[0] + 1 if row[1] in ("queued", "running") else None
    return {
        "job_id": row[0],
        "status": row[1],
        "error": row[2],
        "zip_path": row[3],
        "queue_position": position,
    }

@screen_router.get("/download/")
async def screen_download_results(job_id: str = Query(...)):
    with get_db() as conn:
        cur = conn.execute("SELECT zip_path, status FROM jobs WHERE job_id=?", (job_id,))
        row = cur.fetchone()
    if not row or row[1] != "completed":
        return JSONResponse(status_code=404, content={"error": "Results not available"})
    zip_path = row[0]
    if not zip_path or not os.path.exists(zip_path):
        return JSONResponse(status_code=404, content={"error": "Zip file not found"})
    return FileResponse(zip_path, filename=os.path.basename(zip_path))


# --- API ROUTES ---
# /synglue/api/screen : for screening jobs (hybrid mapping)
# /synglue/api/design : for design jobs (existing endpoints)
app.include_router(design_router)  # /synglue/api/design endpoints
app.include_router(screen_router)  # /synglue/api/screen endpoints

# If you want to explicitly set /synglue/api/design, you can do:
# design_router = APIRouter(prefix="/synglue/api/design")
# (move existing api_router endpoints to design_router)


@app.get("/")
def root():
    return {"message": "SynGlue API backend is running. All endpoints are under /synglue/api"}
