import sqlite3
from contextlib import contextmanager

DB_PATH = 'jobs.db'

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                job_type TEXT, -- 'design' or 'screen'
                target TEXT,
                threshold REAL,
                status TEXT,
                error TEXT,
                output_dir TEXT,
                zip_path TEXT,
                queue_no INTEGER,
                ip_address TEXT
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS rate_limit (
                ip_address TEXT,
                endpoint TEXT,
                timestamp REAL
            )
        ''')


# Helper to get next queue number, optionally filtered by job_type
def get_next_queue_no(job_type=None):
    with get_db() as conn:
        if job_type:
            cur = conn.execute('SELECT MAX(queue_no) FROM jobs WHERE job_type=?', (job_type,))
        else:
            cur = conn.execute('SELECT MAX(queue_no) FROM jobs')
        row = cur.fetchone()
        return (row[0] or 0) + 1

# Call this at startup
init_db()
