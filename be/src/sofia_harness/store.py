from __future__ import annotations
import json, sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, manifest_path TEXT NOT NULL, schema_version TEXT, status TEXT, created_at TEXT, ingested_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS pages(page_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), source_path TEXT NOT NULL, source_sha256 TEXT NOT NULL, width INTEGER, height INTEGER, qc_status TEXT, qc_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS articles(article_id TEXT PRIMARY KEY, page_id TEXT NOT NULL REFERENCES pages(page_id), reading_order INTEGER, title TEXT, verbatim_text TEXT NOT NULL DEFAULT '', normalized_text TEXT NOT NULL DEFAULT '', confidence REAL, provenance_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS regions(region_id TEXT PRIMARY KEY, page_id TEXT NOT NULL REFERENCES pages(page_id), article_id TEXT, type TEXT NOT NULL, reading_order INTEGER, verbatim_text TEXT, normalized_text TEXT, confidence REAL, status TEXT NOT NULL, polygon_json TEXT NOT NULL, provenance_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reviews(review_id INTEGER PRIMARY KEY AUTOINCREMENT, page_id TEXT NOT NULL, region_id TEXT, reasons_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open', correction TEXT, reviewer TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS entities(entity_id TEXT PRIMARY KEY, kind TEXT NOT NULL, canonical_name TEXT NOT NULL, aliases_json TEXT NOT NULL DEFAULT '[]');
CREATE TABLE IF NOT EXISTS assertions(assertion_id INTEGER PRIMARY KEY AUTOINCREMENT, subject_id TEXT NOT NULL REFERENCES entities(entity_id), predicate TEXT NOT NULL, object_id TEXT NOT NULL, page_id TEXT NOT NULL, region_id TEXT, evidence TEXT NOT NULL, start_offset INTEGER, end_offset INTEGER, confidence REAL, provenance_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs(job_id INTEGER PRIMARY KEY AUTOINCREMENT, dedupe_key TEXT UNIQUE NOT NULL, stage TEXT NOT NULL, status TEXT NOT NULL, payload_json TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS memory(key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS page_approvals(page_id TEXT PRIMARY KEY REFERENCES pages(page_id), status TEXT NOT NULL, reviewer TEXT, note TEXT, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS batches(batch_id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL, mode TEXT, created_at TEXT NOT NULL, started_at TEXT, completed_at TEXT);
CREATE TABLE IF NOT EXISTS batch_scans(scan_id TEXT PRIMARY KEY, batch_id TEXT NOT NULL REFERENCES batches(batch_id), filename TEXT NOT NULL, source_path TEXT NOT NULL, source_sha256 TEXT NOT NULL, status TEXT NOT NULL, selected INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL, UNIQUE(batch_id,source_path));
CREATE TABLE IF NOT EXISTS stage_events(event_id INTEGER PRIMARY KEY AUTOINCREMENT, scan_id TEXT NOT NULL REFERENCES batch_scans(scan_id), stage TEXT NOT NULL, status TEXT NOT NULL, detail_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS batch_pages(batch_id TEXT NOT NULL REFERENCES batches(batch_id), scan_id TEXT NOT NULL REFERENCES batch_scans(scan_id), page_id TEXT NOT NULL REFERENCES pages(page_id), PRIMARY KEY(batch_id,scan_id,page_id));
CREATE TABLE IF NOT EXISTS page_publications(page_id TEXT PRIMARY KEY REFERENCES pages(page_id), published_at TEXT NOT NULL, reviewer TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS page_transcriptions(page_id TEXT PRIMARY KEY REFERENCES pages(page_id), verbatim_text TEXT NOT NULL, confidence REAL, source_variant TEXT NOT NULL, provenance_json TEXT NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS article_fts USING fts5(article_id UNINDEXED, title, normalized_text, tokenize='unicode61');
"""


class Store:
    def __init__(self, path: str | Path="data/sofia.db"):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path); db.row_factory=sqlite3.Row; db.execute("PRAGMA foreign_keys=ON")
        try: yield db; db.commit()
        except Exception: db.rollback(); raise
        finally: db.close()
    def init(self):
        with self.connect() as db: db.executescript(SCHEMA)
    def enqueue(self,dedupe_key,stage,payload):
        now=datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO jobs(dedupe_key,stage,status,payload_json,created_at,updated_at) VALUES(?,?, 'pending',?,?,?)",(dedupe_key,stage,json.dumps(payload,ensure_ascii=False),now,now))
            return db.execute("SELECT job_id FROM jobs WHERE dedupe_key=?",(dedupe_key,)).fetchone()[0]
    def claim(self):
        with self.connect() as db:
            row=db.execute("SELECT * FROM jobs WHERE status IN ('pending','retry') ORDER BY job_id LIMIT 1").fetchone()
            if not row:return None
            db.execute("UPDATE jobs SET status='running', attempts=attempts+1, updated_at=? WHERE job_id=?",(datetime.now(timezone.utc).isoformat(),row["job_id"]))
            return dict(row)
    def complete_job(self,job_id,error=None,max_attempts=3):
        with self.connect() as db:
            attempts=db.execute("SELECT attempts FROM jobs WHERE job_id=?",(job_id,)).fetchone()[0]
            status="complete" if not error else ("retry" if attempts<max_attempts else "failed")
            db.execute("UPDATE jobs SET status=?,error=?,updated_at=? WHERE job_id=?",(status,error,datetime.now(timezone.utc).isoformat(),job_id))
    def stats(self):
        with self.connect() as db:
            return {table:db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in ["runs","pages","articles","regions","reviews","entities","assertions","jobs"]}
    def set_memory(self,key,value):
        now=datetime.now(timezone.utc).isoformat()
        with self.connect() as db:db.execute("INSERT INTO memory(key,value_json,updated_at) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,updated_at=excluded.updated_at",(key,json.dumps(value,ensure_ascii=False),now))
    def get_memory(self,key,default=None):
        with self.connect() as db:row=db.execute("SELECT value_json FROM memory WHERE key=?",(key,)).fetchone();return json.loads(row[0]) if row else default
