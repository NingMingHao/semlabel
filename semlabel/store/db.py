"""SQLite metadata store (projects, classes, images, jobs, events, evaluations)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
  id INTEGER PRIMARY KEY, slug TEXT UNIQUE NOT NULL, name TEXT NOT NULL, description TEXT DEFAULT '',
  created REAL, updated REAL, config TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS classes (
  project_id INTEGER NOT NULL, idx INTEGER NOT NULL, name TEXT NOT NULL, color TEXT NOT NULL,
  created REAL, PRIMARY KEY (project_id, idx)
);
CREATE TABLE IF NOT EXISTS images (
  id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL, name TEXT NOT NULL, file TEXT NOT NULL,
  width INTEGER, height INTEGER, status TEXT DEFAULT 'new', source TEXT DEFAULT '', split TEXT DEFAULT '',
  added REAL, updated REAL, conf REAL, labeled_frac REAL DEFAULT 0, in_memory INTEGER DEFAULT 0,
  pred_model TEXT DEFAULT '', notes TEXT DEFAULT '', sha1 TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS images_project ON images(project_id, status);
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY, project_id INTEGER, kind TEXT, status TEXT, progress INTEGER DEFAULT 0,
  total INTEGER DEFAULT 0, message TEXT DEFAULT '', params TEXT DEFAULT '{}', result TEXT DEFAULT '{}',
  created REAL, started REAL, finished REAL
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY, project_id INTEGER, image_id INTEGER, kind TEXT, ts REAL, payload TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS evals (
  id INTEGER PRIMARY KEY, project_id INTEGER, ts REAL, miou REAL, pixacc REAL, n_images INTEGER,
  memory_size INTEGER, config TEXT, per_class TEXT
);
"""


class DB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.write_lock = threading.RLock()
        c = self.conn()
        c.executescript(SCHEMA)
        c.commit()

    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA foreign_keys=ON")
            self._local.conn = c
        return c

    def q(self, sql: str, args=()) -> list[dict]:
        return [dict(r) for r in self.conn().execute(sql, args).fetchall()]

    def one(self, sql: str, args=()) -> dict | None:
        r = self.conn().execute(sql, args).fetchone()
        return dict(r) if r else None

    def x(self, sql: str, args=()) -> int:
        with self.write_lock:
            c = self.conn()
            cur = c.execute(sql, args)
            c.commit()
            return cur.lastrowid

    def xmany(self, sql: str, rows) -> None:
        with self.write_lock:
            c = self.conn()
            c.executemany(sql, rows)
            c.commit()

    def event(self, project_id: int, kind: str, image_id: int | None = None, **payload):
        self.x("INSERT INTO events(project_id, image_id, kind, ts, payload) VALUES (?,?,?,?,?)",
               (project_id, image_id, kind, time.time(), json.dumps(payload)))
