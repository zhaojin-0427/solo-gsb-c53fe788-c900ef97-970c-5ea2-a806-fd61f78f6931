"""SQLite persistence layer for watermarked assets."""

import os
import sqlite3
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    asset_id          TEXT PRIMARY KEY,
    sha256            TEXT NOT NULL,
    version           INTEGER NOT NULL,
    watermark         TEXT NOT NULL,
    canvas_size       INTEGER NOT NULL,
    alpha             REAL NOT NULL,
    params            TEXT NOT NULL,
    original_image    BLOB NOT NULL,
    watermarked_image BLOB NOT NULL,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class Database:
    """Thin thread-safe wrapper around a SQLite asset store."""

    def __init__(self, path):
        self.path = path
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def insert_asset(self, asset_id, sha256, version, watermark,
                     canvas_size, alpha, params_json,
                     original_png, watermarked_png):
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO assets (asset_id, sha256, version, watermark,"
                " canvas_size, alpha, params, original_image,"
                " watermarked_image) VALUES (?,?,?,?,?,?,?,?,?)",
                (asset_id, sha256, version, watermark, canvas_size, alpha,
                 params_json, original_png, watermarked_png))

    def get_asset(self, asset_id):
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT asset_id, sha256, version, watermark, canvas_size,"
                " alpha, params, created_at FROM assets WHERE asset_id = ?",
                (asset_id,)).fetchone()
        return dict(row) if row else None

    def get_original_image(self, asset_id):
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT original_image FROM assets WHERE asset_id = ?",
                (asset_id,)).fetchone()
        return row["original_image"] if row else None
