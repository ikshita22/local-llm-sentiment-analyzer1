"""Local SQLite history for completed analyses."""
import sqlite3
from io import StringIO
from pathlib import Path
import pandas as pd

DB = Path(__file__).resolve().parent.parent / "analysis_history.db"


def connect():
    con = sqlite3.connect(DB)
    con.execute("CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY, created_at TEXT DEFAULT CURRENT_TIMESTAMP, model TEXT, text_column TEXT, label_column TEXT, labels TEXT, duration REAL, results_csv TEXT)")
    return con


def save_run(results, model, text_column, label_column, labels, duration):
    with connect() as con:
        con.execute("INSERT INTO runs(model,text_column,label_column,labels,duration,results_csv) VALUES (?,?,?,?,?,?)", (model, text_column, label_column, ",".join(labels), duration, results.to_csv(index=False)))


def list_runs():
    with connect() as con:
        return pd.read_sql_query("SELECT id,created_at,model,label_column,duration FROM runs ORDER BY id DESC LIMIT 15", con)


def load_run(run_id):
    with connect() as con:
        row = con.execute("SELECT model,text_column,label_column,labels,duration,results_csv FROM runs WHERE id=?", (run_id,)).fetchone()
    if not row:
        raise ValueError("Saved run not found")
    return pd.read_csv(StringIO(row[5])), {"model": row[0], "text_column": row[1], "label_column": row[2], "labels": row[3].split(","), "duration": row[4]}
