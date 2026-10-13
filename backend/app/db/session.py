import sqlite3
from contextlib import contextmanager
from app.core.config import settings

def init_db():
    with sqlite3.connect(settings.DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    status TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    num_inference_steps INTEGER NOT NULL,
                    current_step INTEGER NOT NULL DEFAULT 0,
                    file_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 기존 DB에는 current_step 컬럼이 없을 수 있으므로 마이그레이션
        try:
            conn.execute("ALTER TABLE tasks ADD COLUMN current_step INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # 이미 존재함

        conn.commit()

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        yield conn
    finally:
        conn.close()
                     