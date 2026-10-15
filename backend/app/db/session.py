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
                    precision TEXT NOT NULL DEFAULT 'bf16',
                    input_images TEXT,
                    file_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 기존 DB에는 current_step 컬럼이 없을 수 있으므로 마이그레이션
        try:
            conn.execute("ALTER TABLE tasks ADD COLUMN current_step INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # 이미 존재함

        # 기존 DB에는 precision 컬럼이 없을 수 있으므로 마이그레이션
        try:
            conn.execute("ALTER TABLE tasks ADD COLUMN precision TEXT NOT NULL DEFAULT 'bf16'")
        except sqlite3.OperationalError:
            pass  # 이미 존재함

        # 기존 DB에는 input_images 컬럼이 없을 수 있으므로 마이그레이션 (입력 이미지 파일 이름의 JSON 배열)
        try:
            conn.execute("ALTER TABLE tasks ADD COLUMN input_images TEXT")
        except sqlite3.OperationalError:
            pass  # 이미 존재함

        conn.commit()

def recover_interrupted_tasks() -> tuple[int, list[sqlite3.Row]]:
    """
    서버가 중단된 뒤 다시 시작할 때 이전에 처리 중이던 작업을 정리한다.
    - processing: 생성 도중 중단돼 결과가 없으므로 failed로 바꾼다.
    - queued: 아직 시작하지 않았으므로 오래된 순으로 반환해서 다시 큐에 넣게 한다.
    반환값: (failed로 바꾼 작업 수, 다시 큐에 넣을 작업 목록)
    """
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        interrupted = conn.execute(
            "UPDATE tasks SET status = 'failed' WHERE status = 'processing'"
        ).rowcount
        queued = conn.execute(
            """
            SELECT task_id, prompt, height, width, num_inference_steps, seed, precision, input_images
            FROM tasks
            WHERE status = 'queued'
            ORDER BY created_at, rowid
            """
        ).fetchall()
        conn.execute("UPDATE tasks SET current_step = 0 WHERE status = 'queued'")
        conn.commit()
        return interrupted, queued
    finally:
        conn.close()

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        yield conn
    finally:
        conn.close()
                     