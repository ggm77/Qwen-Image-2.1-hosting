from fastapi import HTTPException

from io import BytesIO

from app.schemas.tasks.tasks import TasksResponse
from app.db.session import get_db_connection
from app.core.config import settings

class TasksService:

    def get_all_task_id(self) -> list[TasksResponse]:
        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                SELECT task_id, status, created_at, current_step, num_inference_steps
                FROM tasks
                ORDER BY created_at DESC
                """
            )
            rows = cursor.fetchall()

            tasks = []
            for row in rows:
                task = dict(row)
                tasks.append(
                    TasksResponse(
                        task_id=task["task_id"],
                        status=task["status"],
                        created_at=task["created_at"],
                        current_step=task["current_step"],
                        num_inference_steps=task["num_inference_steps"]
                    )
                )

            return tasks

    def get_task_status(self, task_id: str) -> TasksResponse:

        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                SELECT task_id, status, created_at, current_step, num_inference_steps
                FROM tasks
                WHERE task_id = ?
                """,
                (task_id,)
            )
            row = cursor.fetchone()

            if row is None:
                raise HTTPException(status_code=400, detail="작업을 찾을 수 없습니다.")

            task = dict(row)

            return TasksResponse(
                task_id=task["task_id"],
                status=task["status"],
                created_at=task["created_at"],
                current_step=task["current_step"],
                num_inference_steps=task["num_inference_steps"]
            )
        
    def get_task_result(self, task_id: str) -> BytesIO:
        
        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                SELECT task_id, status, file_path
                FROM tasks
                WHERE task_id = ?
                """,
                (task_id,)
            )
            row = cursor.fetchone()

            if row is None:
                raise HTTPException(status_code=400, detail="작업 결과를 찾을 수 없습니다.")
            
            task = dict(row)

            # 완료되지 않은 작업은 결과 파일이 없다
            if task["status"] == "failed":
                raise HTTPException(status_code=409, detail="작업이 실패해서 결과가 없습니다.")
            if task["status"] != "completed":
                raise HTTPException(
                    status_code=409,
                    detail=f"작업이 아직 완료되지 않았습니다. (상태: {task['status']})"
                )

            file_path = settings.IMG_DIR + "/" + (task["file_path"] or "")

            try:
                with open(file_path, "rb") as f:
                    image_bytes = f.read()
            except (FileNotFoundError, IsADirectoryError):
                raise HTTPException(status_code=404, detail="결과 이미지 파일을 찾을 수 없습니다.")

            buf = BytesIO(image_bytes)
            buf.seek(0)

            return buf
