from fastapi import HTTPException

import json
from io import BytesIO

from app.schemas.tasks.tasks import TasksResponse
from app.db.session import get_db_connection
from app.core.config import settings
from app.utils.input_images import input_image_path

class TasksService:

    @staticmethod
    def _to_response(row) -> TasksResponse:
        task = dict(row)
        return TasksResponse(
            task_id=task["task_id"],
            status=task["status"],
            created_at=task["created_at"],
            current_step=task["current_step"],
            num_inference_steps=task["num_inference_steps"],
            prompt=task["prompt"],
            height=task["height"],
            width=task["width"],
            seed=task["seed"],
            precision=task["precision"],
            input_image_count=len(json.loads(task["input_images"] or "[]")),
        )

    def get_all_task_id(self, limit: int = 100, offset: int = 0) -> list[TasksResponse]:
        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                SELECT task_id, status, created_at, current_step, num_inference_steps,
                       prompt, height, width, seed, precision, input_images
                FROM tasks
                ORDER BY created_at DESC, rowid DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset)
            )
            return [self._to_response(row) for row in cursor.fetchall()]

    def get_task_status(self, task_id: str) -> TasksResponse:

        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                SELECT task_id, status, created_at, current_step, num_inference_steps,
                       prompt, height, width, seed, precision, input_images
                FROM tasks
                WHERE task_id = ?
                """,
                (task_id,)
            )
            row = cursor.fetchone()

            if row is None:
                raise HTTPException(status_code=400, detail="작업을 찾을 수 없습니다.")

            return self._to_response(row)

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

    def get_task_input_image(self, task_id: str, index: int) -> BytesIO:

        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT input_images FROM tasks WHERE task_id = ?",
                (task_id,)
            ).fetchone()

        if row is None:
            raise HTTPException(status_code=400, detail="작업을 찾을 수 없습니다.")

        names = json.loads(row["input_images"] or "[]")
        if not 0 <= index < len(names):
            raise HTTPException(status_code=404, detail="입력 이미지를 찾을 수 없습니다.")

        try:
            with open(input_image_path(names[index]), "rb") as f:
                return BytesIO(f.read())
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="입력 이미지 파일을 찾을 수 없습니다.")
