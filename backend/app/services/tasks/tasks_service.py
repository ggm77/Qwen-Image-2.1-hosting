from fastapi import HTTPException

import json
import os
from io import BytesIO

from app.core.logger import logger
from app.schemas.tasks.tasks import ClearTasksResponse, TasksResponse
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
            encoder_precision=task["encoder_precision"],
            input_image_count=len(json.loads(task["input_images"] or "[]")),
        )

    def get_all_task_id(self, limit: int = 100, offset: int = 0) -> list[TasksResponse]:
        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                SELECT task_id, status, created_at, current_step, num_inference_steps,
                       prompt, height, width, seed, precision, encoder_precision, input_images
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
                       prompt, height, width, seed, precision, encoder_precision, input_images
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


    def clear_finished_tasks(self) -> ClearTasksResponse:
        """
        끝난(completed/failed) 작업의 기록과 결과 이미지, 더 이상 쓰이지 않는 입력 이미지를 모두 지운다.
        대기 중이거나 생성 중인 작업은 워커가 아직 쓰고 있으므로 건드리지 않는다.
        """
        with get_db_connection() as conn:
            finished = conn.execute(
                "SELECT task_id, file_path FROM tasks WHERE status IN ('completed', 'failed')"
            ).fetchall()
            # 조회한 작업만 지운다 (그 사이 끝난 작업의 결과 파일이 고아로 남지 않도록)
            conn.executemany("DELETE FROM tasks WHERE task_id = ?", [(row["task_id"],) for row in finished])
            conn.commit()

            remaining = conn.execute("SELECT input_images FROM tasks").fetchall()
            kept = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

        # 입력 이미지는 내용 해시로 저장돼 작업끼리 공유되므로, 남은 작업이 쓰지 않는 것만 지운다
        in_use = {name for row in remaining for name in json.loads(row["input_images"] or "[]")}

        removed_files = 0
        for row in finished:
            if row["file_path"] and self._remove_file(os.path.join(settings.IMG_DIR, os.path.basename(row["file_path"]))):
                removed_files += 1
        for name in os.listdir(settings.INPUT_IMG_DIR):
            if name not in in_use and self._remove_file(os.path.join(settings.INPUT_IMG_DIR, name)):
                removed_files += 1

        logger.info(f"작업 기록 삭제: 작업 {len(finished)}개, 이미지 파일 {removed_files}개 (진행 중이라 남긴 작업 {kept}개)")

        return ClearTasksResponse(deleted_tasks=len(finished), deleted_files=removed_files, kept_tasks=kept)

    @staticmethod
    def _remove_file(path: str) -> bool:
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return False
        except OSError as ex:
            logger.error(f"파일 삭제 실패: {path}: {ex}")
            return False
