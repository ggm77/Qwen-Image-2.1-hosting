from fastapi import Request, BackgroundTasks
import uuid

from app.schemas.generate.generate import GenerateRequest, GenerateResponse
from app.db.session import get_db_connection
from app.utils.pipeline_runner import run_pipeline
from app.core.logger import logger

class GenerateService:

    def request_generate(
            self,
            generate_request: GenerateRequest,
            background_tasks: BackgroundTasks,
            request: Request
    ) -> GenerateResponse:
        
        # task_id 생성
        task_id = str(uuid.uuid4())

        with get_db_connection() as conn:
            # DB에 작업 저장
            conn.execute(
            """
            INSERT INTO tasks (
                    task_id, prompt, status, seed, height, width, num_inference_steps
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                generate_request.prompt,
                "queued",
                generate_request.seed,
                generate_request.height,
                generate_request.width,
                generate_request.num_inference_steps,
            )
            )
            conn.commit()

        # 백그라운드에서 이미지 생성 작업 실행
        background_tasks.add_task(
            run_pipeline,
            task_id=task_id,
            prompt=generate_request.prompt,
            height=generate_request.height,
            width=generate_request.width,
            num_inference_steps=generate_request.num_inference_steps,
            seed=generate_request.seed,
            model_manager=request.app.state.model_manager,
            device=request.app.state.device,
            semaphore=request.app.state.semaphore
        )

        logger.info(f"이미지 생성 요청 됨 task id: {task_id}")

        return GenerateResponse(
            task_id=task_id,
            status="queued",
            prompt=generate_request.prompt,
            height=generate_request.height,
            width=generate_request.width,
            num_inference_steps=generate_request.num_inference_steps,
            seed=generate_request.seed,
        )