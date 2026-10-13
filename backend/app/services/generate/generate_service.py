from fastapi import Request
import uuid

from app.schemas.generate.generate import GenerateRequest, GenerateResponse
from app.db.session import get_db_connection
from app.utils.pipeline_runner import GenerationJob
from app.core.logger import logger

class GenerateService:

    def request_generate(
            self,
            generate_request: GenerateRequest,
            request: Request
    ) -> GenerateResponse:
        
        # task_id 생성
        task_id = str(uuid.uuid4())

        with get_db_connection() as conn:
            # DB에 작업 저장
            conn.execute(
            """
            INSERT INTO tasks (
                    task_id, prompt, status, seed, height, width, num_inference_steps, precision
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                generate_request.prompt,
                "queued",
                generate_request.seed,
                generate_request.height,
                generate_request.width,
                generate_request.num_inference_steps,
                generate_request.precision,
            )
            )
            conn.commit()

        # 생성 큐에 작업 등록 (워커가 대기 중인 요청을 묶어서 처리)
        request.app.state.generation_worker.submit(
            GenerationJob(
                task_id=task_id,
                prompt=generate_request.prompt,
                height=generate_request.height,
                width=generate_request.width,
                num_inference_steps=generate_request.num_inference_steps,
                seed=generate_request.seed,
                precision=generate_request.precision,
            )
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
            precision=generate_request.precision,
        )