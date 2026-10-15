from fastapi import HTTPException, Request
import json
import uuid

from app.schemas.generate.generate import GenerateRequest, GenerateResponse
from app.db.session import get_db_connection
from app.utils.pipeline_runner import GenerationJob
from app.core.logger import logger
from app.utils.input_images import InvalidImageError, default_output_size, store_input_image

class GenerateService:

    def request_generate(
            self,
            generate_request: GenerateRequest,
            request: Request
    ) -> GenerateResponse:
        
        # 입력 이미지 검증 및 저장 (DB에 작업을 넣기 전에 해서, 잘못된 이미지로 작업이 남지 않게 한다)
        input_images: list[str] = []
        last_size = None
        for i, data in enumerate(generate_request.images):
            try:
                name, last_size = store_input_image(data)
            except InvalidImageError as ex:
                raise HTTPException(status_code=422, detail=f"images[{i}]: {ex}")
            input_images.append(name)

        # 크기를 생략했으면 마지막 입력 이미지의 종횡비를 따르고, 이미지도 없으면 768x768
        height, width = generate_request.height, generate_request.width
        if height is None:
            height, width = default_output_size(last_size) if last_size else (768, 768)

        # task_id 생성
        task_id = str(uuid.uuid4())

        with get_db_connection() as conn:
            # DB에 작업 저장
            conn.execute(
            """
            INSERT INTO tasks (
                    task_id, prompt, status, seed, height, width, num_inference_steps, precision, encoder_precision, input_images
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                generate_request.prompt,
                "queued",
                generate_request.seed,
                height,
                width,
                generate_request.num_inference_steps,
                generate_request.precision,
                generate_request.encoder_precision,
                json.dumps(input_images) if input_images else None,
            )
            )
            conn.commit()

        # 생성 큐에 작업 등록 (워커가 대기 중인 요청을 묶어서 처리)
        request.app.state.generation_worker.submit(
            GenerationJob(
                task_id=task_id,
                prompt=generate_request.prompt,
                height=height,
                width=width,
                num_inference_steps=generate_request.num_inference_steps,
                seed=generate_request.seed,
                precision=generate_request.precision,
                encoder_precision=generate_request.encoder_precision,
                input_images=input_images,
            )
        )

        logger.info(f"이미지 생성 요청 됨 task id: {task_id}")

        return GenerateResponse(
            task_id=task_id,
            status="queued",
            prompt=generate_request.prompt,
            height=height,
            width=width,
            num_inference_steps=generate_request.num_inference_steps,
            seed=generate_request.seed,
            precision=generate_request.precision,
            encoder_precision=generate_request.encoder_precision,
            input_image_count=len(input_images),
        )