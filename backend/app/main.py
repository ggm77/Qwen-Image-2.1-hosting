# uvicorn app.main:app --port=8000

from fastapi import FastAPI
from contextlib import asynccontextmanager

import torch

from app.core.logger import logger
from app.core.middleware import setup_cors
from app.api.v1.endpoints.generate.generate import router as generate_router
from app.api.v1.endpoints.tasks.task import router as tasks_router
from app.db.session import init_db, recover_interrupted_tasks
from app.utils.model_manager import ModelManager
from app.utils.job_queue import GenerationWorker
from app.utils.pipeline_runner import GenerationJob

@asynccontextmanager
async def lifespan(app: FastAPI):

    try:
        init_db()
        logger.info("데이터베이스 초기화 완료")
    except Exception as ex:
        logger.error(f"데이터베이스 초기화 실패: {ex}")
        raise ex

    """앱 시작시 1번만 Diffusers Pipeline 초기화"""

    # device 감지
    if torch.backends.mps.is_available():
        logger.info("MPS를 사용합니다.")
        device = "mps"
    elif torch.cuda.is_available():
        logger.info("CUDA를 사용합니다.")
        device = "cuda"
    else:
        logger.info("CUDA나 MPS를 찾을 수 없습니다. CPU를 사용합니다.")
        device = "cpu"

    # 모델은 요청 시 단계별로 로딩한다 (메모리 절약, ModelManager 참고)
    app.state.model_manager = ModelManager(device)
    app.state.device = device
    # 워커 스레드 1개가 큐의 요청을 묶어서 처리 (동시 생성 1개로 제한)
    app.state.generation_worker = GenerationWorker(app.state.model_manager, device)
    # 이전 실행이 중단된 흔적 정리: 처리 중이던 작업은 실패 처리, 대기 중이던 작업은 다시 큐에 넣는다
    # (워커 시작 전에 모두 넣어서 한 배치로 묶어 처리되게 한다)
    interrupted, requeued = recover_interrupted_tasks()
    for row in requeued:
        app.state.generation_worker.submit(GenerationJob(
            task_id=row["task_id"], prompt=row["prompt"], height=row["height"], width=row["width"],
            num_inference_steps=row["num_inference_steps"], seed=row["seed"], precision=row["precision"],
        ))
    if interrupted or requeued:
        logger.info(f"이전 실행에서 중단된 작업 정리: 실패 처리 {interrupted}개, 다시 큐에 등록 {len(requeued)}개")

    app.state.generation_worker.start()
    logger.info("Qwen-Image-2.1 ModelManager 준비 완료")

    try:
        yield
    finally:
        logger.info("앱 종료 중: 자원 해제 중...")
        app.state.generation_worker.stop()
        del app.state.model_manager

app = FastAPI(lifespan=lifespan)

setup_cors(app)

app.include_router(generate_router)
app.include_router(tasks_router)

@app.get("/ping")
async def ping():
    return "pong"