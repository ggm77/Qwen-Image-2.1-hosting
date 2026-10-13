# uvicorn app.main:app --port=8000

from fastapi import FastAPI
from contextlib import asynccontextmanager

import torch

from app.core.logger import logger
from app.core.middleware import setup_cors
from app.api.v1.endpoints.generate.generate import router as generate_router
from app.api.v1.endpoints.tasks.task import router as tasks_router
from app.db.session import init_db
from app.utils.model_manager import ModelManager
from app.utils.job_queue import GenerationWorker

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