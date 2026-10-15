from fastapi import APIRouter, Request, Depends
from fastapi.concurrency import run_in_threadpool

from app.schemas.generate.generate import GenerateRequest, GenerateResponse
from app.services.generate.generate_service import GenerateService

router = APIRouter(
    prefix="/api/v1",
    tags=["generate"],
)

@router.post("/generate")
async def generate(
    body: GenerateRequest,
    request: Request,
    service: GenerateService = Depends()
) -> GenerateResponse:

    # 입력 이미지 디코딩/저장이 이벤트 루프를 막지 않도록 스레드풀에서 실행
    return await run_in_threadpool(
        service.request_generate,
        generate_request=body,
        request=request,
    )