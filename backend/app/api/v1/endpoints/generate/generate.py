from fastapi import APIRouter, Request, Depends, BackgroundTasks

from app.schemas.generate.generate import GenerateRequest, GenerateResponse
from app.services.generate.generate_service import GenerateService

router = APIRouter(
    prefix="/api/v1",
    tags=["generate"],
)

@router.post("/generate")
async def generate(
    body: GenerateRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    service: GenerateService = Depends()
) -> GenerateResponse:
    
    return service.request_generate(
        generate_request=body,
        background_tasks=background_tasks,
        request=request
    )