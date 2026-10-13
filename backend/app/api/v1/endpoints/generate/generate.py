from fastapi import APIRouter, Request, Depends

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

    return service.request_generate(
        generate_request=body,
        request=request
    )