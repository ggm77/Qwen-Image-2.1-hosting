from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from app.schemas.tasks.tasks import TasksResponse
from app.services.tasks.tasks_service import TasksService

router = APIRouter(
    prefix="/api/v1",
    tags=["tasks"],
)

@router.get("/tasks")
async def get_all_tasks(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    service: TasksService = Depends()
) -> list[TasksResponse]:
    """최근 작업부터 limit개를 돌려준다. 더 오래된 작업은 offset으로 이어서 조회한다."""
    return service.get_all_task_id(limit=limit, offset=offset)

@router.get("/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    service: TasksService = Depends()
) -> TasksResponse:
    return service.get_task_status(task_id=task_id)

@router.get("/tasks/{task_id}/result")
async def get_task_result(
    task_id: str,
    service: TasksService = Depends()
):
    buf = service.get_task_result(task_id=task_id)

    return StreamingResponse(buf, media_type="image/png")

@router.get("/tasks/{task_id}/inputs/{index}")
async def get_task_input_image(
    task_id: str,
    index: int,
    service: TasksService = Depends()
):
    """작업에 첨부된 입력 이미지를 순서(0부터)로 돌려준다."""
    buf = service.get_task_input_image(task_id=task_id, index=index)

    return StreamingResponse(buf, media_type="image/png")
