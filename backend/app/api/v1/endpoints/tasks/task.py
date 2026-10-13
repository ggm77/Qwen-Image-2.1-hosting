from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.schemas.tasks.tasks import TasksResponse
from app.services.tasks.tasks_service import TasksService

router = APIRouter(
    prefix="/api/v1",
    tags=["tasks"],
)

@router.get("/tasks")
async def get_all_tasks(
    service: TasksService = Depends()
) -> list[TasksResponse]:
    return service.get_all_task_id()

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