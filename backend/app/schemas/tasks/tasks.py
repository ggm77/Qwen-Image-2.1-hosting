from pydantic import BaseModel

class TasksResponse(BaseModel):
    task_id: str
    status: str
    created_at: str
    current_step: int = 0
    num_inference_steps: int
    prompt: str
    height: int
    width: int
    seed: int
    precision: str
    input_image_count: int = 0


class ClearTasksResponse(BaseModel):
    deleted_tasks: int
    deleted_files: int
    kept_tasks: int  # 진행 중이라 지우지 않은 작업 수
