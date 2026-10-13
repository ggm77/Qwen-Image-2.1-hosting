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
