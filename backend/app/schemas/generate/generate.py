from pydantic import BaseModel

class GenerateRequest(BaseModel):
    prompt: str
    height: int = 768
    width: int = 768
    num_inference_steps: int = 20
    seed: int = 42

class GenerateResponse(BaseModel):
    task_id: str
    status: str
    prompt: str
    height: int
    width: int
    num_inference_steps: int
    seed: int