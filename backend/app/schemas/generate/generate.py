from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

# 요청 값 범위. 24GB 메모리에서 시험한 범위(최대 1024x1024) 안으로 제한한다.
MIN_IMAGE_SIZE = 64
MAX_IMAGE_SIZE = 1024
IMAGE_SIZE_MULTIPLE = 32  # 파이프라인이 32의 배수로만 생성하므로 다른 값은 조용히 줄어드는 대신 거절한다
MAX_STEPS = 100
MAX_SEED = 2**63 - 1  # SQLite INTEGER(부호 있는 64비트) 범위
MAX_PROMPT_LENGTH = 2000

Prompt = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_PROMPT_LENGTH)]
ImageSize = Annotated[int, Field(ge=MIN_IMAGE_SIZE, le=MAX_IMAGE_SIZE, multiple_of=IMAGE_SIZE_MULTIPLE)]

class GenerateRequest(BaseModel):
    prompt: Prompt
    height: ImageSize = 768
    width: ImageSize = 768
    num_inference_steps: int = Field(default=20, ge=1, le=MAX_STEPS)
    seed: int = Field(default=42, ge=0, le=MAX_SEED)
    precision: Literal["bf16", "int8"] = "bf16"  # DiT 가중치 정밀도

class GenerateResponse(BaseModel):
    task_id: str
    status: str
    prompt: str
    height: int
    width: int
    num_inference_steps: int
    seed: int
    precision: str
