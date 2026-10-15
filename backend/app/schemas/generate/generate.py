from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator

# 요청 값 범위. 24GB 메모리에서 시험한 범위(최대 1024x1024) 안으로 제한한다.
MIN_IMAGE_SIZE = 64
MAX_IMAGE_SIZE = 1024
IMAGE_SIZE_MULTIPLE = 32  # 파이프라인이 32의 배수로만 생성하므로 다른 값은 조용히 줄어드는 대신 거절한다
MAX_STEPS = 100
MAX_SEED = 2**63 - 1  # SQLite INTEGER(부호 있는 64비트) 범위
MAX_PROMPT_LENGTH = 2000
# 모델은 참조 이미지를 최대 10장까지 받지만, 24GB 메모리에서는 장수에 비례해 토큰이 늘어 메모리/시간이 크게 든다.
MAX_INPUT_IMAGES = 4
MAX_INPUT_IMAGE_B64_LENGTH = 28 * 1024 * 1024  # 20MB 이미지의 base64 길이(약 27MB)까지

Prompt = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_PROMPT_LENGTH)]
ImageSize = Annotated[int, Field(ge=MIN_IMAGE_SIZE, le=MAX_IMAGE_SIZE, multiple_of=IMAGE_SIZE_MULTIPLE)]
Base64Image = Annotated[str, StringConstraints(min_length=1, max_length=MAX_INPUT_IMAGE_B64_LENGTH)]

class GenerateRequest(BaseModel):
    prompt: Prompt
    # height/width는 둘 다 주거나 둘 다 생략한다. 생략하면 이미지가 있을 때는 마지막 이미지의 종횡비를 따르고,
    # 없으면 768x768로 생성한다.
    height: ImageSize | None = None
    width: ImageSize | None = None
    num_inference_steps: int = Field(default=20, ge=1, le=MAX_STEPS)
    seed: int = Field(default=42, ge=0, le=MAX_SEED)
    precision: Literal["bf16", "int8"] = "bf16"  # DiT 가중치 정밀도
    # 입력(참조) 이미지. base64 문자열 또는 data URL. 있으면 이미지 편집/참조 생성으로 동작한다.
    # 프롬프트에서는 순서대로 이미지1, 이미지2 ... 로 가리킨다.
    images: list[Base64Image] = Field(default_factory=list, max_length=MAX_INPUT_IMAGES)

    @model_validator(mode="after")
    def _height_width_together(self):
        if (self.height is None) != (self.width is None):
            raise ValueError("height와 width는 둘 다 지정하거나 둘 다 생략해야 합니다.")
        return self

class GenerateResponse(BaseModel):
    task_id: str
    status: str
    prompt: str
    height: int
    width: int
    num_inference_steps: int
    seed: int
    precision: str
    input_image_count: int = 0
