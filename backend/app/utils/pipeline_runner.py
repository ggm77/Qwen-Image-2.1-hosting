import functools
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime

import torch

from app.core.logger import logger

from app.db.session import get_db_connection
from app.core.config import settings
from app.utils.input_images import load_input_images

class GenerationInterrupted(Exception):
    """서버 종료 때문에 생성을 중단할 때 사용한다. 작업은 queued로 되돌아가 다음 시작 때 다시 처리된다."""

@dataclass
class GenerationJob:
    task_id: str
    prompt: str
    height: int
    width: int
    num_inference_steps: int
    seed: int
    precision: str = "bf16"
    input_images: list[str] = field(default_factory=list)  # 입력 이미지 파일 이름 (INPUT_IMG_DIR 기준)

    @property
    def embed_key(self) -> tuple[str, tuple[str, ...]]:
        """프롬프트 임베딩 캐시 키. 입력 이미지도 임베딩에 들어가므로 프롬프트만으로는 결정되지 않는다."""
        return self.prompt, tuple(self.input_images)

    @classmethod
    def from_row(cls, row) -> "GenerationJob":
        return cls(
            task_id=row["task_id"], prompt=row["prompt"], height=row["height"], width=row["width"],
            num_inference_steps=row["num_inference_steps"], seed=row["seed"], precision=row["precision"],
            input_images=json.loads(row["input_images"] or "[]"),
        )

# 작업 실패 처리
def mark_failed(task_id: str):
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE tasks SET status = ? WHERE task_id = ? AND status IN ('queued', 'processing')",
            ("failed", task_id)
        )
        conn.commit()

# 미리 계산한 임베딩과 입력 이미지를 같이 쓰려면 image_pad_mask(임베딩 중 이미지 토큰 위치)를 넘겨야 한다.
# 그런데 pipe.__call__은 이 인자를 받지 않고 encode_prompt를 직접 부르므로, 호출하는 동안만 encode_prompt에 끼워 넣는다.
# 텍스트만 쓸 때(image_pad_mask=None)는 파이프라인 기본 동작을 그대로 쓴다.
@contextmanager
def _inject_image_pad_mask(pipe, image_pad_mask: torch.Tensor | None, device: str):
    if image_pad_mask is None:
        yield
        return

    pipe.encode_prompt = functools.partial(pipe.encode_prompt, image_pad_mask=image_pad_mask.to(device))
    try:
        yield
    finally:
        del pipe.encode_prompt  # 인스턴스에 얹은 것만 지워서 원래 메서드로 되돌린다

# 이미지 1장을 생성하는 함수 (모델은 이미 로딩된 상태로 전달받는다)
def generate_image(
        job: GenerationJob,
        pipe,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor | None,
        image_pad_mask: torch.Tensor | None,
        device: str,
        should_stop=lambda: False,
):
    # DB 연결
    with get_db_connection() as conn:

        try:
            # 상태를 processing으로 변경
            conn.execute(
                "UPDATE tasks SET status = ? WHERE task_id = ?",
                ("processing", job.task_id)
            )
            conn.commit()

            # 이미지 생성 준비
            generator = torch.Generator(device=device).manual_seed(job.seed)
            start_time = time.perf_counter()

            # 이미지 생성 콜백 함수
            def on_step_end(pipeline, step_index, timestep, callback_kwargs):
                if should_stop():
                    raise GenerationInterrupted()

                total = pipeline.num_timesteps
                current_step = step_index + 1
                progress = current_step / total * 100

                # 진행 상황을 DB에 기록 -> 프론트가 폴링으로 조회
                conn.execute(
                    "UPDATE tasks SET current_step = ? WHERE task_id = ?",
                    (current_step, job.task_id)
                )
                conn.commit()

                logger.info(f"[Qwen] step {current_step}/{total} ({progress:.1f}%)")

                return callback_kwargs

            # 입력 이미지가 있으면 VAE 쪽에도 같은 이미지가 필요하다 (텍스트 인코더 쪽은 임베딩에 이미 반영됨)
            condition_images = load_input_images(job.input_images) if job.input_images else None

            # Qwen-Image 2.1은 CFG 없이 샘플링 (true_cfg_scale=1.0)
            with _inject_image_pad_mask(pipe, image_pad_mask, device):
                out = pipe(
                    prompt_embeds=prompt_embeds.to(device, dtype=torch.bfloat16),
                    prompt_embeds_mask=(
                        prompt_embeds_mask.to(device) if prompt_embeds_mask is not None else None
                    ),
                    image=condition_images,
                    output_resolution=settings.CONDITION_RESOLUTION,
                    height=job.height,
                    width=job.width,
                    num_inference_steps=job.num_inference_steps,
                    true_cfg_scale=1.0,
                    generator=generator,
                    callback_on_step_end=on_step_end,
                )
            image = out.images[0]

            # 이미지 저장
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{current_time}_{job.task_id}.png"
            file_path = settings.IMG_DIR + "/" + file_name
            image.save(file_path, format="PNG")

            # 상태를 completed로 변경
            conn.execute(
                "UPDATE tasks SET status = ?, file_path = ? WHERE task_id = ?",
                ("completed", file_name, job.task_id)
            )
            conn.commit()

            # 로깅
            elapsed_time = time.perf_counter() - start_time
            logger.info(f"[Qwen] 이미지 생성 완료 ({elapsed_time:.2f}초 소요)")

        except GenerationInterrupted:
            # 서버 종료로 중단: 다음 시작 때 다시 처리되도록 queued로 되돌린다
            conn.execute(
                "UPDATE tasks SET status = ?, current_step = 0 WHERE task_id = ?",
                ("queued", job.task_id)
            )
            conn.commit()
            raise

        except Exception as ex:
            logger.error(f"[Qwen] 이미지 생성 중 오류 발생: {ex}")
            conn.execute(
                "UPDATE tasks SET status = ? WHERE task_id = ?",
                ("failed", job.task_id)
            )
            conn.commit()
