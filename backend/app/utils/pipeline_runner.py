import time
from dataclasses import dataclass
from datetime import datetime

import torch

from app.core.logger import logger

from app.db.session import get_db_connection
from app.core.config import settings

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

# 작업 실패 처리
def mark_failed(task_id: str):
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE tasks SET status = ? WHERE task_id = ? AND status IN ('queued', 'processing')",
            ("failed", task_id)
        )
        conn.commit()

# 이미지 1장을 생성하는 함수 (모델은 이미 로딩된 상태로 전달받는다)
def generate_image(
        job: GenerationJob,
        pipe,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor | None,
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

            # Qwen-Image 2.1은 CFG 없이 샘플링 (true_cfg_scale=1.0)
            out = pipe(
                prompt_embeds=prompt_embeds.to(device, dtype=torch.bfloat16),
                prompt_embeds_mask=(
                    prompt_embeds_mask.to(device) if prompt_embeds_mask is not None else None
                ),
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
