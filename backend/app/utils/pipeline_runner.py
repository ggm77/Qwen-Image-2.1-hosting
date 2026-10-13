import time
from datetime import datetime

import torch

from app.utils.model_manager import ModelManager
from app.core.logger import logger

from app.db.session import get_db_connection
from app.core.config import settings

# 이미지 생성하는 함수
def run_pipeline(
        task_id: str,
        prompt: str,
        height: int,
        width: int,
        num_inference_steps: int,
        seed: int,
        model_manager: ModelManager,
        device: str,
        semaphore
):
    # 전역 세마포어로 동시 생성 제한
    with semaphore:

        # DB 연결
        with get_db_connection() as conn:

            try:
                # 상태를 processing으로 변경
                conn.execute(
                    "UPDATE tasks SET status = ? WHERE task_id = ?",
                    ("processing", task_id)
                )
                conn.commit()
            
                # 이미지 생성 준비
                generator = torch.Generator(device=device).manual_seed(seed)
                start_time = time.perf_counter()

                # 이미지 생성 콜백 함수
                def on_step_end(pipeline, step_index, timestep, callback_kwargs):
                    total = pipeline.num_timesteps
                    current_step = step_index + 1
                    progress = current_step / total * 100

                    # 진행 상황을 DB에 기록 -> 프론트가 폴링으로 조회
                    conn.execute(
                        "UPDATE tasks SET current_step = ? WHERE task_id = ?",
                        (current_step, task_id)
                    )
                    conn.commit()

                    logger.info(f"[Qwen] step {current_step}/{total} ({progress:.1f}%)")

                    return callback_kwargs

                # 1단계: 텍스트 인코더로 프롬프트 임베딩 (끝나면 인코더 해제)
                prompt_embeds, prompt_embeds_mask = model_manager.encode_prompt(prompt)

                # 2단계: DiT + VAE로 이미지 생성
                pipe = model_manager.load_generator()
                try:
                    # Qwen-Image 2.1은 CFG 없이 40스텝 샘플링 (true_cfg_scale=1.0)
                    out = pipe(
                        prompt_embeds=prompt_embeds.to(device, dtype=torch.bfloat16),
                        prompt_embeds_mask=(
                            prompt_embeds_mask.to(device) if prompt_embeds_mask is not None else None
                        ),
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        true_cfg_scale=1.0,
                        generator=generator,
                        callback_on_step_end=on_step_end,
                    )
                    image = out.images[0]
                finally:
                    del pipe
                    model_manager.free_memory()

                # 이미지 저장
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f"{current_time}_{task_id}.png"
                file_path = settings.IMG_DIR + "/" + file_name
                image.save(file_path, format="PNG")

                # 상태를 completed로 변경
                conn.execute(
                    "UPDATE tasks SET status = ?, file_path = ? WHERE task_id = ?",
                    ("completed", file_name, task_id)
                )
                conn.commit()

                # 로깅
                elapsed_time = time.perf_counter() - start_time
                logger.info(f"[Qwen] 이미지 생성 완료 ({elapsed_time:.2f}초 소요)")
        
            except Exception as ex:
                logger.error(f"[Qwen] 이미지 생성 중 오류 발생: {ex}")
                conn.execute(
                    "UPDATE tasks SET status = ? WHERE task_id = ?",
                    ("failed", task_id)
                )
                conn.commit()