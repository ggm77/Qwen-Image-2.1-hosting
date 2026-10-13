import gc
import threading
from collections import OrderedDict

import torch
from diffusers import QwenImage21Pipeline

from app.core.logger import logger
from app.core.config import settings

# 프롬프트 임베딩 캐시 최대 개수 (임베딩은 CPU 메모리에 보관, 하나당 수 MB~수십 MB)
PROMPT_CACHE_SIZE = 32

PromptEmbeds = tuple[torch.Tensor, torch.Tensor | None]

# Qwen-Image 2.1은 텍스트 인코더(Qwen3-VL 8B, 약 16GB)와 DiT(7B, 약 14GB)가 커서
# 24GB 통합 메모리에서는 동시에 올릴 수 없다.
# 그래서 (1) 인코더만 올려 프롬프트를 임베딩하고 해제한 뒤 (2) DiT + VAE만 올려 생성한다.
# 로딩 비용은 요청 단위가 아니라 배치 단위로 낸다 (job_queue.py 참고).
class ModelManager:

    def __init__(self, device: str):
        self.device = device
        self.dtype = torch.bfloat16
        self._embed_cache: OrderedDict[str, PromptEmbeds] = OrderedDict()
        self._lock = threading.Lock()

    def free_memory(self):
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

    def get_cached(self, prompt: str) -> PromptEmbeds | None:
        """캐시된 임베딩을 반환한다. 없으면 None."""
        with self._lock:
            if prompt not in self._embed_cache:
                return None
            self._embed_cache.move_to_end(prompt)
            return self._embed_cache[prompt]

    def _put_cache(self, prompt: str, value: PromptEmbeds):
        self._embed_cache[prompt] = value
        self._embed_cache.move_to_end(prompt)
        while len(self._embed_cache) > PROMPT_CACHE_SIZE:
            self._embed_cache.popitem(last=False)

    def encode_prompts(self, prompts: list[str]) -> dict[str, PromptEmbeds]:
        """
        여러 프롬프트를 한 번에 임베딩한다.
        캐시에 없는 프롬프트가 있을 때만 텍스트 인코더를 한 번 로딩하고, 끝나면 즉시 해제한다.
        """
        with self._lock:
            results: dict[str, PromptEmbeds] = {}
            missing: list[str] = []

            for prompt in dict.fromkeys(prompts):  # 중복 제거 (순서 유지)
                if prompt in self._embed_cache:
                    self._embed_cache.move_to_end(prompt)
                    results[prompt] = self._embed_cache[prompt]
                else:
                    missing.append(prompt)

            if not missing:
                logger.info("[Qwen] 프롬프트 임베딩 캐시 사용 (인코더 로딩 생략)")
                return results

            logger.info(f"[Qwen] 텍스트 인코더 로딩... (프롬프트 {len(missing)}개)")
            pipe = QwenImage21Pipeline.from_pretrained(
                settings.MODEL_PATH,
                transformer=None,
                vae=None,
                torch_dtype=self.dtype,
            ).to(self.device)

            try:
                for prompt in missing:
                    with torch.no_grad():
                        embeds, mask, _ = pipe.encode_prompt(prompt=prompt, device=self.device)

                    # 결과만 CPU로 옮겨 보관
                    value = (embeds.cpu(), mask.cpu() if mask is not None else None)
                    self._put_cache(prompt, value)
                    results[prompt] = value
            finally:
                del pipe
                self.free_memory()
                logger.info("[Qwen] 텍스트 인코더 해제 완료")

            return results

    def load_generator(self) -> QwenImage21Pipeline:
        """DiT + VAE만 로딩한다 (텍스트 인코더 없음)."""
        logger.info("[Qwen] DiT/VAE 로딩...")
        return QwenImage21Pipeline.from_pretrained(
            settings.MODEL_PATH,
            text_encoder=None,  # 프로세서(토크나이저)는 파이프라인 초기화에 필요하므로 유지
            torch_dtype=self.dtype,
        ).to(self.device)
