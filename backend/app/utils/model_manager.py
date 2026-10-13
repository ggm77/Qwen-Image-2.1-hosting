import gc
import threading

import torch
from diffusers import QwenImage21Pipeline

from app.core.logger import logger
from app.core.config import settings

# Qwen-Image 2.1은 텍스트 인코더(Qwen3-VL 8B, 약 16GB)와 DiT(7B, 약 14GB)가 커서
# 24GB 통합 메모리에서는 동시에 올릴 수 없다.
# 그래서 (1) 인코더만 올려 프롬프트를 임베딩하고 해제한 뒤 (2) DiT + VAE만 올려 생성한다.
class ModelManager:

    def __init__(self, device: str):
        self.device = device
        self.dtype = torch.bfloat16
        self._embed_cache: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
        self._lock = threading.Lock()

    def free_memory(self):
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

    def encode_prompt(self, prompt: str):
        """텍스트 인코더만 로딩해서 임베딩을 만들고, 인코더는 즉시 해제한다."""
        with self._lock:
            if prompt in self._embed_cache:
                logger.info("[Qwen] 프롬프트 임베딩 캐시 사용")
                return self._embed_cache[prompt]

            logger.info("[Qwen] 텍스트 인코더 로딩...")
            pipe = QwenImage21Pipeline.from_pretrained(
                settings.MODEL_PATH,
                transformer=None,
                vae=None,
                torch_dtype=self.dtype,
            ).to(self.device)

            with torch.no_grad():
                embeds, mask, _ = pipe.encode_prompt(prompt=prompt, device=self.device)

            # 결과만 CPU로 옮겨 보관 (수 MB 수준)
            result = (embeds.cpu(), mask.cpu() if mask is not None else None)
            self._embed_cache = {prompt: result}  # 마지막 프롬프트만 캐시

            del pipe, embeds, mask
            self.free_memory()
            logger.info("[Qwen] 텍스트 인코더 해제 완료")
            return result

    def load_generator(self) -> QwenImage21Pipeline:
        """DiT + VAE만 로딩한다 (텍스트 인코더 없음)."""
        logger.info("[Qwen] DiT/VAE 로딩...")
        return QwenImage21Pipeline.from_pretrained(
            settings.MODEL_PATH,
            text_encoder=None,  # 프로세서(토크나이저)는 파이프라인 초기화에 필요하므로 유지
            torch_dtype=self.dtype,
        ).to(self.device)
