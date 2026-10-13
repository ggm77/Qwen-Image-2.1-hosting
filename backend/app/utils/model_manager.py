import gc
import glob
import os
import threading
from collections import OrderedDict

import torch
from accelerate import init_empty_weights
from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from app.core.logger import logger
from app.core.config import settings
from app.utils.quantization import quantize_linears

# 프롬프트 임베딩 캐시 최대 개수 (임베딩은 CPU 메모리에 보관, 하나당 수 MB~수십 MB)
PROMPT_CACHE_SIZE = 32

PromptEmbeds = tuple[torch.Tensor, torch.Tensor | None]

# DiT 가중치 정밀도: bf16(원본, 약 14GB) / int8(가중치 8비트 양자화, 약 7GB)
PRECISIONS = ("bf16", "int8")

# int8 캐시 파일 형식 버전 (양자화 방식을 바꾸면 올려서 기존 캐시를 무효화한다)
INT8_CACHE_FORMAT = "int8-v1"

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

    def _source_signature(self) -> str:
        """원본 DiT 파일이 바뀌었는지 판별하기 위한 값 (샤드 파일 이름과 크기)."""
        files = sorted(glob.glob(os.path.join(settings.MODEL_PATH, "transformer", "*.safetensors")))
        return ",".join(f"{os.path.basename(f)}:{os.path.getsize(f)}" for f in files)

    def _int8_cache_valid(self) -> bool:
        path = settings.INT8_CACHE_PATH
        if not os.path.exists(path):
            return False
        try:
            with safe_open(path, framework="pt") as f:
                meta = f.metadata() or {}
        except Exception as ex:
            logger.warning(f"[Qwen] int8 캐시를 읽을 수 없음, 다시 만듭니다: {ex}")
            return False
        return (
            meta.get("format") == INT8_CACHE_FORMAT
            and meta.get("source") == self._source_signature()
        )

    def _load_int8_transformer_from_cache(self) -> QwenImage21Transformer2DModel:
        """bf16 가중치를 읽지 않고, 빈 모델에 저장된 int8 가중치를 바로 채운다."""
        transformer_dir = os.path.join(settings.MODEL_PATH, "transformer")
        config = QwenImage21Transformer2DModel.load_config(transformer_dir)
        with init_empty_weights():  # 파라미터를 meta로 만들어 메모리를 쓰지 않는다
            transformer = QwenImage21Transformer2DModel.from_config(config)
        quantize_linears(transformer, self.dtype, empty=True)

        state_dict = load_file(settings.INT8_CACHE_PATH)  # CPU (mmap)
        transformer.load_state_dict(state_dict, strict=True, assign=True)

        leftover = [n for n, t in [*transformer.named_parameters(), *transformer.named_buffers()] if t.is_meta]
        if leftover:
            raise RuntimeError(f"캐시에서 채워지지 않은 텐서가 있음: {leftover[:3]}")
        return transformer.eval()

    def _save_int8_cache(self, transformer: QwenImage21Transformer2DModel):
        """양자화된 DiT를 디스크에 저장한다. 실패해도 생성에는 영향을 주지 않는다."""
        path = settings.INT8_CACHE_PATH
        tmp_path = path + ".tmp"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            state_dict = {k: v.contiguous() for k, v in transformer.state_dict().items()}
            save_file(
                state_dict, tmp_path,
                metadata={"format": INT8_CACHE_FORMAT, "source": self._source_signature()},
            )
            os.replace(tmp_path, path)  # 쓰다 중단된 파일이 캐시로 인식되지 않도록 원자적으로 교체
            logger.info(f"[Qwen] int8 가중치 캐시 저장 완료: {path}")
        except Exception as ex:
            logger.warning(f"[Qwen] int8 가중치 캐시 저장 실패 (무시하고 계속): {ex}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def load_generator(self, precision: str = "bf16") -> QwenImage21Pipeline:
        """
        DiT + VAE만 로딩한다 (텍스트 인코더 없음).
        int8은 디스크 캐시가 있으면 그것을 쓰고, 없으면 원본을 양자화한 뒤 캐시로 저장한다.
        """
        if precision not in PRECISIONS:
            raise ValueError(f"지원하지 않는 precision: {precision}")

        logger.info(f"[Qwen] DiT/VAE 로딩... (precision={precision})")
        transformer = None

        if precision == "int8" and self._int8_cache_valid():
            try:
                transformer = self._load_int8_transformer_from_cache()
                logger.info("[Qwen] int8 가중치 캐시에서 DiT 로딩")
            except Exception as ex:
                logger.warning(f"[Qwen] int8 캐시 로딩 실패, 원본에서 다시 양자화합니다: {ex}")
                transformer = None

        if transformer is not None:
            pipe = QwenImage21Pipeline.from_pretrained(
                settings.MODEL_PATH,
                transformer=transformer,
                text_encoder=None,
                torch_dtype=self.dtype,
            )
        else:
            pipe = QwenImage21Pipeline.from_pretrained(
                settings.MODEL_PATH,
                text_encoder=None,  # 프로세서(토크나이저)는 파이프라인 초기화에 필요하므로 유지
                torch_dtype=self.dtype,
            )  # CPU에 로딩

            if precision == "int8":
                count = quantize_linears(pipe.transformer, self.dtype)
                gc.collect()
                logger.info(f"[Qwen] DiT int8 양자화 완료 (Linear {count}개)")
                self._save_int8_cache(pipe.transformer)

        return pipe.to(self.device)
