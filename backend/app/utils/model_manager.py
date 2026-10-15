import gc
import glob
import os
import threading
import time
from collections import OrderedDict

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, Qwen3VLForConditionalGeneration
from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from app.core.logger import logger
from app.core.config import settings
from app.utils.input_images import load_input_images
from app.utils.quantization import quantize_linears

# 프롬프트 임베딩 캐시 최대 개수 (임베딩은 CPU 메모리에 보관, 하나당 수 MB~수십 MB)
PROMPT_CACHE_SIZE = 32

# (임베딩, 어텐션 마스크, 이미지 토큰 위치 마스크). 마지막 값은 입력 이미지가 있을 때만 있다.
PromptEmbeds = tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]
# 캐시 키: (프롬프트, 입력 이미지 파일 이름들, 인코더 정밀도). GenerationJob.embed_key와 같다.
EmbedKey = tuple[str, tuple[str, ...], str]

# 가중치 정밀도 (DiT와 텍스트 인코더 모두): bf16(원본) / int8(Linear 가중치 8비트 양자화, 약 절반 크기)
#   DiT: 약 14GB -> 약 7GB / 인코더: 약 16GB -> 약 10GB (언어모델만 양자화, 비전 타워와 lm_head는 그대로)
PRECISIONS = ("bf16", "int8")

# int8 캐시 파일 형식 버전 (양자화 방식을 바꾸면 올려서 기존 캐시를 무효화한다)
INT8_CACHE_FORMAT = "int8-v1"
ENCODER_INT8_CACHE_FORMAT = "encoder-int8-v1"

# Qwen-Image 2.1은 텍스트 인코더(Qwen3-VL 8B, 약 16GB)와 DiT(7B, 약 14GB)가 커서
# 24GB 통합 메모리에서는 동시에 올릴 수 없다.
# 그래서 (1) 인코더만 올려 프롬프트를 임베딩하고 해제한 뒤 (2) DiT + VAE만 올려 생성한다.
# 로딩 비용은 요청 단위가 아니라 배치 단위로 낸다 (job_queue.py 참고).
class ModelManager:

    def __init__(self, device: str):
        self.device = device
        self.dtype = torch.bfloat16
        self._embed_cache: OrderedDict[EmbedKey, PromptEmbeds] = OrderedDict()
        self._lock = threading.Lock()

    def free_memory(self):
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

    def get_cached(self, key: EmbedKey) -> PromptEmbeds | None:
        """캐시된 임베딩을 반환한다. 없으면 None."""
        with self._lock:
            if key not in self._embed_cache:
                return None
            self._embed_cache.move_to_end(key)
            return self._embed_cache[key]

    def _put_cache(self, key: EmbedKey, value: PromptEmbeds):
        self._embed_cache[key] = value
        self._embed_cache.move_to_end(key)
        while len(self._embed_cache) > PROMPT_CACHE_SIZE:
            self._embed_cache.popitem(last=False)

    @staticmethod
    def _condition_images(pipe: QwenImage21Pipeline, names: list[str]):
        """
        입력 이미지를 파이프라인이 생성 때 하는 것과 똑같이 전처리한다 (RGBA 변환 후 약 CONDITION_RESOLUTION² 면적으로 리사이즈).
        인코더가 보는 이미지와 생성 단계의 VAE 입력이 어긋나면 이미지 토큰 수가 달라지므로 계산을 맞춰야 한다.
        """
        resolution = settings.CONDITION_RESOLUTION
        resized = []
        for img in load_input_images(names):
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            width, height, _ = calculate_dimensions(resolution * resolution, img.width / img.height)
            resized.append(pipe.image_processor.resize(img, width=width, height=height))
        return resized

    def load_encoder(self, precision: str = "bf16") -> QwenImage21Pipeline:
        """
        텍스트 인코더만 로딩한다 (DiT/VAE 없음).
        int8은 디스크 캐시가 있으면 그것을 쓰고, 없으면 원본을 양자화한 뒤 캐시로 저장한다.
        """
        if precision not in PRECISIONS:
            raise ValueError(f"지원하지 않는 encoder_precision: {precision}")

        pipe = None
        if precision == "int8" and self._int8_cache_valid(
            settings.ENCODER_INT8_CACHE_PATH, ENCODER_INT8_CACHE_FORMAT, "text_encoder"
        ):
            try:
                text_encoder = self._load_int8_encoder_from_cache()
                pipe = QwenImage21Pipeline.from_pretrained(
                    settings.MODEL_PATH,
                    transformer=None,
                    vae=None,
                    text_encoder=text_encoder,
                    torch_dtype=self.dtype,
                )
                logger.info("[Qwen] int8 인코더 캐시에서 로딩")
            except Exception as ex:
                logger.warning(f"[Qwen] int8 인코더 캐시 로딩 실패, 원본에서 다시 양자화합니다: {ex}")
                pipe = None
        if pipe is not None:
            return pipe.to(self.device)

        pipe = QwenImage21Pipeline.from_pretrained(
            settings.MODEL_PATH,
            transformer=None,
            vae=None,
            torch_dtype=self.dtype,
        )  # CPU에 로딩

        if precision == "int8":
            count = quantize_linears(pipe.text_encoder.model.language_model, self.dtype)
            gc.collect()
            logger.info(f"[Qwen] 텍스트 인코더 int8 양자화 완료 (Linear {count}개)")
            self._save_int8_cache(
                pipe.text_encoder, settings.ENCODER_INT8_CACHE_PATH, ENCODER_INT8_CACHE_FORMAT, "text_encoder"
            )

        return pipe.to(self.device)

    def _load_int8_encoder_from_cache(self) -> Qwen3VLForConditionalGeneration:
        """bf16 가중치를 읽지 않고, 빈 모델에 저장된 int8 가중치를 바로 채운다."""
        config = AutoConfig.from_pretrained(os.path.join(settings.MODEL_PATH, "text_encoder"))
        with init_empty_weights():  # 파라미터를 meta로 만들어 메모리를 쓰지 않는다
            text_encoder = Qwen3VLForConditionalGeneration._from_config(config, torch_dtype=self.dtype)
        quantize_linears(text_encoder.model.language_model, self.dtype, empty=True)

        state_dict = load_file(settings.ENCODER_INT8_CACHE_PATH)  # CPU (mmap)
        text_encoder.load_state_dict(state_dict, strict=True, assign=True)

        leftover = [n for n, t in [*text_encoder.named_parameters(), *text_encoder.named_buffers()] if t.is_meta]
        if leftover:
            raise RuntimeError(f"캐시에서 채워지지 않은 텐서가 있음: {leftover[:3]}")
        return text_encoder.eval()

    def encode_prompts(self, jobs: list) -> dict[EmbedKey, PromptEmbeds]:
        """
        여러 요청(프롬프트 + 입력 이미지)을 한 번에 임베딩한다.
        캐시에 없는 요청이 있을 때만 텍스트 인코더를 로딩하고, 끝나면 즉시 해제한다.
        인코더 정밀도가 다른 요청이 섞여 있으면 정밀도별로 한 번씩 로딩한다.
        """
        with self._lock:
            results: dict[EmbedKey, PromptEmbeds] = {}
            missing: dict[EmbedKey, object] = {}

            for job in jobs:
                key = job.embed_key
                if key in self._embed_cache:
                    self._embed_cache.move_to_end(key)
                    results[key] = self._embed_cache[key]
                elif key not in missing:  # 중복 제거 (순서 유지)
                    missing[key] = job

            if not missing:
                logger.info("[Qwen] 프롬프트 임베딩 캐시 사용 (인코더 로딩 생략)")
                return results

            groups: dict[str, dict[EmbedKey, object]] = {}
            for key, job in missing.items():
                groups.setdefault(job.encoder_precision, {})[key] = job

            for precision, group in groups.items():
                self._encode_group(precision, group, results)

            return results

    def _encode_group(self, precision: str, group: dict[EmbedKey, object], results: dict[EmbedKey, PromptEmbeds]):
        """같은 정밀도의 인코더로 처리할 요청들을 임베딩해 results와 캐시에 넣는다."""
        logger.info(f"[Qwen] 텍스트 인코더 로딩... (encoder_precision={precision}, 요청 {len(group)}개)")
        pipe = self.load_encoder(precision)

        try:
            for index, (key, job) in enumerate(group.items(), start=1):
                logger.info(
                    f"[Qwen] 프롬프트 인코딩 {index}/{len(group)} "
                    f"(입력 이미지 {len(job.input_images)}장): {job.prompt[:40]!r}"
                )
                start_time = time.perf_counter()
                images = self._condition_images(pipe, job.input_images) if job.input_images else None
                with torch.no_grad():
                    embeds, mask, image_pad_mask = pipe.encode_prompt(
                        prompt=job.prompt, image=images, device=self.device,
                    )
                logger.info(f"[Qwen] 프롬프트 인코딩 완료 ({time.perf_counter() - start_time:.2f}초 소요)")

                # 결과만 CPU로 옮겨 보관 (텍스트만 쓸 때는 이미지 토큰이 없으므로 image_pad_mask를 버린다)
                value = (
                    embeds.cpu(),
                    mask.cpu() if mask is not None else None,
                    image_pad_mask.cpu() if images is not None else None,
                )
                self._put_cache(key, value)
                results[key] = value
        finally:
            del pipe
            self.free_memory()
            logger.info("[Qwen] 텍스트 인코더 해제 완료")

    def _source_signature(self, subfolder: str) -> str:
        """원본 가중치 파일이 바뀌었는지 판별하기 위한 값 (샤드 파일 이름과 크기)."""
        files = sorted(glob.glob(os.path.join(settings.MODEL_PATH, subfolder, "*.safetensors")))
        return ",".join(f"{os.path.basename(f)}:{os.path.getsize(f)}" for f in files)

    def _int8_cache_valid(self, path: str, cache_format: str, subfolder: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with safe_open(path, framework="pt") as f:
                meta = f.metadata() or {}
        except Exception as ex:
            logger.warning(f"[Qwen] int8 캐시를 읽을 수 없음, 다시 만듭니다: {ex}")
            return False
        return (
            meta.get("format") == cache_format
            and meta.get("source") == self._source_signature(subfolder)
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

    def _save_int8_cache(self, model: torch.nn.Module, path: str, cache_format: str, subfolder: str):
        """양자화된 모델(DiT 또는 텍스트 인코더)을 디스크에 저장한다. 실패해도 생성에는 영향을 주지 않는다."""
        tmp_path = path + ".tmp"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
            save_file(
                state_dict, tmp_path,
                metadata={"format": cache_format, "source": self._source_signature(subfolder)},
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

        if precision == "int8" and self._int8_cache_valid(settings.INT8_CACHE_PATH, INT8_CACHE_FORMAT, "transformer"):
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
                self._save_int8_cache(pipe.transformer, settings.INT8_CACHE_PATH, INT8_CACHE_FORMAT, "transformer")

        return pipe.to(self.device)
