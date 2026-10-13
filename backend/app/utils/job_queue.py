import queue
import threading
from collections import deque

from app.core.logger import logger
from app.utils.model_manager import ModelManager, PromptEmbeds
from app.utils.pipeline_runner import GenerationJob, generate_image, mark_failed

# 이미지 생성 요청을 큐에 쌓고, 워커 스레드 하나가 묶어서 처리한다.
# 대기 중인 요청이 여러 개면
#   1) 텍스트 인코더를 한 번만 올려 모든 프롬프트를 임베딩하고
#   2) DiT/VAE를 한 번만 올려 모든 이미지를 생성한다.
# 생성 도중 들어온 요청 중 프롬프트 임베딩이 캐시에 있는 것은 DiT를 내리지 않고 이어서 처리한다.
# 워커가 하나뿐이라 동시에 생성되는 이미지는 항상 1개다.
class GenerationWorker:

    def __init__(self, model_manager: ModelManager, device: str):
        self.model_manager = model_manager
        self.device = device
        self._queue: queue.Queue[GenerationJob | None] = queue.Queue()
        self._deferred: list[GenerationJob] = []  # 캐시 미스라 다음 배치로 넘긴 요청
        self._thread = threading.Thread(target=self._loop, name="generation-worker", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._queue.put(None)
        self._thread.join()

    def submit(self, job: GenerationJob):
        self._queue.put(job)

    def _drain(self) -> list[GenerationJob]:
        """지금 큐에 쌓여 있는 요청을 모두 꺼낸다 (기다리지 않는다)."""
        jobs = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return jobs
            if item is None:  # 종료 신호는 다음 get()에서 다시 읽도록 되돌린다
                self._queue.put(None)
                return jobs
            jobs.append(item)

    def _loop(self):
        while True:
            if self._deferred:
                batch, self._deferred = self._deferred, []
            else:
                first = self._queue.get()  # 요청이 올 때까지 대기
                if first is None:
                    return
                batch = [first]
            batch += self._drain()

            try:
                self._process(batch)
            except Exception as ex:
                logger.error(f"[Qwen] 배치 처리 중 예기치 못한 오류: {ex}")
                for job in batch:
                    mark_failed(job.task_id)

    def _process(self, batch: list[GenerationJob]):
        mm = self.model_manager
        logger.info(f"[Qwen] 배치 처리 시작 (요청 {len(batch)}개)")

        # 1단계: 프롬프트 임베딩 (캐시에 없는 프롬프트가 있을 때만 인코더 로딩)
        try:
            embeds = mm.encode_prompts([job.prompt for job in batch])
        except Exception as ex:
            logger.error(f"[Qwen] 프롬프트 인코딩 실패: {ex}")
            for job in batch:
                mark_failed(job.task_id)
            return

        # 2단계: DiT + VAE를 한 번 올려 배치 전체 생성
        try:
            pipe = mm.load_generator()
        except Exception as ex:
            logger.error(f"[Qwen] 생성 모델 로딩 실패: {ex}")
            for job in batch:
                mark_failed(job.task_id)
            return

        pending: deque[tuple[GenerationJob, PromptEmbeds]] = deque(
            (job, embeds[job.prompt]) for job in batch
        )
        try:
            while pending:
                job, (prompt_embeds, prompt_embeds_mask) = pending.popleft()
                generate_image(job, pipe, prompt_embeds, prompt_embeds_mask, self.device)

                if not pending:
                    pending.extend(self._take_cached())
        finally:
            del pipe
            mm.free_memory()

    def _take_cached(self) -> list[tuple[GenerationJob, PromptEmbeds]]:
        """생성 중 새로 들어온 요청 중 임베딩이 캐시에 있는 것만 가져오고, 나머지는 다음 배치로 넘긴다."""
        ready = []
        for job in self._drain():
            cached = self.model_manager.get_cached(job.prompt)
            if cached is not None:
                ready.append((job, cached))
            else:
                self._deferred.append(job)
        if ready:
            logger.info(f"[Qwen] 캐시된 프롬프트 요청 {len(ready)}개를 DiT 재로딩 없이 이어서 처리")
        return ready
