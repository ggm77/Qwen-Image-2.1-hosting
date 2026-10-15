import queue
import threading
from collections import deque

from app.core.logger import logger
from app.utils.input_images import input_images_exist
from app.utils.model_manager import ModelManager, PromptEmbeds
from app.utils.pipeline_runner import GenerationInterrupted, GenerationJob, generate_image, mark_failed, set_status

# 이미지 생성 요청을 큐에 쌓고, 워커 스레드 하나가 묶어서 처리한다.
# 대기 중인 요청이 여러 개면
#   1) 텍스트 인코더를 한 번만 올려 모든 요청(프롬프트+입력 이미지)을 임베딩하고
#   2) DiT/VAE를 정밀도(bf16/int8)별로 한 번씩만 올려 해당 이미지를 모두 생성한다.
# 생성 도중 들어온 요청 중 같은 정밀도이고 임베딩(프롬프트+입력 이미지)이 캐시에 있는 것은 DiT를 내리지 않고 이어서 처리한다.
# 워커가 하나뿐이라 동시에 생성되는 이미지는 항상 1개다.
class GenerationWorker:

    def __init__(self, model_manager: ModelManager, device: str):
        self.model_manager = model_manager
        self.device = device
        self._queue: queue.Queue[GenerationJob | None] = queue.Queue()
        self._deferred: list[GenerationJob] = []  # 캐시 미스라 다음 배치로 넘긴 요청
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="generation-worker", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        """
        워커를 멈춘다. 대기 중인 요청은 처리하지 않고 DB에 queued로 남겨 다음 시작 때 다시 처리한다.
        생성 중인 이미지는 다음 스텝에서 중단되고 queued로 되돌아간다 (인코더 로딩 중이면 그 단계가 끝난 뒤 멈춘다).
        """
        self._stop_event.set()
        self._queue.put(None)  # get()에서 대기 중인 워커를 깨운다
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
                if first is None or self._stop_event.is_set():
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
        if self._stop_event.is_set():
            return
        logger.info(f"[Qwen] 배치 처리 시작 (요청 {len(batch)}개)")

        # 입력 이미지 파일이 없는 요청은 이 요청만 실패시키고 나머지는 계속 처리한다
        usable = []
        for job in batch:
            if input_images_exist(job.input_images):
                usable.append(job)
            else:
                logger.error(f"[Qwen] 입력 이미지 파일을 찾을 수 없음: task_id={job.task_id}")
                mark_failed(job.task_id)
        batch = usable
        if not batch:
            return

        # 1단계: 프롬프트 임베딩 (캐시에 없는 요청이 있을 때만 인코더 로딩, 정밀도와 무관)
        to_encode = [job.task_id for job in batch if mm.get_cached(job.embed_key) is None]
        set_status(to_encode, "encoding")
        try:
            embeds = mm.encode_prompts(batch)
        except Exception as ex:
            logger.error(f"[Qwen] 프롬프트 인코딩 실패: {ex}")
            for job in batch:
                mark_failed(job.task_id)
            embeds = None
        if embeds is None:
            # 예외의 traceback이 인코더를 붙잡고 있어 encode_prompts 안에서의 해제는 효과가 없었다.
            # except 블록을 벗어나 traceback이 사라진 뒤에 다시 비운다.
            mm.free_memory()
            return
        set_status(to_encode, "queued")  # 인코딩이 끝났다. DiT 순서가 올 때까지는 다시 대기 중이다.

        # 2단계: DiT는 정밀도별로 하나만 올릴 수 있으므로 정밀도별로 묶어 처리 (먼저 들어온 요청의 정밀도 순)
        groups: dict[str, list[GenerationJob]] = {}
        for job in batch:
            groups.setdefault(job.precision, []).append(job)

        for precision, jobs in groups.items():
            if self._stop_event.is_set():
                return
            self._run_group(precision, jobs, embeds)

    def _run_group(
        self,
        precision: str,
        jobs: list[GenerationJob],
        embeds: dict[tuple, PromptEmbeds],
    ):
        mm = self.model_manager

        set_status([jobs[0].task_id], "loading")  # 먼저 생성될 요청만 로딩 중으로 표시하고, 나머지는 대기 중으로 둔다
        try:
            pipe = mm.load_generator(precision)
        except Exception as ex:
            logger.error(f"[Qwen] 생성 모델 로딩 실패 (precision={precision}): {ex}")
            for job in jobs:
                mark_failed(job.task_id)
            pipe = None
        if pipe is None:
            # 로딩 도중 올라간 가중치는 traceback이 사라진 뒤에야 해제되므로 except 블록 밖에서 비운다
            mm.free_memory()
            return

        pending: deque[tuple[GenerationJob, PromptEmbeds]] = deque(
            (job, embeds[job.embed_key]) for job in jobs
        )
        try:
            while pending and not self._stop_event.is_set():
                job, (prompt_embeds, prompt_embeds_mask, image_pad_mask) = pending.popleft()
                generate_image(
                    job, pipe, prompt_embeds, prompt_embeds_mask, image_pad_mask, self.device,
                    should_stop=self._stop_event.is_set,
                )
                # 요청마다 크기(해상도, 입력 이미지 수)가 달라 이전 이미지의 버퍼(KV 캐시, 어텐션 등)가
                # MPS 캐시에 남아 쌓이므로 이미지마다 비운다 (실패한 이미지도 generate_image 안에서 예외가 끝난 뒤라 해제된다)
                mm.free_memory()

                if not pending:
                    pending.extend(self._take_cached(precision))
        except GenerationInterrupted:
            logger.info("[Qwen] 서버 종료로 생성을 중단했습니다 (남은 작업은 다음 시작 때 이어서 처리)")
        except Exception as ex:
            # 처리하지 못한 요청이 queued/processing으로 남지 않도록 실패 처리 (완료된 작업은 유지됨)
            logger.error(f"[Qwen] 그룹 처리 중 오류 (precision={precision}): {ex}")
            for job, _ in pending:
                mark_failed(job.task_id)
        finally:
            del pipe
            mm.free_memory()

    def _take_cached(self, precision: str) -> list[tuple[GenerationJob, PromptEmbeds]]:
        """
        생성 중 새로 들어온 요청 중 지금 올라간 DiT와 같은 정밀도이고 임베딩이 캐시에 있는 것만 가져온다.
        나머지는 다음 배치로 넘긴다.
        """
        ready = []
        for job in self._drain():
            cached = self.model_manager.get_cached(job.embed_key) if job.precision == precision else None
            if cached is not None:
                ready.append((job, cached))
            else:
                self._deferred.append(job)
        if ready:
            logger.info(f"[Qwen] 캐시된 프롬프트 요청 {len(ready)}개를 DiT 재로딩 없이 이어서 처리")
        return ready
