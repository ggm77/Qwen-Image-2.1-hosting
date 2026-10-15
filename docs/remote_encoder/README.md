# 원격 텍스트 인코더 (GTX 1060 4비트) - 보류한 작업

**상태: 보류 (2026-10-15).** 측정까지 끝냈고, 서버 코드는 만들지 않았습니다. 1060 컴퓨터의 RAM을 다른 서비스와 나눠 써야 하는데 부담이 커서 접었습니다. 이 문서는 나중에 다시 하려고 할 때 처음부터 다시 조사하지 않도록 정리한 것입니다. 상세 측정은 `docs/E2E_TEST_REPORT.md`의 7.4를 보세요.

## 1. 왜 하려고 했나

24GB 맥에서는 텍스트 인코더(Qwen3-VL, bf16 약 16GB)와 DiT(약 14GB, int8이면 6.6GB)를 동시에 올릴 수 없어서, 배치마다 인코더를 로딩했다가 해제하고 DiT를 다시 로딩합니다.

| 구간 (int8, 512, 20스텝) | 시간 | 빈도 |
|---|---|---|
| 인코더 로딩 + 임베딩 | 25~34초 | 배치마다 |
| DiT 로딩(int8 캐시) | 1~5초 | 배치마다 |
| 디노이징 + VAE | 약 65초 (3.2초/스텝) | 이미지마다 |

인코더를 다른 GPU에 상주시키면 첫 구간이 약 0.5초가 되고, 맥은 DiT를 계속 올려둘 수 있어 둘째 구간도 사라질 수 있습니다. **요청 하나씩 처리할 때 약 97초가 약 65초**가 됩니다(추정, 서버 연동 측정은 하지 않음). 요청을 한 배치로 묶으면 이득은 이미지당 9초 정도로 줄고, 디노이징 65초는 그대로입니다.

토크나이저만 옮기는 것은 의미가 없습니다(토큰화 약 0.06ms, 프로세서 로딩 0.2초).

## 2. 확인된 것 (측정)

### 품질: 4비트 인코더는 같은 이미지를 재현하지 않지만 쓸 만함
- 인코더를 NF4로 바꾸면 임베딩이 원본 bf16과 달라집니다(토큰별 코사인 유사도 평균 0.94~0.95, 상대 L2 0.30~0.36, 0번 토큰만 0.39).
- 이미지는 구도·디테일이 바뀝니다(PSNR 15.7~26.1dB). 글자(`OPEN 24 HOURS`)·한국어·주제 반영은 유지됐고 깨짐은 없었습니다. **같은 시드로 원본 이미지를 재현해야 하면 쓰면 안 됩니다.**
- 비교 기준: DiT를 int8로 바꿀 때는 28~35dB였습니다.

### GTX 1060 6GB에서 실제로 동작함 (bitsandbytes NF4, fp16 계산)
- 피크 VRAM: `nvidia-smi` 5235MiB(여유 약 900MiB), `max_memory_allocated` 4.98GiB
- 로딩 36.5~40.2초, 프롬프트당 임베딩 0.45초(워밍업 0.82초)
- 시스템 RAM: anon 피크 0.86GiB, RSS 5.5GiB 중 4.9GiB는 safetensors mmap 페이지 캐시. `MemAvailable` 최저 11.5GiB.
  - 보류 사유는 RAM이었습니다. 위 측정으로는 anon이 작지만, 로딩 중 mmap 페이지 캐시와 다른 서비스의 메모리가 겹치면 부담이 됩니다. 사용자가 체감한 정확한 원인은 확인하지 않았습니다. 로딩 방식(예: mmap 대신 순차 읽기, 로딩 후 페이지 캐시 해제 `posix_fadvise` 등)을 바꾸면 달라질 수 있어 다시 시도할 때 먼저 볼 항목입니다.
- NaN/inf 없음(출력 최댓값 448~819), shape는 맥 bf16 기준과 일치.

### 1060 임베딩으로 만든 이미지는 맥 시뮬레이션과 같음
- 임베딩 코사인 유사도가 시뮬레이션과 소수 셋째 자리까지 같았고, 이미지 PSNR은 시뮬레이션 대비 36.5~41.7dB(text 예외: fp16 시뮬레이션 이미지와 같은 모양). 즉 맥에서 시뮬레이션만 해도 품질 판단이 가능합니다.

## 3. 확인하지 못한 것 (다시 하면 먼저 볼 것)
1. **긴 프롬프트의 fp16 오버플로**: 내부 residual이 6~33층에서 최대 13232(fp16 한계 65504의 약 20%)였습니다. 시험한 프롬프트는 26~32토큰이었고 현재 API는 2000자까지 받습니다. 100/500/1000/2000자 영어·한국어로 NaN/inf가 나는 길이를 이진 탐색해야 합니다.
2. VRAM 여유 약 900MiB: 다른 GPU 프로세스가 있으면 OOM이 날 수 있습니다.
3. 프롬프트 4개, 시드 1개, 512 해상도 한 번씩만 비교한 결과라 통계적 품질 평가가 아닙니다.
4. 맥에 연동했을 때의 실제 end-to-end 시간(네트워크 포함)은 측정하지 않았습니다.

## 4. 1060 컴퓨터에서 얻은 구현 지식

- **diffusers를 import할 수 없었습니다.** transformers 4.57.3이 요구하는 `huggingface_hub`(<1.0)와 diffusers main이 요구하는 버전(≥1.32)이 충돌합니다. 대신 설치된 diffusers 파일에서 `QwenImage21Pipeline.encode_prompt` 소스를 **AST로 추출해** 그대로 실행했습니다(템플릿, 왼쪽 패딩, norm hook, `_drop_idx=14`, 마스크 처리 모두 원본 그대로). 출력이 35번째 층 출력에서 앞 14토큰을 뺀 값과 비트 단위로 같았습니다.
- 단일 프롬프트에서 `encode_prompt`는 mask를 `None`으로 돌려줍니다. 1060 쪽에서는 같은 의미의 all-ones int64 마스크를 저장했고, 그 마스크로 만든 이미지가 정상이었습니다.
- VRAM 절약: 비전 타워(`visual`)와 `lm_head`를 GPU에 올리지 않는 것만으로 6GB에 들어갔습니다. `embed_tokens`(fp16 1.16GiB) CPU 이동은 필요 없었습니다.
- NF4 대상은 언어모델의 Linear 252개(36층 × 7), 블록 64. 가중치 상대오차 평균 9.4%.
- 1060은 bf16 하드웨어 지원이 없어서 fp16으로 계산했습니다.
- 환경: torch 2.12.0+cu126, transformers 4.57.3, diffusers 0.41.0.dev0(`--no-deps`, 소스 추출용), bitsandbytes 0.50.2, accelerate 1.15.0, huggingface_hub 0.36.2, 드라이버 580.178.04, 모델 리비전 `d26bb612`. (맥의 diffusers는 0.36.0.dev0 표기이지만 `encode_prompt` 부분은 모든 커밋에서 같음을 확인.)
- 시스템에 `python3.12-venv`가 없어서 `--without-pip`으로 venv를 만들고 시스템 pip로 venv 안에 pip를 넣었습니다(apt 미사용).
- 첫 시도 실패는 자원 부족이 아니라 스크립트 버그였습니다(가중치 키 접두사 오류 → 언어모델이 전부 missing → RAM 6.1GiB 일시 사용). 로딩 후 missing key가 0개인지 꼭 확인하세요.
- 1060 컴퓨터의 기존 작업물: `~/enc4bit_test` (스크립트 `measure_enc4bit.py`, 로그 `logs/`, 임베딩 `embeds_nf4_gtx1060.pt`, 모델 `hf_home`, `venv`, 디스크 약 25GB). 삭제했을 수 있으니 먼저 확인하세요.

## 5. 이 폴더의 스크립트 (맥에서 품질을 다시 확인할 때)

둘 다 `backend/`에서 `unset PYTHONPATH` 후 `.venv/bin/python`으로 실행합니다(전역 PYTHONPATH가 venv를 오염시킴).

- `nf4_sim.py <출력폴더>`: 인코더 가중치를 NF4로 양자화했다 복원해(시뮬레이션) bf16 / NF4 bf16 / NF4 fp16 임베딩과 이미지(int8 DiT, 512, 20스텝, 시드 7)를 만들고 코사인 유사도·PSNR을 출력합니다. 4개 프롬프트, 약 14분.
- `cmp1060.py <nf4_sim 출력폴더>`: 1060에서 가져온 `~/embeds_nf4_gtx1060.pt`로 맥에서 이미지를 만들어 시뮬레이션·원본과 비교합니다. 경로(`/Users/user/embeds_nf4_gtx1060.pt`)가 코드에 고정돼 있고 `nf4_sim.py`가 만든 `*_A_bf16.png`, `*_B_nf4_bf16.png`가 필요합니다.

## 6. 다시 하는 순서

1. **1060에 인코더 서버 만들기** - 아래 7번 프롬프트를 1060 컴퓨터의 Claude Code에 붙여넣습니다(프로그램 이름 추천: `qwen-encoder-server`).
2. **긴 프롬프트 검증 결과 확인** - 오버플로가 시작되는 길이가 2000자보다 짧으면, 맥 쪽에서 그 길이를 넘는 프롬프트는 로컬 인코더로 처리하는 규칙을 넣습니다.
3. **맥 쪽 클라이언트** (아직 만들지 않음, 제안 설계):
   - `backend/app/core/config.py`에 `REMOTE_ENCODER_URL`, `REMOTE_ENCODER_TOKEN`(`.env`). 둘 다 없으면 지금처럼 로컬 인코더를 씁니다(옵션 기능).
   - `ModelManager.encode_prompts`(`backend/app/utils/model_manager.py`)에서 캐시에 없는 프롬프트만 원격으로 요청합니다. 응답의 base64 fp16 텐서를 복원해 `(embeds, all-ones mask)`로 만들고 기존 프롬프트 임베딩 캐시에 넣습니다.
   - **폴백**: 연결 실패, 타임아웃, 503, 422(`non-finite` 포함), 임베딩에 NaN/inf → 그 프롬프트는 로컬 인코더로 처리하고 로그를 남깁니다.
   - 원격 인코더 사용 시 DiT를 배치 사이에도 상주시키는 최적화는 별도 작업입니다(현재는 `GenerationWorker`가 배치마다 로딩/해제).
   - 원격 인코더로 만든 이미지는 로컬 인코더와 달라지므로, 응답/DB에 `encoder` 필드(local/remote)를 남기는 것을 권장합니다.
4. **테스트**: 원격 정상, 원격 중단(폴백), NaN 응답(폴백), 토큰 오류, 요청 1개/여러 개, 서버 재시작. 이전 E2E 방식(실제 모델, `docs/E2E_TEST_REPORT.md`)으로 시간을 측정해 약 97초 → 약 65초가 실제로 나오는지 확인합니다.

## 7. 1060 컴퓨터용 프롬프트 (서버 만들기)

````text
목표: 기존에 측정이 끝난 "NF4 4비트 텍스트 인코더(Qwen-Image 2.1, GTX 1060 6GB)"를 HTTP 서비스로 만들어줘. 인코더를 시작 시 한 번만 VRAM에 올려 상주시키고, 다른 컴퓨터(Mac의 이미지 생성 서버)가 프롬프트를 보내면 임베딩을 돌려주는 단순 서비스다. 작업 위치는 ~/enc4bit_test(기존 venv, ~/enc4bit_test/measure_enc4bit.py, ~/enc4bit_test/hf_home 재사용). 이 컴퓨터에는 다른 서비스가 많이 돌고 있으니 그것들과 전역 환경·설정은 건드리지 마. 패키지를 새로 설치해야 하면(예: fastapi, uvicorn) 이름·버전·용량을 알려주고 확인받은 뒤 venv 안에만 설치해. systemd 등록·방화벽 변경은 하지 말고, 필요하면 파일만 만들어 두고 내가 하도록 안내해.

[이미 알려진 사실]
- 이전 측정에서 이 인코더는 4.94GiB VRAM(nvidia-smi 5235MiB, 여유 약 900MiB), 프롬프트당 0.45초, 로딩 36~40초였다. 비전 타워(`visual`)와 `lm_head`를 GPU에 올리지 않는 것만으로 들어갔다.
- diffusers는 import할 수 없어서, 설치된 diffusers 파일에서 `QwenImage21Pipeline.encode_prompt` 소스를 AST로 추출해 그대로 쓰고 있다(템플릿, 왼쪽 패딩, `_drop_idx=14`, 마스크 처리 포함). 이 방식과 기존 measure_enc4bit.py의 로딩·추출 코드를 그대로 재사용하고 다시 짜지 마.
- 이전 측정에서 내부 residual 값이 최대 13232까지 올라갔다(fp16 한계 65504). 아직 긴 프롬프트에서의 안전은 확인되지 않았다. 이 서비스는 NaN/inf를 반드시 감지해서 에러로 돌려줘야 한다.
- Mac 쪽 API는 프롬프트를 최대 2000자까지 받는다. 이 서비스도 2000자까지 받아야 한다.
- 이 컴퓨터는 RAM 16GB를 다른 서비스와 나눠 쓴다. 로딩 중 시스템 RAM(특히 safetensors mmap 페이지 캐시) 사용을 최소화해라(로딩 후 페이지 캐시 해제, 필요하면 순차 로딩). RAM 피크와 로딩 후 상주 RAM을 최종 보고에 꼭 넣어라.

[API 규약 - 반드시 그대로 구현]
1. `POST /encode`
   - 요청(JSON): `{"prompt": "<문자열, 1~2000자>"}`
   - 성공(200, JSON): `{"shape": [1, L, 4096], "dtype": "float16", "data": "<base64: 임베딩 텐서를 C 연속 순서의 raw float16 little-endian 바이트로 직렬화>", "max_abs": <float>, "elapsed_ms": <int>}`
     - 임베딩은 `encode_prompt`가 돌려준 값을 mask로 걸러내지 않은 원래 shape (1, L, 4096) 그대로, fp16으로 보낸다. 마스크는 보내지 않는다(Mac 쪽이 all-ones로 만든다). 만약 `encode_prompt`가 단일 프롬프트에 대해 마스크로 일부 토큰이 빠져야 하는 상황이면 그 사실을 보고하고 나에게 알려줘.
   - 오류: 422 `{"detail": "..."}` (프롬프트 길이 위반 또는 임베딩에 NaN/inf 포함 - 이때 detail에 "non-finite"를 포함), 413 (토큰 길이가 모델이 처리할 수 있는 한도 초과), 503 (CUDA OOM 등 일시적 오류, `Retry-After: 5` 헤더), 401 (토큰 불일치).
2. `GET /health` → `{"status": "ok", "model_loaded": true/false, "vram_used_mb": int, "vram_free_mb": int}` (인증 불필요). 모델 로딩이 끝나기 전에는 `model_loaded: false`이고 `/encode`는 503.
3. 인증: 환경변수 `ENCODER_TOKEN`이 설정돼 있으면 `/encode`는 `Authorization: Bearer <token>` 헤더가 필요하다. 설정돼 있지 않으면 시작 시 경고를 로그에 남긴다.

[구현 요건]
- FastAPI + uvicorn. 모델은 앱 시작(lifespan) 때 한 번 로딩하고 워밍업 한 번 실행. 서버는 `--host`/`--port`를 인자로 받고 기본은 127.0.0.1:9000. 내가 LAN 주소로 열 수 있게 한다.
- GPU 작업은 한 번에 1개만 실행(락 또는 단일 워커). 요청이 겹치면 순서대로 대기한다. 이벤트 루프를 막지 않도록 GPU 호출은 스레드 풀에서 실행.
- VRAM 여유가 약 900MiB뿐이다. 요청 처리 중 불필요한 텐서를 남기지 않고, `torch.no_grad()`와 `torch.inference_mode()`를 쓰고, 요청 후 `torch.cuda.empty_cache()`는 과하지 않게(예: 요청마다 하지 말고, 피크 VRAM이 올라갈 때만) 쓴다. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`를 실행 스크립트에 넣는다.
- CUDA OOM은 잡아서 503으로 돌려주고 서비스는 계속 살아 있어야 한다(OOM 후 `empty_cache()`).
- 로그에는 프롬프트 원문을 남기지 말고 길이(글자 수·토큰 수), 걸린 시간, max_abs, VRAM만 남긴다.
- 실행 스크립트 `run_encoder.sh`(venv 활성화, 환경변수, uvicorn 실행)를 만든다. systemd 유닛은 `enc4bit-server.service.example` 파일로만 만들어 두고 등록하지 마.
- 의존성은 `pip freeze` 결과를 `requirements.encoder.txt`로 저장.

[검증 - 모두 실제로 실행해서 결과를 보고]
1. 일치성: 서비스가 `/encode`로 돌려준 임베딩(base64 복원 후)이 기존 `~/enc4bit_test/embeds_nf4_gtx1060.pt`의 4개 프롬프트 임베딩과 일치하는지(최대 절댓값 차이, 비트 단위 일치 여부). 프롬프트는 이전 측정에 썼던 4개:
   - 'A storefront sign that reads "OPEN 24 HOURS" in glowing neon letters, rainy street at night, photo'
   - 'Close-up portrait of an elderly fisherman with deep wrinkles, natural window light, photorealistic'
   - 'A glass of water on a wooden table with refraction and caustics, macro photo'
   - '눈 내리는 서울 거리에서 한복을 입고 우산을 든 여성, 사진'
2. 긴 프롬프트: 영어와 한국어 각각 100 / 500 / 1000 / 2000자 프롬프트(의미 있는 문장을 이어 붙여 만들 것)로 `/encode`를 호출해서, 각각의 토큰 수, `max_abs`, NaN/inf 여부, 피크 VRAM(`nvidia-smi` 기준), 걸린 시간을 표로 보고. NaN/inf나 OOM이 시작되는 길이가 있으면 정확히 어느 구간부터인지 이진 탐색으로 찾아서 알려줘. 오버플로가 나면 원인이 되는 층과 값도 알려줘(고치는 건 하지 말고, 선택지만 제안: 예를 들어 일부 층만 fp32로 계산하기 등. 속도·VRAM 영향 추정 포함).
3. 동시 요청: 같은 서버에 동시에 5개 요청을 보내서 순서대로 처리되고 모두 성공하는지, 총 시간과 개별 대기 시간.
4. 오류 경로: 빈 프롬프트·2001자·잘못된 토큰·로딩 전 호출이 각각 위 규약의 코드(422/401/503)를 돌려주는지. 서버를 일부러 OOM 상태로 만들 필요는 없고, OOM 처리는 코드 리뷰로 설명해도 된다.
5. 안정성: 같은 서버에 50회 연속 요청을 보내고 VRAM이 늘어나지 않는지(요청 전·후 `nvidia-smi` 사용량).
6. 원격 확인용 `client_check.py`: 서버 주소와 토큰을 인자로 받아 `/health`와 위 4개 프롬프트 `/encode`를 호출하고 shape·`max_abs`·NaN 여부를 출력하는 단독 스크립트(표준 라이브러리 + numpy만 사용)를 만들어 줘. 나는 이걸 Mac에서 실행해 LAN 연결을 확인할 것이다.

[최종 보고]
- 실행 방법(명령 그대로), 필요한 환경변수, 서비스가 듣는 주소, 모든 파일 경로.
- 위 검증 1~6의 결과 요약표(실패·미확인 항목은 숨기지 말고 명시). RAM 피크와 로딩 후 상주 RAM.
- 긴 프롬프트에서 NaN/inf가 시작되는 길이(있다면)와 제안한 선택지.
- 사용한 정확한 버전(torch, transformers, bitsandbytes, fastapi, uvicorn, CUDA, 드라이버).
- 이 작업과 무관한 서비스·설정은 수정하지 않았다는 것 확인.
````

(위 프롬프트는 처음 제공했던 것에서 RAM 관련 두 항목만 추가했습니다. 보류 사유가 RAM이었기 때문입니다.)

## 8. 처음 측정용 프롬프트 요약 (이미 한 일)

1060 컴퓨터에서 "4비트로 6GB VRAM에 들어가는지"를 확인한 독립 스크립트 작업이었습니다. 요점: `encode_prompt` 소스 추출 재사용, 비전 타워·`lm_head` 제외, NF4 + fp16 계산, 4개 프롬프트 임베딩의 VRAM·시간·RAM·NaN/inf·shape 측정, 임베딩을 `.pt`로 저장. 위 4번의 환경·결과가 이 작업의 산출물입니다.
