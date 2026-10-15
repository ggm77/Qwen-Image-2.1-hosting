# 🚀 Qwen-Image-2.1-hosting

**FastAPI + React 기반의 Qwen-Image 2.1 모델 호스팅 프로젝트**

이 프로젝트는 Qwen-Image 2.1 기반 이미지 생성 기능을 실제로 활용하기 위해 만든 **Fullstack AI 이미지 생성 서비스**입니다.  
백엔드에서는 Qwen-Image 2.1 모델을 로딩하고 이미지 생성 API를 제공하며, 프론트엔드는 React 기반 웹 UI로 사용자가 직접 이미지를 생성할 수 있도록 구성됩니다.



## 🛠️ Tech Stack

### **Backend (API Server)**
- FastAPI  
- Python 3.11.13
- Diffusers (QwenImage21Pipeline, GitHub main)  
- Uvicorn  

### **Frontend (Web UI)**
- React (Vite)
- TypeScript
- Axios



## 📁 Project Structure
```
Qwen-Image-2.1-hosting/
├── backend/ # FastAPI 서버 (모델 로딩 & 이미지 생성 API)
│ ├── app/
│ └── requirements.txt
├── frontend/ # React 웹 UI
│ ├── src/
│ └── package.json
├── model/ # Qwen-Image 2.1 모델 위치
├── README.md
└── .gitignore
```



## 🚀 How to Run

### 0) 모델 다운로드 (약 30GB)
```bash
hf download Qwen/Qwen-Image-2.1 --local-dir model/Qwen-Image-2.1
```
> 24GB 통합 메모리에서는 텍스트 인코더(약 16GB)와 DiT(약 14GB)를 동시에 올릴 수 없어,
> 요청마다 `인코더 로딩 → 임베딩 → 해제 → DiT 로딩 → 생성` 순으로 처리합니다. (`backend/app/utils/model_manager.py`)
> 대기 중인 요청은 워커가 묶어서 처리하므로 인코더/DiT 로딩은 배치당 한 번씩만 일어납니다. (`backend/app/utils/job_queue.py`)

**정밀도 선택 (`precision`)**: `POST /api/v1/generate`의 `precision`으로 DiT 가중치 정밀도를 고를 수 있습니다 (웹 UI에도 선택 항목이 있습니다).
- `bf16` (기본): 원본 가중치, DiT 약 14GB
- `int8`: Linear 가중치를 출력 채널별 int8로 양자화(약 7GB), 계산은 bf16. 별도 변환 작업은 필요 없습니다.
  - 첫 int8 요청 때 원본을 양자화하고 결과를 `model/Qwen-Image-2.1-int8/transformer.safetensors`(약 7GB)에 저장합니다. 이후에는 이 파일에서 바로 로딩하므로 로딩이 빠르고(약 16초 → 5~6초) 로딩 중 메모리도 적게 씁니다(약 14.6GB → 2.4GB).
  - 원본 가중치가 바뀌었거나 캐시가 깨졌으면 자동으로 다시 만듭니다. 캐시 파일을 지워도 안전합니다.
- 24GB 메모리에서 768×768 이상은 bf16이 스왑으로 매우 느려질 수 있어 int8을 권장합니다. 512×512에서는 속도 차이가 작습니다.

**이미지 입력 (`images`)**: `POST /api/v1/generate`에 `images`(base64 문자열 또는 data URL 배열, 최대 4장)를 넣으면 입력 이미지를 참조하는 생성/편집으로 동작합니다. 프롬프트에서는 순서대로 이미지1, 이미지2로 가리킵니다.
- `height`/`width`는 둘 다 생략하면 마지막 입력 이미지의 종횡비를 따릅니다 (이미지가 없으면 768×768). 하나만 주면 422입니다.
- 입력 이미지는 `backend/app/input_images/`에 내용 해시 이름의 PNG로 저장되어, 서버를 재시작해도 대기 중인 작업을 이어서 처리합니다. `GET /api/v1/tasks/{task_id}/inputs/{index}`로 다시 받을 수 있고, 작업 응답의 `input_image_count`로 장수를 알 수 있습니다.
- 입력 이미지는 인코더/VAE에 넣기 전에 약 768² 면적으로 맞춥니다(`CONDITION_RESOLUTION`, `backend/app/core/config.py`). 24GB 메모리에서 스텝당 시간이 512: 약 8.5초 / 768: 약 13초 / 1024: 약 45초(스왑이 거의 가득 참)라서 모델 기본값(1024)보다 낮춰 두었습니다.
- 모델은 참조 이미지를 최대 10장까지 지원하지만, 장수에 비례해 토큰이 늘어나므로 24GB 환경에서는 4장으로 제한했습니다(`MAX_INPUT_IMAGES`). 검증은 1장 기준으로만 했습니다.

### 1) Backend
```bash
cd backend
python3.11 -m venv .venv
PYTHONPATH= .venv/bin/pip install -r requirements.txt
./run.sh          # 전역 PYTHONPATH를 비우고 venv로 실행 (--reload 등 인자 전달 가능)
```

### 2) Frontend
```bash
cd frontend
npm install
npm start      # 빌드 후 서빙 (사용용). 탭을 오가도 페이지가 새로고침되지 않는다.
# npm run dev  # 개발용 (HMR). 백그라운드에 있던 탭으로 돌아오면 페이지가 새로고침된다.
```
