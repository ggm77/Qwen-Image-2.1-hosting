# 🚀 Qwen-Image-2.1-hosting

**FastAPI + React 기반의 Qwen-Image 2.1 모델 호스팅 프로젝트**

이 프로젝트는 FastAPI를 학습하면서 동시에 Qwen-Image 2.1 기반 이미지 생성 기능을 실제로 활용하기 위해 만든 **Fullstack AI 이미지 생성 서비스**입니다.  
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
- `int8`: 로딩 시 Linear 가중치를 출력 채널별 int8로 양자화(약 7GB), 계산은 bf16. 모델 파일은 그대로 쓰며 별도 변환이 필요 없습니다.
- 24GB 메모리에서 768×768 이상은 bf16이 스왑으로 매우 느려질 수 있어 int8을 권장합니다. 512×512에서는 속도 차이가 작습니다.

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
npm run dev
```
