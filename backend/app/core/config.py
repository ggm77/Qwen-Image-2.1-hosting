import os

class Settings:
    PROJECT_NAME: str = "Qwen-Image-2.1 Hosting"
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DB_PATH: str = os.path.join(BASE_DIR, "db", "app.db")
    MODEL_PATH: str = os.path.join(os.path.dirname(BASE_DIR), "..", "model", "Qwen-Image-2.1")
    # int8 양자화한 DiT 가중치 캐시 (첫 int8 요청 때 생성되고 이후 로딩에 재사용)
    INT8_CACHE_PATH: str = os.path.normpath(
        os.path.join(MODEL_PATH, "..", "Qwen-Image-2.1-int8", "transformer.safetensors")
    )
    IMG_DIR: str = os.path.join(BASE_DIR, "generated_images")
    # 요청에 첨부된 입력(조건) 이미지. 서버를 다시 시작해도 대기 중인 작업을 이어서 처리하려고 디스크에 보관한다.
    INPUT_IMG_DIR: str = os.path.join(BASE_DIR, "input_images")
    # 입력 이미지를 텍스트 인코더/VAE에 넣기 전에 약 이 값의 제곱 픽셀(종횡비 유지)로 맞추는 기준.
    # 인코더 임베딩과 생성 단계가 같은 값을 써야 하므로 한 곳에서만 정한다.
    # 모델 기본값은 1024지만, 24GB 메모리(int8, 896x672 출력, 이미지 1장)에서 스텝당 시간이
    # 512: 약 8.5초 / 768: 약 13초 / 1024: 약 45초(스왑이 거의 가득 참)로 늘어서 768로 둔다.
    CONDITION_RESOLUTION: int = 768

settings = Settings()

os.makedirs(settings.IMG_DIR, exist_ok=True)
os.makedirs(settings.INPUT_IMG_DIR, exist_ok=True)