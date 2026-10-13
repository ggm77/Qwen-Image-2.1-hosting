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

settings = Settings()

os.makedirs(settings.IMG_DIR, exist_ok=True)