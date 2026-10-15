import base64
import binascii
import hashlib
import os
from io import BytesIO

from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import settings
from app.schemas.generate.generate import MAX_IMAGE_SIZE, MIN_IMAGE_SIZE

# 디코딩 후 이미지 한 장의 최대 크기와 최대 화소 수 (압축 폭탄 방지)
MAX_INPUT_IMAGE_BYTES = 20 * 1024 * 1024
MAX_INPUT_IMAGE_PIXELS = 50_000_000

# 출력 크기를 요청이 정하지 않았을 때 쓰는 기준 면적 (기본 생성 크기 768x768과 같다)
DEFAULT_OUTPUT_AREA = 768 * 768

class InvalidImageError(ValueError):
    """첨부된 이미지를 사용할 수 없을 때 사용한다. 메시지는 그대로 사용자에게 보여줄 수 있다."""

def _decode_base64(data: str) -> bytes:
    if data.startswith("data:"):  # data URL이면 헤더를 떼어낸다
        _, _, data = data.partition(",")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise InvalidImageError("base64로 디코딩할 수 없습니다.")
    if len(raw) > MAX_INPUT_IMAGE_BYTES:
        raise InvalidImageError(f"이미지가 너무 큽니다. (최대 {MAX_INPUT_IMAGE_BYTES // 1024 // 1024}MB)")
    return raw

def store_input_image(data: str) -> tuple[str, tuple[int, int]]:
    """
    base64 이미지를 검증하고 PNG로 정규화해서 저장한다.
    파일 이름은 내용의 해시라서 같은 이미지는 한 번만 저장된다.
    반환값: (파일 이름, (너비, 높이))
    """
    raw = _decode_base64(data)

    try:
        with Image.open(BytesIO(raw)) as img:
            if img.width * img.height > MAX_INPUT_IMAGE_PIXELS:
                raise InvalidImageError(f"이미지 해상도가 너무 큽니다. (최대 {MAX_INPUT_IMAGE_PIXELS // 1_000_000}MP)")
            img.load()
            img = ImageOps.exif_transpose(img)  # 휴대폰 사진의 회전 정보를 픽셀에 반영
            has_alpha = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
            img = img.convert("RGBA" if has_alpha else "RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise InvalidImageError("이미지로 해석할 수 없습니다.")

    buf = BytesIO()
    img.save(buf, format="PNG")
    png = buf.getvalue()

    name = hashlib.sha256(png).hexdigest()[:32] + ".png"
    path = os.path.join(settings.INPUT_IMG_DIR, name)
    if not os.path.exists(path):
        tmp_path = path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(png)
        os.replace(tmp_path, path)  # 쓰다 중단된 파일이 남지 않도록 원자적으로 교체

    return name, img.size

def input_image_path(name: str) -> str:
    # 이름은 우리가 만든 해시 파일명만 허용한다 (경로 조작 방지)
    if os.path.basename(name) != name:
        raise ValueError(f"잘못된 입력 이미지 이름: {name}")
    return os.path.join(settings.INPUT_IMG_DIR, name)

def input_images_exist(names: list[str]) -> bool:
    return all(os.path.isfile(input_image_path(name)) for name in names)

def load_input_images(names: list[str]) -> list[Image.Image]:
    images = []
    for name in names:
        with Image.open(input_image_path(name)) as img:
            img.load()
            images.append(img.copy())
    return images

def default_output_size(reference: tuple[int, int]) -> tuple[int, int]:
    """참조 이미지의 종횡비를 따르는 출력 크기 (height, width). 32의 배수이고 허용 범위 안이다."""
    width, height = calculate_dimensions(DEFAULT_OUTPUT_AREA, reference[0] / reference[1])[:2]
    clamp = lambda v: max(MIN_IMAGE_SIZE, min(MAX_IMAGE_SIZE, v))
    return clamp(height), clamp(width)
