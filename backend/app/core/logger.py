import logging
import sys

from uvicorn.logging import DefaultFormatter

LOG_LEVEL = logging.INFO

FORMAT = "%(levelprefix)s [%(asctime)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 루트 로거와 uvicorn 로거 한번에 세팅
def _configure_root_logger() -> logging.Logger:

    #루트 로거
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    #기존 핸들러 삭제 (중복 방지)
    if root_logger.handlers:
        root_logger.handlers.clear()

    #콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)

    formatter = DefaultFormatter(
        fmt=FORMAT,
        datefmt=DATE_FORMAT,
        use_colors=True,
    )
    console_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)

    #uvicorn 관련 로거들을 동일 핸들러 사용하도록 함
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        uv_logger.setLevel(LOG_LEVEL)
        uv_logger.handlers = [console_handler]
        uv_logger.propagate = False

    return root_logger

_root_logger = _configure_root_logger()

logger = logging.getLogger("app")