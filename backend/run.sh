#!/bin/zsh
# 전역 PYTHONPATH(홈브루 패키지)가 venv를 오염시키지 않도록 비우고 venv의 uvicorn으로 실행
cd "$(dirname "$0")"
unset PYTHONPATH
exec .venv/bin/python -m uvicorn app.main:app --port 8000 "$@"
