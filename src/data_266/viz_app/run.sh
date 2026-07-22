#!/usr/bin/env bash
# run.sh — Patent Litigation Galaxy 실행 스크립트
#
# 사용법:
#   ./run.sh frontend   # Streamlit 대시보드만
#   ./run.sh backend    # FastAPI 서버만
#   ./run.sh all        # 둘 다 (백그라운드 FastAPI + 포어그라운드 Streamlit)

set -e
cd "$(dirname "$0")"   # viz_app/ 기준

case "${1:-frontend}" in

  frontend)
    echo "▶ Streamlit 기동 → http://localhost:8501"
    conda run -n patent_eval streamlit run frontend/app.py \
      --server.port 8501 \
      --server.headless true \
      --theme.base dark \
      --theme.backgroundColor "#050810" \
      --theme.primaryColor "#00FFFF"
    ;;

  xai)
    echo "▶ XAI 해석 앱 기동 → http://localhost:8502"
    conda run -n patent_eval streamlit run frontend/xai_app.py \
      --server.port 8502 \
      --server.headless true \
      --theme.base dark \
      --theme.backgroundColor "#0e1117" \
      --theme.primaryColor "#4fc3f7"
    ;;

  backend)
    echo "▶ FastAPI 기동 → http://localhost:8000/docs"
    uvicorn backend.app:app --reload --port 8000 --host 0.0.0.0
    ;;

  all)
    echo "▶ FastAPI 백그라운드 기동..."
    uvicorn backend.app:app --port 8000 --host 0.0.0.0 &
    BACKEND_PID=$!
    echo "  FastAPI PID: $BACKEND_PID"

    echo "▶ Streamlit 기동 → http://localhost:8501"
    streamlit run frontend/app.py \
      --server.port 8501 \
      --server.headless true \
      --theme.base dark \
      --theme.backgroundColor "#050810" \
      --theme.primaryColor "#00FFFF"

    # Streamlit 종료 시 FastAPI도 종료
    kill $BACKEND_PID 2>/dev/null || true
    ;;

  *)
    echo "사용법: $0 [frontend|backend|all]"
    exit 1
    ;;
esac
